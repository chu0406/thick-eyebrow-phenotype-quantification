import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


# =========================
# 1. Path settings
# =========================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

MASK_CSV = BASE_DIR / "outputs" / "features" / "eyebrow_mask_features_v2.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
QC_DIR = OUTPUT_DIR / "qc" / "eyebrow_thickness_overlay_v2"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

SIDE_OUTPUT_CSV = FEATURE_DIR / "eyebrow_thickness_side_features.csv"
IMAGE_OUTPUT_CSV = FEATURE_DIR / "eyebrow_thickness_image_features.csv"


# =========================
# 2. Settings
# =========================

SAVE_QC_OVERLAY = True
MAX_QC_PER_SOURCE = 60

# PCA thickness settings
MIN_MASK_PIXELS = 8
MIN_PIXELS_PER_BIN = 3
N_BINS_DEFAULT = 30

# aligned eye distance target from Step 2:
# left eye = 0.35 * 512, right eye = 0.65 * 512
# eye distance = 153.6 px
DEFAULT_EYE_DISTANCE = 153.6


# =========================
# 3. Utility functions
# =========================

def make_safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = text.strip("_")
    return text[:160]


def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(str(path))
    return success


def is_success_value(x):
    return str(x).lower() in ["1", "true", "yes"]


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        if str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def safe_eye_distance(row):
    d = safe_float(row.get("aligned_eye_distance", np.nan), np.nan)
    if np.isfinite(d) and d > 1:
        return d

    d = safe_float(row.get("target_eye_distance", np.nan), np.nan)
    if np.isfinite(d) and d > 1:
        return d

    return DEFAULT_EYE_DISTANCE


def percentile_or_nan(values, q):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.percentile(values, q))


# =========================
# 4. PCA thickness
# =========================

def compute_pca_thickness(mask):
    """
    Estimate local eyebrow thickness by:
    1. PCA major axis = eyebrow long direction
    2. Split mask along major axis into bins
    3. In each bin, thickness = max projection on minor axis - min projection
    """
    mask_bool = mask > 0

    ys, xs = np.where(mask_bool)
    n_pixels = len(xs)

    if n_pixels < MIN_MASK_PIXELS:
        return {
            "pca_success": 0,
            "pca_failure_reason": "too_few_mask_pixels",
            "pca_angle_deg": np.nan,
            "pca_length": np.nan,
            "pca_global_width": np.nan,
            "pca_thickness_mean": np.nan,
            "pca_thickness_median": np.nan,
            "pca_thickness_p90": np.nan,
            "pca_thickness_p95": np.nan,
            "pca_thickness_max": np.nan,
            "pca_valid_bins": 0,
            "pca_segments": [],
            "centroid": None,
            "major_axis": None,
            "minor_axis": None,
        }

    coords = np.column_stack([xs, ys]).astype(np.float32)
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    cov = np.cov(centered.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except Exception:
        return {
            "pca_success": 0,
            "pca_failure_reason": "pca_failed",
            "pca_angle_deg": np.nan,
            "pca_length": np.nan,
            "pca_global_width": np.nan,
            "pca_thickness_mean": np.nan,
            "pca_thickness_median": np.nan,
            "pca_thickness_p90": np.nan,
            "pca_thickness_p95": np.nan,
            "pca_thickness_max": np.nan,
            "pca_valid_bins": 0,
            "pca_segments": [],
            "centroid": None,
            "major_axis": None,
            "minor_axis": None,
        }

    order = np.argsort(eigvals)[::-1]
    major_axis = eigvecs[:, order[0]]
    minor_axis = eigvecs[:, order[1]]

    # make direction stable left-to-right
    if major_axis[0] < 0:
        major_axis = -major_axis

    major_proj = centered @ major_axis
    minor_proj = centered @ minor_axis

    major_min = float(np.min(major_proj))
    major_max = float(np.max(major_proj))
    minor_min = float(np.min(minor_proj))
    minor_max = float(np.max(minor_proj))

    pca_length = major_max - major_min + 1.0
    pca_global_width = minor_max - minor_min + 1.0

    if pca_length < 1:
        n_bins = 1
    else:
        n_bins = int(min(N_BINS_DEFAULT, max(8, pca_length / 4)))

    edges = np.linspace(major_min, major_max, n_bins + 1)

    thickness_values = []
    segments = []

    for b in range(n_bins):
        lo = edges[b]
        hi = edges[b + 1]

        if b == n_bins - 1:
            idx = (major_proj >= lo) & (major_proj <= hi)
        else:
            idx = (major_proj >= lo) & (major_proj < hi)

        if np.sum(idx) < MIN_PIXELS_PER_BIN:
            continue

        local_minor = minor_proj[idx]
        local_major = major_proj[idx]

        local_thickness = float(np.max(local_minor) - np.min(local_minor) + 1.0)

        if local_thickness <= 0:
            continue

        thickness_values.append(local_thickness)

        # segment for QC overlay
        major_center = float(np.median(local_major))
        minor_a = float(np.min(local_minor))
        minor_b = float(np.max(local_minor))

        p1 = centroid + major_center * major_axis + minor_a * minor_axis
        p2 = centroid + major_center * major_axis + minor_b * minor_axis

        segments.append((p1, p2))

    thickness_values = np.asarray(thickness_values, dtype=np.float32)

    if len(thickness_values) == 0:
        return {
            "pca_success": 0,
            "pca_failure_reason": "no_valid_bins",
            "pca_angle_deg": np.nan,
            "pca_length": pca_length,
            "pca_global_width": pca_global_width,
            "pca_thickness_mean": np.nan,
            "pca_thickness_median": np.nan,
            "pca_thickness_p90": np.nan,
            "pca_thickness_p95": np.nan,
            "pca_thickness_max": np.nan,
            "pca_valid_bins": 0,
            "pca_segments": [],
            "centroid": centroid,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
        }

    angle_deg = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))

    return {
        "pca_success": 1,
        "pca_failure_reason": "",
        "pca_angle_deg": angle_deg,
        "pca_length": float(pca_length),
        "pca_global_width": float(pca_global_width),
        "pca_thickness_mean": float(np.mean(thickness_values)),
        "pca_thickness_median": float(np.median(thickness_values)),
        "pca_thickness_p90": percentile_or_nan(thickness_values, 90),
        "pca_thickness_p95": percentile_or_nan(thickness_values, 95),
        "pca_thickness_max": float(np.max(thickness_values)),
        "pca_valid_bins": int(len(thickness_values)),
        "pca_segments": segments,
        "centroid": centroid,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
    }


# =========================
# 5. Skeleton thickness
# =========================

def compute_skeleton_thickness(mask):
    """
    Estimate thickness using distance transform on skeleton pixels.
    thickness = 2 * distance-to-background
    """
    binary = (mask > 0).astype(np.uint8) * 255

    if np.sum(binary > 0) < MIN_MASK_PIXELS:
        return {
            "skeleton_success": 0,
            "skeleton_failure_reason": "too_few_mask_pixels",
            "skeleton_thickness_mean": np.nan,
            "skeleton_thickness_median": np.nan,
            "skeleton_thickness_p90": np.nan,
            "skeleton_thickness_p95": np.nan,
            "skeleton_thickness_max": np.nan,
            "skeleton_pixels": 0,
            "skeleton_mask": np.zeros_like(binary),
        }

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    try:
        skeleton = cv2.ximgproc.thinning(binary)
    except Exception:
        return {
            "skeleton_success": 0,
            "skeleton_failure_reason": "cv2_ximgproc_thinning_not_available",
            "skeleton_thickness_mean": np.nan,
            "skeleton_thickness_median": np.nan,
            "skeleton_thickness_p90": np.nan,
            "skeleton_thickness_p95": np.nan,
            "skeleton_thickness_max": np.nan,
            "skeleton_pixels": 0,
            "skeleton_mask": np.zeros_like(binary),
        }

    skel_bool = skeleton > 0
    values = 2.0 * dist[skel_bool]
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if len(values) == 0:
        return {
            "skeleton_success": 0,
            "skeleton_failure_reason": "no_skeleton_values",
            "skeleton_thickness_mean": np.nan,
            "skeleton_thickness_median": np.nan,
            "skeleton_thickness_p90": np.nan,
            "skeleton_thickness_p95": np.nan,
            "skeleton_thickness_max": np.nan,
            "skeleton_pixels": 0,
            "skeleton_mask": skeleton,
        }

    return {
        "skeleton_success": 1,
        "skeleton_failure_reason": "",
        "skeleton_thickness_mean": float(np.mean(values)),
        "skeleton_thickness_median": float(np.median(values)),
        "skeleton_thickness_p90": percentile_or_nan(values, 90),
        "skeleton_thickness_p95": percentile_or_nan(values, 95),
        "skeleton_thickness_max": float(np.max(values)),
        "skeleton_pixels": int(len(values)),
        "skeleton_mask": skeleton,
    }


# =========================
# 6. QC overlay
# =========================

def draw_thickness_overlay(roi_bgr, mask, pca_result, skeleton_mask=None):
    overlay = roi_bgr.copy()

    # mask green overlay
    mask_bool = mask > 0
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255

    overlay[mask_bool] = cv2.addWeighted(
        overlay[mask_bool],
        0.45,
        green[mask_bool],
        0.55,
        0
    )

    # skeleton yellow
    if skeleton_mask is not None:
        skel_bool = skeleton_mask > 0
        overlay[skel_bool] = (0, 255, 255)

    # PCA major axis red
    if pca_result.get("centroid") is not None and pca_result.get("major_axis") is not None:
        centroid = pca_result["centroid"]
        major_axis = pca_result["major_axis"]
        length = pca_result.get("pca_length", 0)

        if np.isfinite(length) and length > 0:
            p1 = centroid - 0.5 * length * major_axis
            p2 = centroid + 0.5 * length * major_axis
            cv2.line(
                overlay,
                tuple(np.round(p1).astype(int)),
                tuple(np.round(p2).astype(int)),
                (0, 0, 255),
                1
            )

    # local thickness segments blue
    segments = pca_result.get("pca_segments", [])
    if len(segments) > 0:
        step = max(1, len(segments) // 12)
        for p1, p2 in segments[::step]:
            cv2.line(
                overlay,
                tuple(np.round(p1).astype(int)),
                tuple(np.round(p2).astype(int)),
                (255, 0, 0),
                1
            )

    return overlay


# =========================
# 7. Load mask results
# =========================

if not MASK_CSV.exists():
    raise FileNotFoundError(f"Cannot find mask csv: {MASK_CSV}")

df = pd.read_csv(MASK_CSV)
df = df.fillna("")

df = df[df["mask_success"].apply(is_success_value)].copy()
df = df.reset_index(drop=True)

print("Side-level masks to measure thickness:")
print(df.groupby(["source", "label", "side"]).size())


# =========================
# 8. Run thickness measurement
# =========================

records = []

qc_counter = {
    "GMDB": 0,
    "UTKFace": 0
}

for i, row in tqdm(df.iterrows(), total=len(df)):
    source = row["source"]
    label = row["label"]
    side = row["side"]
    filename_stem = Path(row["filename"]).stem

    mask_path = Path(row["mask_path"])
    roi_path = Path(row["side_roi_path"])

    eye_distance = safe_eye_distance(row)

    safe_stem = make_safe_name(
        f"{i:06d}_{source}_{label}_{filename_stem}_{side}"
    )

    source_qc_dir = QC_DIR / source
    source_qc_dir.mkdir(parents=True, exist_ok=True)
    qc_path = source_qc_dir / f"{safe_stem}_thickness_overlay.jpg"

    record = row.to_dict()
    record.update({
        "thickness_success": 0,
        "thickness_failure_reason": "",
        "eye_distance_for_norm": eye_distance,
        "thickness_qc_path": "",
    })

    try:
        mask = imread_unicode(mask_path, flags=cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError("failed_to_read_mask")

        mask = (mask > 0).astype(np.uint8) * 255

        pca_result = compute_pca_thickness(mask)
        skeleton_result = compute_skeleton_thickness(mask)

        # remove non-scalar objects before saving
        pca_segments = pca_result.pop("pca_segments", None)
        centroid = pca_result.pop("centroid", None)
        major_axis = pca_result.pop("major_axis", None)
        minor_axis = pca_result.pop("minor_axis", None)

        skeleton_mask = skeleton_result.pop("skeleton_mask", None)

        record.update(pca_result)
        record.update(skeleton_result)

        # normalized thickness
        for col in [
            "pca_length",
            "pca_global_width",
            "pca_thickness_mean",
            "pca_thickness_median",
            "pca_thickness_p90",
            "pca_thickness_p95",
            "pca_thickness_max",
            "skeleton_thickness_mean",
            "skeleton_thickness_median",
            "skeleton_thickness_p90",
            "skeleton_thickness_p95",
            "skeleton_thickness_max",
        ]:
            value = safe_float(record.get(col, np.nan), np.nan)
            if np.isfinite(value) and eye_distance > 0:
                record[col + "_norm"] = value / eye_distance
            else:
                record[col + "_norm"] = np.nan

        record["thickness_success"] = int(
            record.get("pca_success", 0) == 1 or record.get("skeleton_success", 0) == 1
        )

        if record["thickness_success"] == 0:
            record["thickness_failure_reason"] = (
                f"pca={record.get('pca_failure_reason', '')}; "
                f"skeleton={record.get('skeleton_failure_reason', '')}"
            )

        # Restore objects for QC drawing
        pca_for_qc = dict(pca_result)
        pca_for_qc["pca_segments"] = pca_segments if pca_segments is not None else []
        pca_for_qc["centroid"] = centroid
        pca_for_qc["major_axis"] = major_axis
        pca_for_qc["minor_axis"] = minor_axis

        if SAVE_QC_OVERLAY and qc_counter.get(source, 0) < MAX_QC_PER_SOURCE:
            roi_bgr = imread_unicode(roi_path, flags=cv2.IMREAD_COLOR)
            if roi_bgr is not None:
                overlay = draw_thickness_overlay(
                    roi_bgr=roi_bgr,
                    mask=mask,
                    pca_result=pca_for_qc,
                    skeleton_mask=skeleton_mask
                )
                imwrite_unicode(qc_path, overlay)
                record["thickness_qc_path"] = str(qc_path)
                qc_counter[source] = qc_counter.get(source, 0) + 1

    except Exception as e:
        record["thickness_success"] = 0
        record["thickness_failure_reason"] = str(e)

    records.append(record)


side_df = pd.DataFrame(records)
side_df.to_csv(SIDE_OUTPUT_CSV, index=False)

print("\nDone.")
print(f"Saved side-level thickness features to: {SIDE_OUTPUT_CSV}")

print("\nSide-level thickness summary:")
print(side_df.groupby(["source", "label", "side"])["thickness_success"].agg(["count", "sum"]))

failed_df = side_df[side_df["thickness_success"] == 0]
print(f"\nFailed thickness cases: {len(failed_df)}")

if len(failed_df) > 0:
    fail_path = FEATURE_DIR / "eyebrow_thickness_failed.csv"
    failed_df.to_csv(fail_path, index=False)
    print(f"Saved failed thickness cases to: {fail_path}")
    print("\nFailure reasons:")
    print(failed_df["thickness_failure_reason"].value_counts())


# =========================
# 9. Aggregate left/right to image-level
# =========================

valid_df = side_df[side_df["thickness_success"] == 1].copy()

key_cols = [
    "image_path",
    "filename",
    "image_id",
    "source",
    "label",
    "is_control",
    "patient_id",
    "age_year",
    "gender",
    "ethnicity",
    "omim_ids",
]

key_cols = [c for c in key_cols if c in valid_df.columns]

side_feature_cols = [
    "mask_area_ratio",
    "mask_tube_ratio",
    "pca_length_norm",
    "pca_global_width_norm",
    "pca_thickness_mean_norm",
    "pca_thickness_median_norm",
    "pca_thickness_p90_norm",
    "pca_thickness_p95_norm",
    "pca_thickness_max_norm",
    "skeleton_thickness_mean_norm",
    "skeleton_thickness_median_norm",
    "skeleton_thickness_p90_norm",
    "skeleton_thickness_p95_norm",
    "skeleton_thickness_max_norm",
]

side_feature_cols = [c for c in side_feature_cols if c in valid_df.columns]

image_records = []

for _, group in valid_df.groupby(key_cols, dropna=False):
    base = group.iloc[0][key_cols].to_dict()

    left = group[group["side"] == "left"]
    right = group[group["side"] == "right"]

    base["num_valid_sides"] = len(group)

    for col in side_feature_cols:
        vals = pd.to_numeric(group[col], errors="coerce").dropna().values

        if len(vals) > 0:
            base[col + "_lr_mean"] = float(np.mean(vals))
            base[col + "_lr_max"] = float(np.max(vals))
            base[col + "_lr_min"] = float(np.min(vals))
        else:
            base[col + "_lr_mean"] = np.nan
            base[col + "_lr_max"] = np.nan
            base[col + "_lr_min"] = np.nan

        if len(left) > 0:
            base["left_" + col] = safe_float(left.iloc[0].get(col, np.nan), np.nan)
        else:
            base["left_" + col] = np.nan

        if len(right) > 0:
            base["right_" + col] = safe_float(right.iloc[0].get(col, np.nan), np.nan)
        else:
            base["right_" + col] = np.nan

        # asymmetry = absolute difference between left and right
        lv = base["left_" + col]
        rv = base["right_" + col]
        if np.isfinite(lv) and np.isfinite(rv):
            base[col + "_lr_absdiff"] = abs(lv - rv)
        else:
            base[col + "_lr_absdiff"] = np.nan

    image_records.append(base)

image_df = pd.DataFrame(image_records)
image_df.to_csv(IMAGE_OUTPUT_CSV, index=False)

print(f"\nSaved image-level thickness features to: {IMAGE_OUTPUT_CSV}")

print("\nImage-level sample count:")
print(image_df.groupby(["source", "label"]).size())

main_col = "pca_thickness_p95_norm_lr_mean"

if main_col in image_df.columns:
    print(f"\nMain feature summary: {main_col}")
    print(image_df.groupby(["source", "label"])[main_col].describe())

    control = image_df[image_df["source"] == "UTKFace"][main_col].dropna()
    gmdb = image_df[image_df["source"] == "GMDB"][main_col].dropna()

    if len(control) > 0:
        thr95 = float(np.percentile(control, 95))
        thr90 = float(np.percentile(control, 90))

        print("\nControl-based thresholds:")
        print(f"UTKFace 90th percentile: {thr90:.6f}")
        print(f"UTKFace 95th percentile: {thr95:.6f}")

        if len(gmdb) > 0:
            gmdb_over90 = float(np.mean(gmdb > thr90))
            gmdb_over95 = float(np.mean(gmdb > thr95))

            print("\nGMDB thick eyebrow over control threshold:")
            print(f"GMDB > UTKFace P90: {gmdb_over90:.3f}")
            print(f"GMDB > UTKFace P95: {gmdb_over95:.3f}")