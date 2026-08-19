import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from scipy.stats import mannwhitneyu


# =========================
# 1. Path settings
# =========================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

RESULT_DIR = BASE_DIR / "outputs"

SIDE_CSV = RESULT_DIR / "features" / "eyebrow_thickness_side_features.csv"

FEATURE_DIR = RESULT_DIR / "features"
TABLE_DIR = RESULT_DIR / "tables"

SIDE_OUT = FEATURE_DIR / "eyebrow_local_density_side_features.csv"
IMAGE_OUT = FEATURE_DIR / "eyebrow_local_density_image_features.csv"
SUMMARY_OUT = TABLE_DIR / "local_density_group_comparison.csv"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Settings
# =========================

N_SEGMENTS = 24
MIN_TUBE_PIXELS_PER_SEGMENT = 8

TUBE_DILATE_X_RATIO = 0.10
TUBE_DILATE_Y_RATIO = 0.42
MIN_TUBE_DILATE_X = 5
MIN_TUBE_DILATE_Y = 7


# =========================
# 3. MediaPipe eyebrow landmark indices
# =========================

IMAGE_LEFT_EYEBROW_IDXS = [
    70, 63, 105, 66, 107,
    55, 65, 52, 53, 46
]

IMAGE_RIGHT_EYEBROW_IDXS = [
    336, 296, 334, 293, 300,
    285, 295, 282, 283, 276
]


# =========================
# 4. Utility functions
# =========================

def resolve_path(path_str):
    """
    If paths inside CSV still point to /outputs, redirect them to RESULT_DIR.
    """
    p = Path(str(path_str))

    if p.exists():
        return p

    old_prefix = str(BASE_DIR / "outputs")
    new_prefix = str(RESULT_DIR)

    s = str(p)
    if s.startswith(old_prefix):
        candidate = Path(new_prefix + s[len(old_prefix):])
        if candidate.exists():
            return candidate

    return p


def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def make_odd(x):
    x = int(round(x))
    x = max(3, x)
    if x % 2 == 0:
        x += 1
    return x


def get_side_eyebrow_indices(side):
    if side == "left":
        return IMAGE_LEFT_EYEBROW_IDXS
    elif side == "right":
        return IMAGE_RIGHT_EYEBROW_IDXS
    else:
        raise ValueError(f"unknown side: {side}")


def get_roi_box_from_row(row, side):
    if side == "left":
        return (
            int(row["left_x1"]),
            int(row["left_y1"]),
            int(row["left_x2"]),
            int(row["left_y2"]),
        )
    else:
        return (
            int(row["right_x1"]),
            int(row["right_y1"]),
            int(row["right_x2"]),
            int(row["right_y2"]),
        )


def get_eyebrow_points_in_roi(landmarks, roi_box, side):
    x1, y1, x2, y2 = roi_box
    idxs = get_side_eyebrow_indices(side)

    pts = landmarks[idxs, :2].copy()
    pts[:, 0] -= x1
    pts[:, 1] -= y1

    return pts.astype(np.float32)


def build_landmark_tube_mask(roi_shape, eyebrow_pts):
    """
    Reconstruct eyebrow tube from landmarks.
    Same idea as mask v2.
    """
    h, w = roi_shape[:2]

    pts = eyebrow_pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    hull = cv2.convexHull(pts.astype(np.int32))

    core = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(core, hull, 255)

    brow_x_min = float(np.min(pts[:, 0]))
    brow_x_max = float(np.max(pts[:, 0]))
    brow_y_min = float(np.min(pts[:, 1]))
    brow_y_max = float(np.max(pts[:, 1]))

    brow_w = max(1.0, brow_x_max - brow_x_min)
    brow_h = max(1.0, brow_y_max - brow_y_min)

    dilate_x = max(MIN_TUBE_DILATE_X, int(round(brow_w * TUBE_DILATE_X_RATIO)))
    dilate_y = max(MIN_TUBE_DILATE_Y, int(round(brow_h * TUBE_DILATE_Y_RATIO)))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (make_odd(dilate_x), make_odd(dilate_y))
    )

    tube = cv2.dilate(core, kernel, iterations=1)

    return tube


def percentile_or_nan(values, q):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, q))


def summarize_feature(df, feature):
    gmdb = pd.to_numeric(df.loc[df["source"] == "GMDB", feature], errors="coerce").dropna()
    utk = pd.to_numeric(df.loc[df["source"] == "UTKFace", feature], errors="coerce").dropna()

    if len(gmdb) == 0 or len(utk) == 0:
        return None

    try:
        _, p_value = mannwhitneyu(gmdb, utk, alternative="two-sided")
    except Exception:
        p_value = np.nan

    return {
        "feature": feature,
        "gmdb_n": len(gmdb),
        "utk_n": len(utk),
        "gmdb_mean": float(np.mean(gmdb)),
        "utk_mean": float(np.mean(utk)),
        "mean_diff_gmdb_minus_utk": float(np.mean(gmdb) - np.mean(utk)),
        "gmdb_median": float(np.median(gmdb)),
        "utk_median": float(np.median(utk)),
        "median_diff_gmdb_minus_utk": float(np.median(gmdb) - np.median(utk)),
        "gmdb_std": float(np.std(gmdb, ddof=1)),
        "utk_std": float(np.std(utk, ddof=1)),
        "mannwhitney_p": float(p_value),
    }


def compute_local_density(mask, tube_mask, eyebrow_pts):
    """
    Segment eyebrow tube along eyebrow direction and compute local filling ratio.
    """
    mask_bool = mask > 0
    tube_bool = tube_mask > 0

    if np.sum(tube_bool) == 0:
        return {
            "local_density_success": 0,
            "local_density_failure_reason": "empty_tube",
        }

    # PCA direction from eyebrow landmark points
    pts = eyebrow_pts.astype(np.float32)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    try:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        major_axis = eigvecs[:, order[0]]
    except Exception:
        return {
            "local_density_success": 0,
            "local_density_failure_reason": "pca_failed",
        }

    if major_axis[0] < 0:
        major_axis = -major_axis

    ys, xs = np.where(tube_bool)
    coords = np.column_stack([xs, ys]).astype(np.float32)

    proj = (coords - centroid) @ major_axis

    p_min = float(np.min(proj))
    p_max = float(np.max(proj))

    if p_max <= p_min:
        return {
            "local_density_success": 0,
            "local_density_failure_reason": "invalid_projection_range",
        }

    edges = np.linspace(p_min, p_max, N_SEGMENTS + 1)

    local_ratios = []
    local_tube_areas = []
    local_mask_areas = []

    for i in range(N_SEGMENTS):
        lo = edges[i]
        hi = edges[i + 1]

        if i == N_SEGMENTS - 1:
            seg_idx = (proj >= lo) & (proj <= hi)
        else:
            seg_idx = (proj >= lo) & (proj < hi)

        if np.sum(seg_idx) < MIN_TUBE_PIXELS_PER_SEGMENT:
            continue

        seg_coords = coords[seg_idx].astype(int)
        seg_x = seg_coords[:, 0]
        seg_y = seg_coords[:, 1]

        tube_area = len(seg_x)
        mask_area = int(np.sum(mask_bool[seg_y, seg_x]))

        ratio = mask_area / float(tube_area)

        local_ratios.append(ratio)
        local_tube_areas.append(tube_area)
        local_mask_areas.append(mask_area)

    local_ratios = np.asarray(local_ratios, dtype=np.float32)

    if len(local_ratios) == 0:
        return {
            "local_density_success": 0,
            "local_density_failure_reason": "no_valid_segments",
        }

    mean_v = float(np.mean(local_ratios))
    std_v = float(np.std(local_ratios, ddof=1)) if len(local_ratios) > 1 else 0.0

    return {
        "local_density_success": 1,
        "local_density_failure_reason": "",
        "local_filling_mean": mean_v,
        "local_filling_median": float(np.median(local_ratios)),
        "local_filling_p75": percentile_or_nan(local_ratios, 75),
        "local_filling_p90": percentile_or_nan(local_ratios, 90),
        "local_filling_p95": percentile_or_nan(local_ratios, 95),
        "local_filling_max": float(np.max(local_ratios)),
        "local_filling_min": float(np.min(local_ratios)),
        "local_filling_std": std_v,
        "local_filling_cv": float(std_v / mean_v) if mean_v > 1e-8 else np.nan,
        "local_valid_segments": int(len(local_ratios)),
        "local_total_tube_area": int(np.sum(local_tube_areas)),
        "local_total_mask_area": int(np.sum(local_mask_areas)),
    }


# =========================
# 5. Load side features
# =========================

if not SIDE_CSV.exists():
    raise FileNotFoundError(f"Cannot find side CSV: {SIDE_CSV}")

side_df = pd.read_csv(SIDE_CSV).fillna("")

print("Loaded side-level features:")
print(side_df.groupby(["source", "label", "side"]).size())


# =========================
# 6. Compute local density
# =========================

records = []

for _, row in tqdm(side_df.iterrows(), total=len(side_df)):
    record = row.to_dict()

    side = row["side"]

    try:
        mask_path = resolve_path(row["mask_path"])
        landmark_path = resolve_path(row["aligned_landmark_path"])

        mask = imread_unicode(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("failed_to_read_mask")

        mask = (mask > 0).astype(np.uint8) * 255

        landmarks = np.load(landmark_path)

        roi_box = get_roi_box_from_row(row, side)

        eyebrow_pts = get_eyebrow_points_in_roi(
            landmarks=landmarks,
            roi_box=roi_box,
            side=side
        )

        tube_mask = build_landmark_tube_mask(mask.shape, eyebrow_pts)

        result = compute_local_density(
            mask=mask,
            tube_mask=tube_mask,
            eyebrow_pts=eyebrow_pts
        )

        record.update(result)

    except Exception as e:
        record.update({
            "local_density_success": 0,
            "local_density_failure_reason": str(e),
        })

    records.append(record)

side_out_df = pd.DataFrame(records)
side_out_df.to_csv(SIDE_OUT, index=False)

print(f"\nSaved side-level local density features to:\n{SIDE_OUT}")

print("\nLocal density side-level summary:")
print(side_out_df.groupby(["source", "label", "side"])["local_density_success"].agg(["count", "sum"]))


# =========================
# 7. Aggregate left/right to image-level
# =========================

valid_df = side_out_df[side_out_df["local_density_success"] == 1].copy()

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

local_cols = [
    "local_filling_mean",
    "local_filling_median",
    "local_filling_p75",
    "local_filling_p90",
    "local_filling_p95",
    "local_filling_max",
    "local_filling_min",
    "local_filling_std",
    "local_filling_cv",
    "local_valid_segments",
    "mask_tube_ratio",
    "mask_area_ratio",
]

local_cols = [c for c in local_cols if c in valid_df.columns]

image_records = []

for _, group in valid_df.groupby(key_cols, dropna=False):
    base = group.iloc[0][key_cols].to_dict()
    base["num_valid_sides_for_local_density"] = len(group)

    left = group[group["side"] == "left"]
    right = group[group["side"] == "right"]

    for col in local_cols:
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
            base["left_" + col] = float(pd.to_numeric(left.iloc[0][col], errors="coerce"))
        else:
            base["left_" + col] = np.nan

        if len(right) > 0:
            base["right_" + col] = float(pd.to_numeric(right.iloc[0][col], errors="coerce"))
        else:
            base["right_" + col] = np.nan

        lv = base["left_" + col]
        rv = base["right_" + col]

        if np.isfinite(lv) and np.isfinite(rv):
            base[col + "_lr_absdiff"] = abs(lv - rv)
        else:
            base[col + "_lr_absdiff"] = np.nan

    image_records.append(base)

image_out_df = pd.DataFrame(image_records)
image_out_df.to_csv(IMAGE_OUT, index=False)

print(f"\nSaved image-level local density features to:\n{IMAGE_OUT}")

print("\nImage-level sample count:")
print(image_out_df.groupby(["source", "label"]).size())


# =========================
# 8. Group comparison
# =========================

summary_features = [
    "local_filling_mean_lr_mean",
    "local_filling_median_lr_mean",
    "local_filling_p75_lr_mean",
    "local_filling_p90_lr_mean",
    "local_filling_p95_lr_mean",
    "local_filling_max_lr_mean",
    "local_filling_std_lr_mean",
    "local_filling_cv_lr_mean",
    "mask_tube_ratio_lr_mean",
    "mask_area_ratio_lr_mean",
]

summary_records = []

for feature in summary_features:
    if feature not in image_out_df.columns:
        continue

    result = summarize_feature(image_out_df, feature)
    if result is not None:
        summary_records.append(result)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(SUMMARY_OUT, index=False)

print(f"\nSaved local density group comparison to:\n{SUMMARY_OUT}")

print("\nLocal density comparison:")
print(summary_df.to_string(index=False))