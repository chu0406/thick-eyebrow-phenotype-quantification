import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mannwhitneyu


# =========================
# 1. Path settings
# =========================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

# Default: clean control result
if len(sys.argv) >= 2:
    RESULT_DIR = BASE_DIR / sys.argv[1]
else:
    RESULT_DIR = BASE_DIR / "outputs"

SIDE_CSV = RESULT_DIR / "features" / "eyebrow_thickness_side_features.csv"

FEATURE_DIR = RESULT_DIR / "features"
TABLE_DIR = RESULT_DIR / "tables"
PLOT_DIR = RESULT_DIR / "plots" / "eyebrow_length"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SIDE_OUT = FEATURE_DIR / "eyebrow_length_side_features.csv"
IMAGE_OUT = FEATURE_DIR / "eyebrow_length_image_features.csv"
SUMMARY_OUT = TABLE_DIR / "eyebrow_length_group_comparison.csv"


# =========================
# 2. Landmark settings
# =========================

LEFT_EYEBROW_IDXS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_IDXS = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

LEFT_EYE_CENTER_IDXS = [33, 133, 159, 145]
RIGHT_EYE_CENTER_IDXS = [362, 263, 386, 374]

LEFT_EYE_WIDTH_PAIR = (33, 133)
RIGHT_EYE_WIDTH_PAIR = (362, 263)

FACE_WIDTH_PAIR = (234, 454)

FACE_OVAL_IDXS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


# =========================
# 3. Utility functions
# =========================

def resolve_path(path_str):
    p = Path(str(path_str))
    if p.exists():
        return p

    # If CSV path still points to /outputs, redirect to RESULT_DIR
    old_prefix = str(BASE_DIR / "outputs")
    new_prefix = str(RESULT_DIR)

    s = str(p)
    if s.startswith(old_prefix):
        candidate = Path(new_prefix + s[len(old_prefix):])
        if candidate.exists():
            return candidate

    return p


def imread_unicode(path: Path, flags=cv2.IMREAD_GRAYSCALE):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def dist(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def get_side_eyebrow_indices(side):
    if side == "left":
        return LEFT_EYEBROW_IDXS
    elif side == "right":
        return RIGHT_EYEBROW_IDXS
    else:
        raise ValueError(f"Unknown side: {side}")


def get_roi_box(row, side):
    if side == "left":
        return int(row["left_x1"]), int(row["left_y1"]), int(row["left_x2"]), int(row["left_y2"])
    else:
        return int(row["right_x1"]), int(row["right_y1"]), int(row["right_x2"]), int(row["right_y2"])


def compute_scale_features(landmarks):
    xy = landmarks[:, :2].astype(np.float32)

    left_eye_center = xy[LEFT_EYE_CENTER_IDXS].mean(axis=0)
    right_eye_center = xy[RIGHT_EYE_CENTER_IDXS].mean(axis=0)

    interocular = dist(left_eye_center, right_eye_center)

    left_eye_width = dist(xy[LEFT_EYE_WIDTH_PAIR[0]], xy[LEFT_EYE_WIDTH_PAIR[1]])
    right_eye_width = dist(xy[RIGHT_EYE_WIDTH_PAIR[0]], xy[RIGHT_EYE_WIDTH_PAIR[1]])
    eye_width_mean = (left_eye_width + right_eye_width) / 2.0

    face_width_cheek = dist(xy[FACE_WIDTH_PAIR[0]], xy[FACE_WIDTH_PAIR[1]])

    oval = xy[FACE_OVAL_IDXS]
    face_oval_width = float(np.max(oval[:, 0]) - np.min(oval[:, 0]))
    face_oval_height = float(np.max(oval[:, 1]) - np.min(oval[:, 1]))
    face_oval_diag = float(np.sqrt(face_oval_width ** 2 + face_oval_height ** 2))

    return {
        "scale_interocular": interocular,
        "scale_eye_width": eye_width_mean,
        "scale_face_width_cheek": face_width_cheek,
        "scale_face_oval_width": face_oval_width,
        "scale_face_oval_height": face_oval_height,
        "scale_face_oval_diag": face_oval_diag,
    }


def compute_pca_mask_length(mask):
    """
    Length of eyebrow mask along PCA major axis.
    """
    ys, xs = np.where(mask > 0)

    if len(xs) < 10:
        return {
            "pca_mask_length_success": 0,
            "pca_mask_length": np.nan,
            "pca_mask_minor_width": np.nan,
            "pca_mask_aspect_ratio": np.nan,
            "length_failure_reason": "too_few_mask_pixels",
        }

    pts = np.column_stack([xs, ys]).astype(np.float32)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    try:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        major = eigvecs[:, order[0]]
        minor = eigvecs[:, order[1]]

        proj_major = centered @ major
        proj_minor = centered @ minor

        length = float(np.max(proj_major) - np.min(proj_major))
        minor_width = float(np.max(proj_minor) - np.min(proj_minor))

        aspect = float(length / minor_width) if minor_width > 1e-8 else np.nan

        return {
            "pca_mask_length_success": 1,
            "pca_mask_length": length,
            "pca_mask_minor_width": minor_width,
            "pca_mask_aspect_ratio": aspect,
            "length_failure_reason": "",
        }

    except Exception as e:
        return {
            "pca_mask_length_success": 0,
            "pca_mask_length": np.nan,
            "pca_mask_minor_width": np.nan,
            "pca_mask_aspect_ratio": np.nan,
            "length_failure_reason": str(e),
        }


def compute_landmark_brow_length(landmarks, row, side):
    """
    Polyline length of eyebrow landmarks inside ROI coordinates.
    """
    idxs = get_side_eyebrow_indices(side)
    x1, y1, x2, y2 = get_roi_box(row, side)

    pts = landmarks[idxs, :2].astype(np.float32)
    pts[:, 0] -= x1
    pts[:, 1] -= y1

    length = 0.0
    for i in range(len(pts) - 1):
        length += dist(pts[i], pts[i + 1])

    return float(length)


def summarize(df, feature):
    gmdb = pd.to_numeric(df.loc[df["source"] == "GMDB", feature], errors="coerce").dropna()
    neg = pd.to_numeric(df.loc[df["source"] == "UTKFace", feature], errors="coerce").dropna()

    if len(gmdb) == 0 or len(neg) == 0:
        return None

    try:
        _, p = mannwhitneyu(gmdb, neg, alternative="two-sided")
    except Exception:
        p = np.nan

    return {
        "feature": feature,
        "gmdb_n": len(gmdb),
        "utk_n": len(neg),
        "gmdb_mean": float(np.mean(gmdb)),
        "utk_mean": float(np.mean(neg)),
        "mean_diff_gmdb_minus_utk": float(np.mean(gmdb) - np.mean(neg)),
        "gmdb_median": float(np.median(gmdb)),
        "utk_median": float(np.median(neg)),
        "median_diff_gmdb_minus_utk": float(np.median(gmdb) - np.median(neg)),
        "gmdb_std": float(np.std(gmdb, ddof=1)),
        "utk_std": float(np.std(neg, ddof=1)),
        "mannwhitney_p": float(p),
    }


def plot_bar(summary_df, features, out_path):
    sub = summary_df[summary_df["feature"].isin(features)].copy()
    sub["feature"] = pd.Categorical(sub["feature"], categories=features, ordered=True)
    sub = sub.sort_values("feature")

    labels = [
        f.replace("_lr_mean", "")
         .replace("pca_mask_length_norm_by_", "Length / ")
         .replace("pca_thickness_p95_over_pca_length", "Thickness / Length")
         .replace("face_width_cheek", "Face Width")
         .replace("face_oval_width", "Oval Width")
         .replace("face_oval_height", "Oval Height")
         .replace("interocular", "Eye Distance")
        for f in sub["feature"]
    ]

    gmdb = sub["gmdb_mean"].astype(float).values
    utk = sub["utk_mean"].astype(float).values

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, gmdb, width, label="GMDB thick eyebrow")
    plt.bar(x + width / 2, utk, width, label="Control")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Feature value")
    plt.title("Eyebrow Length-related Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================
# 4. Load side-level data
# =========================

if not SIDE_CSV.exists():
    raise FileNotFoundError(f"Cannot find: {SIDE_CSV}")

side_df = pd.read_csv(SIDE_CSV).fillna("")

print("Loaded side-level features:")
print(side_df.groupby(["source", "label", "side"]).size())


# =========================
# 5. Compute side-level length features
# =========================

records = []

for _, row in tqdm(side_df.iterrows(), total=len(side_df)):
    rec = row.to_dict()

    try:
        mask_path = resolve_path(row["mask_path"])
        landmark_path = resolve_path(row["aligned_landmark_path"])

        mask = imread_unicode(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        mask = (mask > 0).astype(np.uint8)

        landmarks = np.load(landmark_path)

        pca_result = compute_pca_mask_length(mask)
        scale_result = compute_scale_features(landmarks)
        landmark_len = compute_landmark_brow_length(landmarks, row, row["side"])

        rec.update(pca_result)
        rec.update(scale_result)
        rec["landmark_brow_length"] = landmark_len

        # Normalized length features
        for scale_name in [
            "interocular",
            "eye_width",
            "face_width_cheek",
            "face_oval_width",
            "face_oval_height",
            "face_oval_diag",
        ]:
            scale_col = f"scale_{scale_name}"
            scale_val = rec.get(scale_col, np.nan)

            if scale_val and np.isfinite(scale_val) and scale_val > 0:
                rec[f"pca_mask_length_norm_by_{scale_name}"] = rec["pca_mask_length"] / scale_val
                rec[f"landmark_brow_length_norm_by_{scale_name}"] = rec["landmark_brow_length"] / scale_val
            else:
                rec[f"pca_mask_length_norm_by_{scale_name}"] = np.nan
                rec[f"landmark_brow_length_norm_by_{scale_name}"] = np.nan

        # Shape ratios
        if "pca_thickness_p95" in row and pd.notna(row["pca_thickness_p95"]):
            thickness = float(row["pca_thickness_p95"])
            if rec["pca_mask_length"] and rec["pca_mask_length"] > 0:
                rec["pca_thickness_p95_over_pca_length"] = thickness / rec["pca_mask_length"]
            else:
                rec["pca_thickness_p95_over_pca_length"] = np.nan
        else:
            rec["pca_thickness_p95_over_pca_length"] = np.nan

        rec["length_success"] = 1

    except Exception as e:
        rec["length_success"] = 0
        rec["length_failure_reason"] = str(e)

    records.append(rec)


side_out = pd.DataFrame(records)
side_out.to_csv(SIDE_OUT, index=False)

print(f"\nSaved side-level length features to:\n{SIDE_OUT}")

print("\nLength success summary:")
print(side_out.groupby(["source", "label", "side"])["length_success"].agg(["count", "sum"]))


# =========================
# 6. Aggregate left/right to image-level
# =========================

key_cols = [
    "image_path", "filename", "image_id", "source", "label", "is_control",
    "patient_id", "age_year", "gender", "ethnicity", "omim_ids"
]
key_cols = [c for c in key_cols if c in side_out.columns]

length_cols = [
    c for c in side_out.columns
    if (
        c.startswith("pca_mask_length")
        or c.startswith("landmark_brow_length")
        or c.startswith("pca_thickness_p95_over_pca_length")
        or c.startswith("pca_mask_aspect_ratio")
    )
]

image_records = []

for _, g in side_out.groupby(key_cols, dropna=False):
    base = g.iloc[0][key_cols].to_dict()
    base["num_valid_sides_for_length"] = int(g["length_success"].sum())

    for col in length_cols:
        vals = pd.to_numeric(g[col], errors="coerce").dropna().values

        if len(vals) > 0:
            base[col + "_lr_mean"] = float(np.mean(vals))
            base[col + "_lr_max"] = float(np.max(vals))
            base[col + "_lr_min"] = float(np.min(vals))
        else:
            base[col + "_lr_mean"] = np.nan
            base[col + "_lr_max"] = np.nan
            base[col + "_lr_min"] = np.nan

    image_records.append(base)

image_out = pd.DataFrame(image_records)
image_out.to_csv(IMAGE_OUT, index=False)

print(f"Saved image-level length features to:\n{IMAGE_OUT}")

print("\nImage-level sample count:")
print(image_out.groupby(["source", "label"]).size())


# =========================
# 7. Group comparison
# =========================

summary_features = [
    "pca_mask_length_norm_by_interocular_lr_mean",
    "pca_mask_length_norm_by_eye_width_lr_mean",
    "pca_mask_length_norm_by_face_width_cheek_lr_mean",
    "pca_mask_length_norm_by_face_oval_width_lr_mean",
    "pca_mask_length_norm_by_face_oval_height_lr_mean",
    "pca_mask_length_norm_by_face_oval_diag_lr_mean",
    "landmark_brow_length_norm_by_face_width_cheek_lr_mean",
    "pca_mask_aspect_ratio_lr_mean",
    "pca_thickness_p95_over_pca_length_lr_mean",
]

summary_records = []

for f in summary_features:
    if f in image_out.columns:
        r = summarize(image_out, f)
        if r is not None:
            summary_records.append(r)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(SUMMARY_OUT, index=False)

print(f"\nSaved length group comparison to:\n{SUMMARY_OUT}")

print("\nLength feature comparison:")
print(summary_df.to_string(index=False))


# =========================
# 8. Plot quick bar chart
# =========================

plot_features = [
    "pca_mask_length_norm_by_face_width_cheek_lr_mean",
    "pca_mask_length_norm_by_face_oval_width_lr_mean",
    "pca_mask_length_norm_by_face_oval_height_lr_mean",
    "pca_thickness_p95_over_pca_length_lr_mean",
]

plot_bar(
    summary_df,
    plot_features,
    PLOT_DIR / "bar_eyebrow_length_features.png"
)

print(f"\nSaved length plot to:\n{PLOT_DIR / 'bar_eyebrow_length_features.png'}")