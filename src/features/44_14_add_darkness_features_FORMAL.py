import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mannwhitneyu


BASE_DIR = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

if len(sys.argv) >= 2:
    RESULT_DIR = BASE_DIR / sys.argv[1]
else:
    RESULT_DIR = BASE_DIR / "outputs"

SIDE_CSV = RESULT_DIR / "features" / "eyebrow_thickness_side_features.csv"

FEATURE_DIR = RESULT_DIR / "features"
TABLE_DIR = RESULT_DIR / "tables"
PLOT_DIR = RESULT_DIR / "plots" / "darkness_features"
QC_DIR = RESULT_DIR / "qc" / "eyebrow_darkness_debug"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

SIDE_OUT = FEATURE_DIR / "eyebrow_darkness_side_features.csv"
IMAGE_OUT = FEATURE_DIR / "eyebrow_darkness_image_features.csv"
SUMMARY_OUT = TABLE_DIR / "darkness_group_comparison.csv"


LEFT_EYEBROW_IDXS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_IDXS = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]


def resolve_path(path_str):
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
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, img)
    if ok:
        encoded.tofile(str(path))
    return ok


def get_image_path_from_row(row):
    candidates = [
        "aligned_image_path",
        "aligned_path",
        "aligned_face_path",
        "image_path",
    ]

    for c in candidates:
        if c in row.index and str(row[c]).strip() != "":
            p = resolve_path(row[c])
            if p.exists():
                return p

    raise ValueError("no_valid_image_path_column")


def get_mask_path_from_row(row):
    candidates = [
        "mask_path",
        "eyebrow_mask_path",
        "mask_image_path",
    ]

    for c in candidates:
        if c in row.index and str(row[c]).strip() != "":
            p = resolve_path(row[c])
            if p.exists():
                return p

    raise ValueError("no_valid_mask_path_column")


def get_landmark_path_from_row(row):
    candidates = [
        "aligned_landmark_path",
        "landmark_path",
        "landmarks_path",
    ]

    for c in candidates:
        if c in row.index and str(row[c]).strip() != "":
            p = resolve_path(row[c])
            if p.exists():
                return p

    raise ValueError("no_valid_landmark_path_column")


def get_roi_box(row, side):
    if side == "left":
        cols = ["left_x1", "left_y1", "left_x2", "left_y2"]
    else:
        cols = ["right_x1", "right_y1", "right_x2", "right_y2"]

    for c in cols:
        if c not in row.index:
            raise ValueError(f"missing_roi_box_column:{c}")

    return tuple(int(float(row[c])) for c in cols)


def crop_roi_from_image(img, row, side):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = get_roi_box(row, side)

    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid_roi_box:{x1},{y1},{x2},{y2}")

    return img[y1:y2, x1:x2].copy()


def make_odd(x):
    x = int(round(x))
    x = max(3, x)
    if x % 2 == 0:
        x += 1
    return x


def get_eyebrow_indices(side):
    return LEFT_EYEBROW_IDXS if side == "left" else RIGHT_EYEBROW_IDXS


def build_landmark_tube_mask(roi_shape, landmarks, row, side):
    h, w = roi_shape[:2]
    x1, y1, x2, y2 = get_roi_box(row, side)

    pts = landmarks[get_eyebrow_indices(side), :2].astype(np.float32)
    pts[:, 0] -= x1
    pts[:, 1] -= y1

    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    core = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(core, hull, 255)

    brow_w = max(1.0, float(np.max(pts[:, 0]) - np.min(pts[:, 0])))
    brow_h = max(1.0, float(np.max(pts[:, 1]) - np.min(pts[:, 1])))

    dilate_x = max(5, int(round(brow_w * 0.10)))
    dilate_y = max(7, int(round(brow_h * 0.42)))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (make_odd(dilate_x), make_odd(dilate_y)),
    )

    tube = cv2.dilate(core, kernel, iterations=1)
    return tube


def robust_skin_reference(gray, tube_mask, mask):
    tube = (tube_mask > 0).astype(np.uint8)
    m = (mask > 0).astype(np.uint8)

    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(tube, kernel, iterations=2)

    skin_ref = ((dilated > 0) & (tube == 0) & (m == 0)).astype(np.uint8)
    vals = gray[skin_ref > 0]

    if len(vals) < 30:
        fallback = ((m == 0) & (tube == 0)).astype(np.uint8)
        vals = gray[fallback > 0]

    if len(vals) < 30:
        vals = gray.reshape(-1)

    return float(np.median(vals))


def compute_pca_axis(binary_mask):
    ys, xs = np.where(binary_mask > 0)

    if len(xs) < 10:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    major = eigvecs[:, order[0]]
    return centroid, major


def compute_global_darkness(gray, mask, skin_gray):
    pix = gray[mask > 0].astype(np.float32)

    if len(pix) < 10:
        return None

    darkness = np.maximum(skin_gray - pix, 0)

    return {
        "mask_darkness_mean": float(np.mean(darkness)),
        "mask_darkness_median": float(np.median(darkness)),
        "mask_darkness_p75": float(np.percentile(darkness, 75)),
        "mask_darkness_p90": float(np.percentile(darkness, 90)),
        "mask_darkness_p95": float(np.percentile(darkness, 95)),
        "mask_darkness_max": float(np.max(darkness)),
        "mask_dark_pixel_ratio_10": float(np.mean(darkness >= 10)),
        "mask_dark_pixel_ratio_20": float(np.mean(darkness >= 20)),
    }


def compute_local_darkness(gray, tube_mask, mask, skin_gray, n_segments=24):
    axis = compute_pca_axis(tube_mask)
    if axis is None:
        return None

    centroid, major = axis

    ys, xs = np.where(tube_mask > 0)
    tube_pts = np.column_stack([xs, ys]).astype(np.float32)
    tube_proj = (tube_pts - centroid) @ major

    p_min = float(np.min(tube_proj))
    p_max = float(np.max(tube_proj))

    if p_max <= p_min:
        return None

    mask_ys, mask_xs = np.where(mask > 0)

    if len(mask_xs) < 10:
        return None

    mask_pts = np.column_stack([mask_xs, mask_ys]).astype(np.float32)
    mask_proj = (mask_pts - centroid) @ major

    edges = np.linspace(p_min, p_max, n_segments + 1)
    segment_darkness = []

    for i in range(n_segments):
        lo = edges[i]
        hi = edges[i + 1]

        if i == n_segments - 1:
            in_seg = (mask_proj >= lo) & (mask_proj <= hi)
        else:
            in_seg = (mask_proj >= lo) & (mask_proj < hi)

        if np.sum(in_seg) < 5:
            continue

        px = gray[mask_ys[in_seg], mask_xs[in_seg]].astype(np.float32)
        darkness = np.maximum(skin_gray - px, 0)
        segment_darkness.append(float(np.mean(darkness)))

    if len(segment_darkness) == 0:
        return None

    arr = np.array(segment_darkness, dtype=np.float32)

    return {
        "local_darkness_mean": float(np.mean(arr)),
        "local_darkness_median": float(np.median(arr)),
        "local_darkness_p75": float(np.percentile(arr, 75)),
        "local_darkness_p90": float(np.percentile(arr, 90)),
        "local_darkness_p95": float(np.percentile(arr, 95)),
        "local_darkness_max": float(np.max(arr)),
        "local_dark_segment_count": int(len(arr)),
    }


def make_debug_image(roi_bgr, tube_mask, mask, skin_gray, out_path):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    darkness = np.maximum(skin_gray - gray.astype(np.float32), 0)
    p99 = max(1.0, float(np.percentile(darkness, 99)))
    darkness_vis = np.clip(darkness / p99 * 255, 0, 255).astype(np.uint8)

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    tube_vis = roi_rgb.copy()
    tube_vis[tube_mask > 0] = [120, 120, 255]

    dark_rgb = cv2.cvtColor(darkness_vis, cv2.COLOR_GRAY2RGB)

    mask_vis = roi_rgb.copy()
    mask_vis[mask > 0] = [0, 255, 0]

    panels = [roi_rgb, tube_vis, dark_rgb, mask_vis]
    labels = ["ROI", "Tube", "Relative Darkness", "Mask"]

    h = max(p.shape[0] for p in panels)
    resized = []

    for p in panels:
        if p.shape[0] != h:
            scale = h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * scale), h))
        resized.append(p)

    canvas = np.concatenate(resized, axis=1)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    x = 5
    for label, p in zip(labels, resized):
        cv2.putText(
            canvas_bgr,
            label,
            (x, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        x += p.shape[1]

    imwrite_unicode(out_path, canvas_bgr)


def summarize_group(df, feature):
    gmdb = pd.to_numeric(df.loc[df["source"] == "GMDB", feature], errors="coerce").dropna()
    neg = pd.to_numeric(df.loc[df["source"] == "UTKFace", feature], errors="coerce").dropna()

    if len(gmdb) == 0 or len(neg) == 0:
        return None

    _, p_value = mannwhitneyu(gmdb, neg, alternative="two-sided")

    return {
        "feature": feature,
        "gmdb_n": len(gmdb),
        "negative_n": len(neg),
        "gmdb_mean": float(np.mean(gmdb)),
        "negative_mean": float(np.mean(neg)),
        "mean_diff_gmdb_minus_negative": float(np.mean(gmdb) - np.mean(neg)),
        "gmdb_median": float(np.median(gmdb)),
        "negative_median": float(np.median(neg)),
        "median_diff_gmdb_minus_negative": float(np.median(gmdb) - np.median(neg)),
        "gmdb_std": float(np.std(gmdb, ddof=1)),
        "negative_std": float(np.std(neg, ddof=1)),
        "mannwhitney_p": float(p_value),
    }


def plot_bar(summary_df, features, out_path):
    if summary_df.empty or "feature" not in summary_df.columns:
        print("No summary rows to plot.")
        return

    sub = summary_df[summary_df["feature"].isin(features)].copy()

    if sub.empty:
        print("No selected features found for plot.")
        return

    sub["feature"] = pd.Categorical(sub["feature"], categories=features, ordered=True)
    sub = sub.sort_values("feature")

    label_map = {
        "mask_darkness_mean_lr_mean": "Mask Mean",
        "mask_darkness_p90_lr_mean": "Mask P90",
        "mask_darkness_p95_lr_mean": "Mask P95",
        "local_darkness_mean_lr_mean": "Local Mean",
        "local_darkness_p90_lr_mean": "Local P90",
        "local_darkness_p95_lr_mean": "Local P95",
        "mask_dark_pixel_ratio_20_lr_mean": "Dark Ratio ≥20",
    }

    labels = [label_map.get(x, x) for x in sub["feature"]]
    gmdb = sub["gmdb_mean"].astype(float).values
    neg = sub["negative_mean"].astype(float).values
    pvals = sub["mannwhitney_p"].astype(float).values

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, gmdb, width, label="GMDB thick eyebrow")
    plt.bar(x + width / 2, neg, width, label="UTKFace clean control")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Relative darkness")
    plt.title("Darkness-based Eyebrow Features")
    plt.legend()

    for i, p in enumerate(pvals):
        y = max(gmdb[i], neg[i])
        plt.text(x[i], y + max(0.5, y * 0.03), f"p={p:.1e}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if not SIDE_CSV.exists():
    raise FileNotFoundError(f"Cannot find: {SIDE_CSV}")

side_df = pd.read_csv(SIDE_CSV).fillna("")

print("Loaded side features:")
print(side_df.groupby(["source", "label", "side"]).size())

records = []

for idx, row in tqdm(side_df.iterrows(), total=len(side_df)):
    rec = row.to_dict()

    try:
        image_path = get_image_path_from_row(row)
        mask_path = get_mask_path_from_row(row)
        landmark_path = get_landmark_path_from_row(row)

        full_img = imread_unicode(image_path, cv2.IMREAD_COLOR)
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        landmarks = np.load(landmark_path)

        if full_img is None:
            raise ValueError(f"failed_to_read_image:{image_path}")
        if mask is None:
            raise ValueError(f"failed_to_read_mask:{mask_path}")

        roi_bgr = crop_roi_from_image(full_img, row, row["side"])

        # mask should already be ROI-sized
        if mask.shape[:2] != roi_bgr.shape[:2]:
            mask = cv2.resize(mask, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.uint8)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        tube_mask = build_landmark_tube_mask(roi_bgr.shape, landmarks, row, row["side"])

        skin_gray = robust_skin_reference(gray, tube_mask, mask)
        rec["skin_reference_gray"] = skin_gray

        global_dark = compute_global_darkness(gray, mask, skin_gray)
        local_dark = compute_local_darkness(gray, tube_mask, mask, skin_gray, n_segments=24)

        if global_dark is None:
            raise ValueError("global_darkness_failed")
        if local_dark is None:
            raise ValueError("local_darkness_failed")

        rec.update(global_dark)
        rec.update(local_dark)
        rec["darkness_success"] = 1
        rec["darkness_failure_reason"] = ""

        if idx % 200 == 0:
            source = str(row.get("source", "source"))
            label = str(row.get("label", "label"))
            side = str(row.get("side", "side"))

            out_subdir = QC_DIR / source / label
            out_subdir.mkdir(parents=True, exist_ok=True)

            base_name = Path(str(row.get("filename", f"idx_{idx}"))).stem
            out_path = out_subdir / f"{base_name}_{side}_darkness_debug.jpg"
            make_debug_image(roi_bgr, tube_mask, mask, skin_gray, out_path)

    except Exception as e:
        rec["darkness_success"] = 0
        rec["darkness_failure_reason"] = str(e)

    records.append(rec)

side_out = pd.DataFrame(records)
side_out.to_csv(SIDE_OUT, index=False)

print(f"\nSaved side-level darkness features to:\n{SIDE_OUT}")

print("\nDarkness success summary:")
print(side_out.groupby(["source", "label", "side"])["darkness_success"].agg(["count", "sum"]))

if "darkness_failure_reason" in side_out.columns:
    print("\nFailure reasons:")
    print(side_out.loc[side_out["darkness_success"] == 0, "darkness_failure_reason"].value_counts().head(10))

key_cols = [
    "image_path", "filename", "image_id", "source", "label", "is_control",
    "patient_id", "age_year", "gender", "ethnicity", "omim_ids"
]
key_cols = [c for c in key_cols if c in side_out.columns]

dark_cols = [
    c for c in side_out.columns
    if (
        c.startswith("mask_darkness")
        or c.startswith("mask_dark_pixel_ratio")
        or c.startswith("local_darkness")
        or c.startswith("local_dark_segment_count")
        or c == "skin_reference_gray"
    )
]

image_records = []

for _, g in side_out.groupby(key_cols, dropna=False):
    base = g.iloc[0][key_cols].to_dict()
    base["num_valid_sides_for_darkness"] = int(g["darkness_success"].sum())

    for col in dark_cols:
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

print(f"\nSaved image-level darkness features to:\n{IMAGE_OUT}")

print("\nImage-level sample count:")
print(image_out.groupby(["source", "label"]).size())

summary_features = [
    "mask_darkness_mean_lr_mean",
    "mask_darkness_median_lr_mean",
    "mask_darkness_p75_lr_mean",
    "mask_darkness_p90_lr_mean",
    "mask_darkness_p95_lr_mean",
    "mask_darkness_max_lr_mean",
    "mask_dark_pixel_ratio_10_lr_mean",
    "mask_dark_pixel_ratio_20_lr_mean",
    "local_darkness_mean_lr_mean",
    "local_darkness_median_lr_mean",
    "local_darkness_p75_lr_mean",
    "local_darkness_p90_lr_mean",
    "local_darkness_p95_lr_mean",
    "local_darkness_max_lr_mean",
]

summary_records = []

for f in summary_features:
    if f in image_out.columns:
        r = summarize_group(image_out, f)
        if r is not None:
            summary_records.append(r)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(SUMMARY_OUT, index=False)

print(f"\nSaved darkness group comparison to:\n{SUMMARY_OUT}")

print("\nDarkness feature comparison:")
if len(summary_df) > 0:
    print(summary_df.to_string(index=False))
else:
    print("Empty summary.")

plot_features = [
    "mask_darkness_mean_lr_mean",
    "mask_darkness_p90_lr_mean",
    "mask_darkness_p95_lr_mean",
    "local_darkness_mean_lr_mean",
    "local_darkness_p90_lr_mean",
    "local_darkness_p95_lr_mean",
    "mask_dark_pixel_ratio_20_lr_mean",
]

plot_bar(summary_df, plot_features, PLOT_DIR / "bar_darkness_features.png")

print(f"\nSaved darkness plot to:\n{PLOT_DIR / 'bar_darkness_features.png'}")
print(f"Saved darkness debug images to:\n{QC_DIR}")