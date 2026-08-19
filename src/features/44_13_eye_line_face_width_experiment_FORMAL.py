import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =========================
# 1. Path settings
# =========================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

# 預設跑 clean control
# 如果要跑 CelebA-HQ hard negative，可用：
# python 13_eye_line_face_width_experiment.py outputs_celeba_hq_bushy_hardnegative_result
if len(sys.argv) >= 2:
    RESULT_DIR = BASE_DIR / sys.argv[1]
else:
    RESULT_DIR = BASE_DIR / "outputs"

SIDE_CSV = RESULT_DIR / "features" / "eyebrow_thickness_side_features.csv"
MULTI_NORM_IMAGE_CSV = RESULT_DIR / "features" / "eyebrow_thickness_image_features_multi_norm.csv"
LOCAL_DENSITY_CSV = RESULT_DIR / "features" / "eyebrow_local_density_image_features.csv"

FEATURE_DIR = RESULT_DIR / "features"
TABLE_DIR = RESULT_DIR / "tables"
PLOT_DIR = RESULT_DIR / "plots" / "eye_line_face_width"
RULE_DIR = RESULT_DIR / "multifeature_rule_eye_line"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RULE_DIR.mkdir(parents=True, exist_ok=True)

SIDE_OUT = FEATURE_DIR / "eyebrow_thickness_eye_line_width_side_features.csv"
IMAGE_OUT = FEATURE_DIR / "eyebrow_thickness_eye_line_width_image_features.csv"
SUMMARY_OUT = TABLE_DIR / "eye_line_face_width_normalization_summary.csv"
RULE_SUMMARY_OUT = RULE_DIR / "multifeature_rule_eye_line_summary.csv"
RULE_PRED_OUT = RULE_DIR / "multifeature_rule_eye_line_predictions.csv"


# =========================
# 2. MediaPipe landmark indices
# =========================

LEFT_EYE_CENTER_IDXS = [33, 133, 159, 145]
RIGHT_EYE_CENTER_IDXS = [362, 263, 386, 374]

# Face oval landmarks in ordered contour
FACE_OVAL_IDXS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


RAW_THICKNESS_COLS = [
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
]


# =========================
# 3. Utility functions
# =========================

def resolve_path(path_str):
    p = Path(str(path_str))
    if p.exists():
        return p

    # 如果 CSV 仍然記錄舊 outputs 路徑，自動轉成目前 RESULT_DIR
    old_prefix = str(BASE_DIR / "outputs")
    new_prefix = str(RESULT_DIR)

    s = str(p)
    if s.startswith(old_prefix):
        candidate = Path(new_prefix + s[len(old_prefix):])
        if candidate.exists():
            return candidate

    return p


def dist(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def compute_eye_line_face_width(landmarks):
    """
    Eye-line face width:
    1. Compute eye-line y coordinate from left/right eye centers.
    2. Intersect horizontal eye line with face oval contour.
    3. Width = rightmost intersection x - leftmost intersection x.
    """
    xy = landmarks[:, :2].astype(np.float32)

    left_eye_center = xy[LEFT_EYE_CENTER_IDXS].mean(axis=0)
    right_eye_center = xy[RIGHT_EYE_CENTER_IDXS].mean(axis=0)

    y_eye = float((left_eye_center[1] + right_eye_center[1]) / 2.0)

    oval = xy[FACE_OVAL_IDXS]

    xs = []

    n = len(oval)
    for i in range(n):
        p1 = oval[i]
        p2 = oval[(i + 1) % n]

        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])

        # Avoid horizontal segments
        if abs(y2 - y1) < 1e-8:
            continue

        # Half-open crossing rule to avoid double-counting vertices
        if (y1 <= y_eye < y2) or (y2 <= y_eye < y1):
            t = (y_eye - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            xs.append(float(x))

    if len(xs) < 2:
        return {
            "scale_face_width_eye_line": np.nan,
            "eye_line_y": y_eye,
            "eye_line_left_x": np.nan,
            "eye_line_right_x": np.nan,
            "eye_line_width_success": 0,
            "eye_line_width_failure_reason": f"less_than_2_intersections:{len(xs)}",
        }

    left_x = float(np.min(xs))
    right_x = float(np.max(xs))
    width = right_x - left_x

    if width <= 0:
        return {
            "scale_face_width_eye_line": np.nan,
            "eye_line_y": y_eye,
            "eye_line_left_x": left_x,
            "eye_line_right_x": right_x,
            "eye_line_width_success": 0,
            "eye_line_width_failure_reason": "non_positive_width",
        }

    return {
        "scale_face_width_eye_line": float(width),
        "eye_line_y": y_eye,
        "eye_line_left_x": left_x,
        "eye_line_right_x": right_x,
        "eye_line_width_success": 1,
        "eye_line_width_failure_reason": "",
    }


def summarize_group(df, feature):
    gmdb = pd.to_numeric(df.loc[df["source"] == "GMDB", feature], errors="coerce").dropna()
    neg = pd.to_numeric(df.loc[df["source"] == "UTKFace", feature], errors="coerce").dropna()

    if len(gmdb) == 0 or len(neg) == 0:
        return None

    try:
        _, p_value = mannwhitneyu(gmdb, neg, alternative="two-sided")
    except Exception:
        p_value = np.nan

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
        "negative_p80": float(np.percentile(neg, 80)),
        "negative_p85": float(np.percentile(neg, 85)),
        "negative_p90": float(np.percentile(neg, 90)),
        "negative_p95": float(np.percentile(neg, 95)),
        "gmdb_over_negative_p80_ratio": float(np.mean(gmdb > np.percentile(neg, 80))),
        "gmdb_over_negative_p90_ratio": float(np.mean(gmdb > np.percentile(neg, 90))),
        "gmdb_over_negative_p95_ratio": float(np.mean(gmdb > np.percentile(neg, 95))),
        "mannwhitney_p": float(p_value),
    }


def evaluate_rule(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =========================
# 4. Load side-level thickness features
# =========================

if not SIDE_CSV.exists():
    raise FileNotFoundError(f"Cannot find side CSV: {SIDE_CSV}")

side_df = pd.read_csv(SIDE_CSV).fillna("")

print("Loaded side-level features:")
print(side_df.groupby(["source", "label", "side"]).size())


# =========================
# 5. Compute eye-line face width and normalized thickness
# =========================

landmark_cache = {}
records = []

for _, row in side_df.iterrows():
    rec = row.to_dict()
    landmark_path = resolve_path(row["aligned_landmark_path"])

    cache_key = str(landmark_path)

    if cache_key in landmark_cache:
        scale_result = landmark_cache[cache_key]
    else:
        try:
            landmarks = np.load(landmark_path)
            scale_result = compute_eye_line_face_width(landmarks)
        except Exception as e:
            scale_result = {
                "scale_face_width_eye_line": np.nan,
                "eye_line_y": np.nan,
                "eye_line_left_x": np.nan,
                "eye_line_right_x": np.nan,
                "eye_line_width_success": 0,
                "eye_line_width_failure_reason": str(e),
            }

        landmark_cache[cache_key] = scale_result

    rec.update(scale_result)

    scale = rec["scale_face_width_eye_line"]

    for raw_col in RAW_THICKNESS_COLS:
        if raw_col in side_df.columns:
            raw = pd.to_numeric(pd.Series([row[raw_col]]), errors="coerce").iloc[0]

            out_col = f"{raw_col}_norm_by_face_width_eye_line"

            if pd.notna(raw) and pd.notna(scale) and scale > 0:
                rec[out_col] = float(raw) / float(scale)
            else:
                rec[out_col] = np.nan

    records.append(rec)

side_out = pd.DataFrame(records)
side_out.to_csv(SIDE_OUT, index=False)

print(f"\nSaved side-level eye-line features to:\n{SIDE_OUT}")

print("\nEye-line width success summary:")
print(side_out.groupby(["source", "label", "side"])["eye_line_width_success"].agg(["count", "sum"]))


# =========================
# 6. Aggregate left/right to image-level
# =========================

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

key_cols = [c for c in key_cols if c in side_out.columns]

feature_cols = [
    c for c in side_out.columns
    if c.endswith("_norm_by_face_width_eye_line")
]

# scale is image-level, but repeated on left/right rows
feature_cols += [
    "scale_face_width_eye_line",
    "eye_line_y",
]

image_records = []

for _, group in side_out.groupby(key_cols, dropna=False):
    base = group.iloc[0][key_cols].to_dict()
    base["num_valid_sides_eye_line"] = int(group["eye_line_width_success"].sum())

    for col in feature_cols:
        vals = pd.to_numeric(group[col], errors="coerce").dropna().values

        if len(vals) > 0:
            base[col + "_lr_mean"] = float(np.mean(vals))
            base[col + "_lr_max"] = float(np.max(vals))
            base[col + "_lr_min"] = float(np.min(vals))
        else:
            base[col + "_lr_mean"] = np.nan
            base[col + "_lr_max"] = np.nan
            base[col + "_lr_min"] = np.nan

    image_records.append(base)

image_df = pd.DataFrame(image_records)
image_df.to_csv(IMAGE_OUT, index=False)

print(f"\nSaved image-level eye-line features to:\n{IMAGE_OUT}")

print("\nImage-level sample count:")
print(image_df.groupby(["source", "label"]).size())


# =========================
# 7. Normalization summary
# =========================

summary_features = [
    "pca_thickness_mean_norm_by_face_width_eye_line_lr_mean",
    "pca_thickness_p90_norm_by_face_width_eye_line_lr_mean",
    "pca_thickness_p95_norm_by_face_width_eye_line_lr_mean",
    "pca_thickness_max_norm_by_face_width_eye_line_lr_mean",
    "skeleton_thickness_p95_norm_by_face_width_eye_line_lr_mean",
]

summary_records = []

for f in summary_features:
    if f in image_df.columns:
        r = summarize_group(image_df, f)
        if r is not None:
            summary_records.append(r)

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(SUMMARY_OUT, index=False)

print(f"\nSaved eye-line normalization summary to:\n{SUMMARY_OUT}")

print("\nEye-line face width normalization summary:")
if len(summary_df) > 0:
    print(summary_df.to_string(index=False))
else:
    print("No summary rows.")


# =========================
# 8. Compare with existing normalization if available
# =========================

compare_df = image_df.copy()

if MULTI_NORM_IMAGE_CSV.exists():
    old_df = pd.read_csv(MULTI_NORM_IMAGE_CSV).fillna("")

    keep_cols = [
        "image_path",
        "pca_thickness_p95_norm_by_interocular_lr_mean",
        "pca_thickness_p95_norm_by_eye_width_lr_mean",
        "pca_thickness_p95_norm_by_face_width_cheek_lr_mean",
        "pca_thickness_p95_norm_by_face_oval_width_lr_mean",
        "pca_thickness_p95_norm_by_face_oval_height_lr_mean",
    ]
    keep_cols = [c for c in keep_cols if c in old_df.columns]

    compare_df = compare_df.merge(
        old_df[keep_cols],
        on="image_path",
        how="left",
    )

compare_features = [
    "pca_thickness_p95_norm_by_interocular_lr_mean",
    "pca_thickness_p95_norm_by_eye_width_lr_mean",
    "pca_thickness_p95_norm_by_face_width_cheek_lr_mean",
    "pca_thickness_p95_norm_by_face_oval_width_lr_mean",
    "pca_thickness_p95_norm_by_face_oval_height_lr_mean",
    "pca_thickness_p95_norm_by_face_width_eye_line_lr_mean",
]

compare_records = []
for f in compare_features:
    if f in compare_df.columns:
        r = summarize_group(compare_df, f)
        if r is not None:
            compare_records.append(r)

compare_summary_df = pd.DataFrame(compare_records)
compare_summary_path = TABLE_DIR / "normalization_with_eye_line_face_width_summary.csv"
compare_summary_df.to_csv(compare_summary_path, index=False)

print(f"\nSaved comparison with previous normalizers to:\n{compare_summary_path}")

if len(compare_summary_df) > 0:
    print("\nComparison summary:")
    print(compare_summary_df[[
        "feature",
        "gmdb_mean",
        "negative_mean",
        "mean_diff_gmdb_minus_negative",
        "mannwhitney_p",
    ]].to_string(index=False))


# =========================
# 9. Quick plot
# =========================

if len(compare_summary_df) > 0:
    plot_features = [
        f for f in compare_features
        if f in compare_summary_df["feature"].values
    ]

    sub = compare_summary_df[compare_summary_df["feature"].isin(plot_features)].copy()
    sub["feature"] = pd.Categorical(sub["feature"], categories=plot_features, ordered=True)
    sub = sub.sort_values("feature")

    label_map = {
        "pca_thickness_p95_norm_by_interocular_lr_mean": "Eye distance",
        "pca_thickness_p95_norm_by_eye_width_lr_mean": "Eye width",
        "pca_thickness_p95_norm_by_face_width_cheek_lr_mean": "Cheek width",
        "pca_thickness_p95_norm_by_face_oval_width_lr_mean": "Oval width",
        "pca_thickness_p95_norm_by_face_oval_height_lr_mean": "Oval height",
        "pca_thickness_p95_norm_by_face_width_eye_line_lr_mean": "Eye-line face width",
    }

    labels = [label_map.get(x, x) for x in sub["feature"]]
    gmdb_values = sub["gmdb_mean"].astype(float).values
    neg_values = sub["negative_mean"].astype(float).values

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, gmdb_values, width, label="GMDB thick eyebrow")
    plt.bar(x + width / 2, neg_values, width, label="Control / negative")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Normalized Thickness")
    plt.title("PCA Thickness P95 with Different Normalization Scales")
    plt.legend()
    plt.tight_layout()

    plot_path = PLOT_DIR / "bar_pca_thickness_p95_with_eye_line_width.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nSaved plot to:\n{plot_path}")


# =========================
# 10. Rule-based prediction with eye-line face width
# =========================

rule_df = image_df.copy()
rule_df["y_true"] = (rule_df["source"] == "GMDB").astype(int)

# Merge local density
if LOCAL_DENSITY_CSV.exists():
    local_df = pd.read_csv(LOCAL_DENSITY_CSV).fillna("")

    local_cols = [
        "image_path",
        "mask_tube_ratio_lr_mean",
        "mask_area_ratio_lr_mean",
        "local_filling_p90_lr_mean",
        "local_filling_p95_lr_mean",
        "local_filling_mean_lr_mean",
    ]
    local_cols = [c for c in local_cols if c in local_df.columns]

    rule_df = rule_df.merge(local_df[local_cols], on="image_path", how="left")

thickness_feature = "pca_thickness_p95_norm_by_face_width_eye_line_lr_mean"
global_density_feature = "mask_tube_ratio_lr_mean"
local_density_feature = "local_filling_p90_lr_mean"

needed = [thickness_feature, global_density_feature, local_density_feature]

if all(c in rule_df.columns for c in needed):
    for c in needed:
        rule_df[c] = pd.to_numeric(rule_df[c], errors="coerce")

    rule_df = rule_df.dropna(subset=needed).copy()

    neg_df = rule_df[rule_df["y_true"] == 0].copy()

    rule_records = []
    pred_frames = []

    for pct in [70, 75, 80, 85, 90, 95]:
        th_thickness = np.nanpercentile(neg_df[thickness_feature], pct)
        th_global = np.nanpercentile(neg_df[global_density_feature], pct)
        th_local = np.nanpercentile(neg_df[local_density_feature], pct)

        temp = rule_df.copy()

        temp["cond_thickness_eye_line"] = temp[thickness_feature] > th_thickness
        temp["cond_global_density"] = temp[global_density_feature] > th_global
        temp["cond_local_density"] = temp[local_density_feature] > th_local

        cond_cols = [
            "cond_thickness_eye_line",
            "cond_global_density",
            "cond_local_density",
        ]

        temp["rule_score"] = temp[cond_cols].sum(axis=1)

        rules = {
            "single_thickness_eye_line": temp["cond_thickness_eye_line"].astype(int),
            "single_global_density": temp["cond_global_density"].astype(int),
            "single_local_density": temp["cond_local_density"].astype(int),
            "any_1_of_3": (temp["rule_score"] >= 1).astype(int),
            "majority_2_of_3": (temp["rule_score"] >= 2).astype(int),
            "strict_3_of_3": (temp["rule_score"] >= 3).astype(int),
        }

        for rule_name, y_pred in rules.items():
            metrics = evaluate_rule(temp["y_true"].values, y_pred.values)

            row = {
                "percentile_threshold": pct,
                "rule": rule_name,
                "thickness_feature": thickness_feature,
                "global_density_feature": global_density_feature,
                "local_density_feature": local_density_feature,
                "threshold_thickness_eye_line": th_thickness,
                "threshold_global_density": th_global,
                "threshold_local_density": th_local,
            }
            row.update(metrics)
            rule_records.append(row)

        temp["percentile_threshold"] = pct
        temp["pred_majority_2_of_3"] = (temp["rule_score"] >= 2).astype(int)
        temp["pred_strict_3_of_3"] = (temp["rule_score"] >= 3).astype(int)

        keep_cols = [
            "image_path",
            "filename",
            "source",
            "label",
            "y_true",
            "percentile_threshold",
            "rule_score",
            "cond_thickness_eye_line",
            "cond_global_density",
            "cond_local_density",
            "pred_majority_2_of_3",
            "pred_strict_3_of_3",
            thickness_feature,
            global_density_feature,
            local_density_feature,
        ]
        keep_cols = [c for c in keep_cols if c in temp.columns]
        pred_frames.append(temp[keep_cols].copy())

    rule_summary = pd.DataFrame(rule_records)
    rule_preds = pd.concat(pred_frames, ignore_index=True)

    rule_summary.to_csv(RULE_SUMMARY_OUT, index=False)
    rule_preds.to_csv(RULE_PRED_OUT, index=False)

    print(f"\nSaved eye-line rule summary to:\n{RULE_SUMMARY_OUT}")
    print(f"Saved eye-line rule predictions to:\n{RULE_PRED_OUT}")

    display_cols = [
        "percentile_threshold",
        "rule",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    print("\nEye-line rule-based prediction summary:")
    print(rule_summary[display_cols].to_string(index=False))

    print("\nRecommended rows: majority_2_of_3 with P80 / P85 / P90 thresholds")
    rec = rule_summary[
        (rule_summary["rule"] == "majority_2_of_3")
        & (rule_summary["percentile_threshold"].isin([80, 85, 90]))
    ]
    print(rec[display_cols].to_string(index=False))

else:
    print("\nSkipped rule-based prediction because some needed columns are missing:")
    for c in needed:
        print(c, c in rule_df.columns)