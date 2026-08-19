import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
import shap
from joblib import dump


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_ANALYSIS_ROOT", Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))) / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"))

FEATURE_TABLE = BASE_DIR / "formal_model_input/merged_model_input_features.csv"

OUT_DIR = BASE_DIR / "xgboost_shap_interpretability"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fixed 5 interpretable features
# ============================================================

FEATURES = [
    "mask_tube_ratio_lr_mean",
    "mask_area_ratio_lr_mean",
    "local_darkness_p95_lr_mean",
    "pca_mask_length_norm_by_face_width_cheek_lr_mean",
    "pca_thickness_mean_norm_lr_mean",
]

DISPLAY_NAMES = {
    "mask_tube_ratio_lr_mean": "Mask / tube ratio",
    "mask_area_ratio_lr_mean": "Mask area ratio",
    "local_darkness_p95_lr_mean": "Local darkness P95",
    "pca_mask_length_norm_by_face_width_cheek_lr_mean": "Normalized mask length",
    "pca_thickness_mean_norm_lr_mean": "Normalized thickness mean",
}


# ============================================================
# Helper functions
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def percentile_rank(reference_values, value):
    ref = pd.Series(reference_values).dropna().astype(float).values
    if len(ref) == 0 or pd.isna(value):
        return np.nan
    return 100.0 * np.mean(ref <= float(value))


def make_xgb_model(random_state=42):
    """
    Conservative XGBoost settings for small tabular dataset.
    The goal is not to overfit, but to provide a nonlinear comparison model.
    """
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=120,
        max_depth=2,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=random_state,
        n_jobs=-1,
    )


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(FEATURE_TABLE)
df = df[df["y_true"].isin([0, 1])].copy()
df["y_true"] = df["y_true"].astype(int)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

if "patient_id" in df.columns:
    ID_COL = "patient_id"
elif "image_id" in df.columns:
    ID_COL = "image_id"
elif "filename" in df.columns:
    ID_COL = "filename"
else:
    ID_COL = None

X_raw = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df["y_true"].values

print("Data shape:", df.shape)
print("Class counts:")
print(df["y_true"].value_counts().sort_index())


# ============================================================
# 5-fold OOF XGBoost + SHAP
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

oof_prob = np.zeros(len(df))
oof_pred = np.zeros(len(df), dtype=int)
oof_raw_margin = np.zeros(len(df))

fold_records = []
shap_long_records = []

# collect all OOF SHAP values for global importance
all_oof_shap = np.zeros((len(df), len(FEATURES)))
all_oof_X_imputed = np.zeros((len(df), len(FEATURES)))

for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y), start=1):
    print(f"\nFold {fold}")

    X_train_raw = X_raw.iloc[train_idx].copy()
    X_test_raw = X_raw.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Impute using training fold only
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    X_train_df = pd.DataFrame(X_train_imp, columns=FEATURES)
    X_test_df = pd.DataFrame(X_test_imp, columns=FEATURES)

    model = make_xgb_model(random_state=RANDOM_STATE)
    model.fit(X_train_df, y_train)

    prob = model.predict_proba(X_test_df)[:, 1]
    pred = (prob >= 0.5).astype(int)

    # Raw margin = log-odds before sigmoid
    raw_margin = model.predict(X_test_df, output_margin=True)

    oof_prob[test_idx] = prob
    oof_pred[test_idx] = pred
    oof_raw_margin[test_idx] = raw_margin

    # SHAP values for held-out fold
    # By default for binary XGBoost, TreeExplainer explains raw margin/log-odds.
    explainer = shap.TreeExplainer(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_test_df)

    # Some SHAP versions return a list for classifiers; handle both cases.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    all_oof_shap[test_idx, :] = shap_values
    all_oof_X_imputed[test_idx, :] = X_test_imp

    # Expected value / base value
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).ravel()[-1]
    expected_value = float(expected_value)

    # Check additivity approximately:
    # expected_value + sum(SHAP) should approximate raw margin.
    shap_sum_margin = expected_value + shap_values.sum(axis=1)
    max_additivity_error = np.max(np.abs(shap_sum_margin - raw_margin))

    fold_auc = roc_auc_score(y_test, prob)
    fold_ap = average_precision_score(y_test, prob)
    fold_acc = accuracy_score(y_test, pred)
    fold_f1 = f1_score(y_test, pred)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    fold_records.append({
        "fold": fold,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_pos_test": int(y_test.sum()),
        "n_neg_test": int((1 - y_test).sum()),
        "roc_auc": fold_auc,
        "average_precision": fold_ap,
        "accuracy": fold_acc,
        "f1": fold_f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "shap_expected_value_raw_margin": expected_value,
        "max_shap_additivity_error": max_additivity_error,
    })

    print(
        f"  AUC={fold_auc:.3f}, AP={fold_ap:.3f}, "
        f"Acc={fold_acc:.3f}, F1={fold_f1:.3f}, "
        f"SHAP additivity error={max_additivity_error:.6f}"
    )

    # Reference distribution from training-fold annotation-negative group
    train_ref_neg = X_train_raw.iloc[np.where(y_train == 0)[0]].copy()
    train_ref_all = X_train_raw.copy()

    for local_i, global_i in enumerate(test_idx):
        row = df.iloc[global_i]
        patient_key = row[ID_COL] if ID_COL else global_i

        for j, feature in enumerate(FEATURES):
            raw_value = X_test_raw.iloc[local_i][feature]
            imputed_value = X_test_imp[local_i, j]
            shap_value = shap_values[local_i, j]

            shap_long_records.append({
                "patient_index": global_i,
                "patient_key": patient_key,
                "fold": fold,
                "y_true": int(row["y_true"]),
                "xgb_oof_score": float(prob[local_i]),
                "xgb_oof_pred": int(pred[local_i]),
                "xgb_raw_margin": float(raw_margin[local_i]),
                "shap_expected_value_raw_margin": expected_value,

                "feature": feature,
                "feature_display": DISPLAY_NAMES.get(feature, feature),
                "raw_value": raw_value,
                "imputed_value": float(imputed_value),
                "percentile_vs_train_negative": percentile_rank(train_ref_neg[feature], raw_value),
                "percentile_vs_train_all": percentile_rank(train_ref_all[feature], raw_value),

                # SHAP contribution to raw log-odds margin
                "shap_value_raw_margin": float(shap_value),
                "abs_shap_value": float(abs(shap_value)),
            })


# ============================================================
# Save OOF metrics and predictions
# ============================================================

fold_df = pd.DataFrame(fold_records)
fold_df.to_csv(OUT_DIR / "xgb_shap_fold_metrics.csv", index=False)

overall_auc = roc_auc_score(y, oof_prob)
overall_ap = average_precision_score(y, oof_prob)
overall_acc = accuracy_score(y, oof_pred)
overall_f1 = f1_score(y, oof_pred)
tn, fp, fn, tp = confusion_matrix(y, oof_pred).ravel()

overall_df = pd.DataFrame([{
    "n": len(df),
    "n_pos": int(y.sum()),
    "n_neg": int((1 - y).sum()),
    "roc_auc": overall_auc,
    "average_precision": overall_ap,
    "accuracy": overall_acc,
    "f1": overall_f1,
    "tp": tp,
    "fp": fp,
    "tn": tn,
    "fn": fn,
}])
overall_df.to_csv(OUT_DIR / "xgb_shap_oof_metrics.csv", index=False)

df_pred = df.copy()
df_pred["xgb_oof_score"] = oof_prob
df_pred["xgb_oof_pred"] = oof_pred
df_pred["xgb_raw_margin"] = oof_raw_margin
df_pred.to_csv(OUT_DIR / "xgb_oof_predictions.csv", index=False)

shap_long_df = pd.DataFrame(shap_long_records)
shap_long_df.to_csv(OUT_DIR / "xgb_shap_instance_long.csv", index=False)


# ============================================================
# Global SHAP importance
# ============================================================

importance_records = []

for j, feature in enumerate(FEATURES):
    importance_records.append({
        "feature": feature,
        "feature_display": DISPLAY_NAMES.get(feature, feature),
        "mean_abs_shap": float(np.mean(np.abs(all_oof_shap[:, j]))),
        "mean_shap": float(np.mean(all_oof_shap[:, j])),
    })

importance_df = pd.DataFrame(importance_records)
importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
importance_df.to_csv(OUT_DIR / "xgb_shap_global_importance.csv", index=False)

plt.figure(figsize=(8, 5))
plt.barh(
    importance_df["feature_display"][::-1],
    importance_df["mean_abs_shap"][::-1],
)
plt.xlabel("Mean absolute SHAP value")
plt.title("XGBoost SHAP Global Feature Importance")
plt.tight_layout()
plt.savefig(OUT_DIR / "xgb_shap_global_importance_bar.png", dpi=300)
plt.close()


# ============================================================
# SHAP summary dot plot
# ============================================================

X_oof_df = pd.DataFrame(all_oof_X_imputed, columns=[DISPLAY_NAMES.get(f, f) for f in FEATURES])

plt.figure()
shap.summary_plot(
    all_oof_shap,
    X_oof_df,
    show=False,
    plot_type="dot",
)
plt.tight_layout()
plt.savefig(OUT_DIR / "xgb_shap_summary_dot.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# Patient-level summary
# ============================================================

summary_records = []

for patient_index, g in shap_long_df.groupby("patient_index"):
    g = g.copy()
    first = g.iloc[0]

    top_positive = g.sort_values("shap_value_raw_margin", ascending=False).head(3)
    top_negative = g.sort_values("shap_value_raw_margin", ascending=True).head(3)

    top_pos_text = "; ".join([
        f"{r['feature_display']} ({r['shap_value_raw_margin']:+.3f})"
        for _, r in top_positive.iterrows()
    ])

    top_neg_text = "; ".join([
        f"{r['feature_display']} ({r['shap_value_raw_margin']:+.3f})"
        for _, r in top_negative.iterrows()
    ])

    summary_records.append({
        "patient_index": patient_index,
        "patient_key": first["patient_key"],
        "fold": int(first["fold"]),
        "y_true": int(first["y_true"]),
        "xgb_oof_score": float(first["xgb_oof_score"]),
        "xgb_oof_pred": int(first["xgb_oof_pred"]),
        "xgb_raw_margin": float(first["xgb_raw_margin"]),
        "top_positive_shap_features": top_pos_text,
        "top_negative_shap_features": top_neg_text,
    })

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(OUT_DIR / "xgb_shap_patient_summary.csv", index=False)


# ============================================================
# Train final XGBoost model on all data for future new images
# ============================================================

final_imputer = SimpleImputer(strategy="median")
X_all_imp = final_imputer.fit_transform(X_raw)
X_all_df = pd.DataFrame(X_all_imp, columns=FEATURES)

final_model = make_xgb_model(random_state=RANDOM_STATE)
final_model.fit(X_all_df, y)

dump({
    "features": FEATURES,
    "display_names": DISPLAY_NAMES,
    "imputer": final_imputer,
    "model": final_model,
}, OUT_DIR / "final_xgb_model.joblib")


# ============================================================
# Print summary
# ============================================================

print("\nDone.")
print("\nSaved files:")
print(" -", OUT_DIR / "xgb_shap_oof_metrics.csv")
print(" -", OUT_DIR / "xgb_oof_predictions.csv")
print(" -", OUT_DIR / "xgb_shap_instance_long.csv")
print(" -", OUT_DIR / "xgb_shap_patient_summary.csv")
print(" -", OUT_DIR / "xgb_shap_global_importance.csv")
print(" -", OUT_DIR / "xgb_shap_global_importance_bar.png")
print(" -", OUT_DIR / "xgb_shap_summary_dot.png")
print(" -", OUT_DIR / "final_xgb_model.joblib")

print("\nOverall OOF metrics:")
print(overall_df)

print("\nGlobal SHAP importance:")
print(importance_df.to_string(index=False))