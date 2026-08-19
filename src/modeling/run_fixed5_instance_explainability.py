import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from joblib import dump


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_ANALYSIS_ROOT", Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))) / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"))

FEATURE_TABLE = BASE_DIR / "formal_model_input/merged_model_input_features.csv"

OUT_DIR = BASE_DIR / "fixed5_instance_explainability"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fixed 5 features
# ============================================================

FIXED_FEATURES = [
    "mask_tube_ratio_lr_mean",
    "mask_area_ratio_lr_mean",
    "local_darkness_p95_lr_mean",
    "pca_mask_length_norm_by_face_width_cheek_lr_mean",
    "pca_thickness_mean_norm_lr_mean",
]

DISPLAY_NAMES = {
    "mask_tube_ratio_lr_mean": "Mask / eyebrow tube ratio",
    "mask_area_ratio_lr_mean": "Eyebrow mask area ratio",
    "local_darkness_p95_lr_mean": "Local darkness P95",
    "pca_mask_length_norm_by_face_width_cheek_lr_mean": "Normalized mask length",
    "pca_thickness_mean_norm_lr_mean": "Normalized thickness mean",
}


# ============================================================
# Helpers
# ============================================================

def percentile_rank(reference_values, value):
    """
    Return percentile rank: percentage of reference values <= value.
    Example: 88 means the value is higher than or equal to 88% of reference.
    """
    ref = pd.Series(reference_values).dropna().astype(float).values

    if len(ref) == 0 or pd.isna(value):
        return np.nan

    return 100.0 * np.mean(ref <= float(value))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def top_features_for_patient(feature_rows, top_k=3):
    """
    feature_rows: dataframe for one patient, one row per feature.
    Return top positive and top negative contributors to the linear score.
    """
    tmp = feature_rows.copy()

    pos = tmp.sort_values("logit_contribution", ascending=False).head(top_k)
    neg = tmp.sort_values("logit_contribution", ascending=True).head(top_k)

    pos_text = "; ".join([
        f"{r['feature_display']} ({r['logit_contribution']:+.3f})"
        for _, r in pos.iterrows()
    ])

    neg_text = "; ".join([
        f"{r['feature_display']} ({r['logit_contribution']:+.3f})"
        for _, r in neg.iterrows()
    ])

    return pos_text, neg_text


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(FEATURE_TABLE)
df = df[df["y_true"].isin([0, 1])].copy()
df["y_true"] = df["y_true"].astype(int)

missing = [c for c in FIXED_FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing fixed feature columns: {missing}")

# optional ID column
if "patient_id" in df.columns:
    ID_COL = "patient_id"
elif "image_id" in df.columns:
    ID_COL = "image_id"
elif "filename" in df.columns:
    ID_COL = "filename"
else:
    ID_COL = None

X_raw = df[FIXED_FEATURES].apply(pd.to_numeric, errors="coerce")
y = df["y_true"].values

print("Data shape:", df.shape)
print("Class counts:")
print(df["y_true"].value_counts().sort_index())


# ============================================================
# 5-fold OOF explanation
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

patient_records = []
feature_records = []
fold_records = []

df["fold"] = -1
df["fixed5_oof_score"] = np.nan
df["fixed5_oof_pred"] = np.nan
df["fixed5_oof_logit"] = np.nan

for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y), start=1):
    X_train_raw = X_raw.iloc[train_idx].copy()
    X_test_raw = X_raw.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Imputation fitted only on training fold
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    # StandardScaler fitted only on training fold
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train_imp)
    X_test_z = scaler.transform(X_test_imp)

    # Logistic regression model
    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )
    model.fit(X_train_z, y_train)

    coef = model.coef_[0]
    intercept = float(model.intercept_[0])

    logit = intercept + np.dot(X_test_z, coef)
    prob = sigmoid(logit)
    pred = (prob >= 0.5).astype(int)

    # sanity check with sklearn predict_proba
    prob_sklearn = model.predict_proba(X_test_z)[:, 1]
    if not np.allclose(prob, prob_sklearn, atol=1e-8):
        raise RuntimeError("Manual sigmoid probability does not match sklearn predict_proba.")

    df.loc[df.index[test_idx], "fold"] = fold
    df.loc[df.index[test_idx], "fixed5_oof_score"] = prob
    df.loc[df.index[test_idx], "fixed5_oof_pred"] = pred
    df.loc[df.index[test_idx], "fixed5_oof_logit"] = logit

    fold_auc = roc_auc_score(y_test, prob)
    fold_ap = average_precision_score(y_test, prob)
    fold_acc = accuracy_score(y_test, pred)
    fold_f1 = f1_score(y_test, pred)

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
        "intercept": intercept,
    })

    print(f"\nFold {fold}")
    print(f"  AUC={fold_auc:.3f}, AP={fold_ap:.3f}, Acc={fold_acc:.3f}, F1={fold_f1:.3f}")
    print(f"  Intercept={intercept:.4f}")

    # Reference distributions from training fold only
    train_ref_all = X_train_raw.copy()
    train_ref_neg = X_train_raw.iloc[np.where(y_train == 0)[0]].copy()
    train_ref_pos = X_train_raw.iloc[np.where(y_train == 1)[0]].copy()

    # For each held-out patient, record feature-level explanations
    for local_i, global_i in enumerate(test_idx):
        row = df.iloc[global_i]
        patient_key = row[ID_COL] if ID_COL else global_i

        # contributions to logit
        contributions = X_test_z[local_i, :] * coef

        for j, feature in enumerate(FIXED_FEATURES):
            raw_value = X_test_raw.iloc[local_i][feature]
            imputed_value = X_test_imp[local_i, j]
            z_value = X_test_z[local_i, j]
            beta = coef[j]
            contrib = contributions[j]

            feature_records.append({
                "patient_index": global_i,
                "patient_key": patient_key,
                "fold": fold,
                "y_true": int(row["y_true"]),
                "fixed5_oof_score": float(prob[local_i]),
                "fixed5_oof_pred": int(pred[local_i]),
                "fixed5_oof_logit": float(logit[local_i]),
                "intercept": intercept,

                "feature": feature,
                "feature_display": DISPLAY_NAMES.get(feature, feature),
                "raw_value": raw_value,
                "imputed_value": float(imputed_value),
                "z_value": float(z_value),
                "beta": float(beta),
                "logit_contribution": float(contrib),

                # Percentile against training-fold references
                "percentile_vs_train_all": percentile_rank(train_ref_all[feature], raw_value),
                "percentile_vs_train_negative": percentile_rank(train_ref_neg[feature], raw_value),
                "percentile_vs_train_positive": percentile_rank(train_ref_pos[feature], raw_value),
            })


# ============================================================
# Save long explanations
# ============================================================

feature_expl_df = pd.DataFrame(feature_records)
feature_expl_csv = OUT_DIR / "instance_explainability_long.csv"
feature_expl_df.to_csv(feature_expl_csv, index=False)

fold_df = pd.DataFrame(fold_records)
fold_csv = OUT_DIR / "fold_metrics_and_parameters.csv"
fold_df.to_csv(fold_csv, index=False)

# Overall OOF metrics
overall_auc = roc_auc_score(y, df["fixed5_oof_score"])
overall_ap = average_precision_score(y, df["fixed5_oof_score"])
overall_acc = accuracy_score(y, df["fixed5_oof_pred"])
overall_f1 = f1_score(y, df["fixed5_oof_pred"])

overall_metrics = pd.DataFrame([{
    "n": len(df),
    "n_pos": int(y.sum()),
    "n_neg": int((1 - y).sum()),
    "roc_auc": overall_auc,
    "average_precision": overall_ap,
    "accuracy": overall_acc,
    "f1": overall_f1,
}])
overall_metrics.to_csv(OUT_DIR / "overall_oof_metrics.csv", index=False)


# ============================================================
# Patient-level summary
# ============================================================

summary_records = []

for patient_index, g in feature_expl_df.groupby("patient_index"):
    g = g.copy()
    first = g.iloc[0]

    top_pos, top_neg = top_features_for_patient(g, top_k=3)

    # percentile summary for key features
    percentile_text = "; ".join([
        f"{r['feature_display']}: {r['percentile_vs_train_negative']:.1f}% vs neg"
        for _, r in g.iterrows()
    ])

    summary_records.append({
        "patient_index": patient_index,
        "patient_key": first["patient_key"],
        "fold": int(first["fold"]),
        "y_true": int(first["y_true"]),
        "fixed5_oof_score": float(first["fixed5_oof_score"]),
        "fixed5_oof_pred": int(first["fixed5_oof_pred"]),
        "fixed5_oof_logit": float(first["fixed5_oof_logit"]),
        "top_positive_contributors": top_pos,
        "top_negative_contributors": top_neg,
        "percentile_summary_vs_train_negative": percentile_text,
    })

summary_df = pd.DataFrame(summary_records)
summary_csv = OUT_DIR / "instance_explainability_patient_summary.csv"
summary_df.to_csv(summary_csv, index=False)

# Save all cases with score
all_cases_csv = OUT_DIR / "all_cases_with_fixed5_oof_score.csv"
df.to_csv(all_cases_csv, index=False)


# ============================================================
# Train final model on all data for future new images
# ============================================================

final_imputer = SimpleImputer(strategy="median")
X_imp_all = final_imputer.fit_transform(X_raw)

final_scaler = StandardScaler()
X_z_all = final_scaler.fit_transform(X_imp_all)

final_model = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    max_iter=5000,
)
final_model.fit(X_z_all, y)

dump({
    "features": FIXED_FEATURES,
    "display_names": DISPLAY_NAMES,
    "imputer": final_imputer,
    "scaler": final_scaler,
    "model": final_model,
}, OUT_DIR / "final_fixed5_logistic_model.joblib")

# Save full-cohort reference distributions for future new images
reference_records = []

for feature in FIXED_FEATURES:
    for ref_name, mask in [
        ("all", np.ones(len(df), dtype=bool)),
        ("annotation_negative", df["y_true"].values == 0),
        ("positive", df["y_true"].values == 1),
    ]:
        values = X_raw.loc[mask, feature].dropna().astype(float)

        reference_records.append({
            "feature": feature,
            "feature_display": DISPLAY_NAMES.get(feature, feature),
            "reference_group": ref_name,
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "p05": values.quantile(0.05),
            "p25": values.quantile(0.25),
            "p50": values.quantile(0.50),
            "p75": values.quantile(0.75),
            "p95": values.quantile(0.95),
            "min": values.min(),
            "max": values.max(),
        })

ref_df = pd.DataFrame(reference_records)
ref_df.to_csv(OUT_DIR / "feature_reference_distribution_summary.csv", index=False)

print("\nDone.")
print("Saved:")
print(" -", feature_expl_csv)
print(" -", summary_csv)
print(" -", OUT_DIR / "overall_oof_metrics.csv")
print(" -", OUT_DIR / "final_fixed5_logistic_model.joblib")
print(" -", OUT_DIR / "feature_reference_distribution_summary.csv")
print("\nOverall OOF metrics:")
print(overall_metrics)