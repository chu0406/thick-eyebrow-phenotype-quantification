import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
)

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_ANALYSIS_ROOT", Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))) / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"))

FEATURE_TABLE = BASE_DIR / "formal_model_input/merged_model_input_features.csv"

OUT_DIR = BASE_DIR / "fixed5_feature_5fold_per_fold_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Fixed 5 features
# 這裡是固定的 5 個 features，不會在每個 fold 重新做 RFE
# 若欄位名稱不符合，script 會印出可能的欄位讓你改
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
    "local_darkness_p95_lr_mean": "Local darkness (P95)",
    "pca_mask_length_norm_by_face_width_cheek_lr_mean": "Normalized mask length",
    "pca_thickness_mean_norm_lr_mean": "Normalized thickness (mean)",
}

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(FEATURE_TABLE)

df = df[df["y_true"].isin([0, 1])].copy()
df["y_true"] = df["y_true"].astype(int)

print("Data shape:", df.shape)
print("Class count:")
print(df["y_true"].value_counts().sort_index())

# ============================================================
# Check feature names
# ============================================================

missing = [c for c in FIXED_FEATURES if c not in df.columns]

if missing:
    print("\n[ERROR] Missing fixed feature columns:")
    for c in missing:
        print(" -", c)

    print("\nAvailable columns containing likely keywords:")
    keywords = [
        "mask",
        "tube",
        "dark",
        "p95",
        "length",
        "thickness",
        "mean",
        "face",
        "cheek",
    ]
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in keywords):
            print(c)

    raise SystemExit(
        "\nPlease update FIXED_FEATURES using the actual column names printed above."
    )

feature_cols = FIXED_FEATURES

X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median(numeric_only=True))
y = df["y_true"].values

# ============================================================
# Stratified 5-fold CV
# 每個 fold 盡量保持 positive / negative 比例平衡
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

oof_score = np.zeros(len(df), dtype=float)
oof_pred = np.zeros(len(df), dtype=int)

fold_rows = []
coef_rows = []
roc_curves = []
pr_curves = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Standardization 必須只 fit 在 training fold
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z = scaler.transform(X_test)

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )

    model.fit(X_train_z, y_train)

    prob = model.predict_proba(X_test_z)[:, 1]
    pred = (prob >= 0.5).astype(int)

    oof_score[test_idx] = prob
    oof_pred[test_idx] = pred

    # ROC for this fold
    fpr, tpr, _ = roc_curve(y_test, prob)
    fold_auc = auc(fpr, tpr)

    # PR for this fold
    precision, recall, _ = precision_recall_curve(y_test, prob)
    fold_ap = average_precision_score(y_test, prob)

    fold_acc = accuracy_score(y_test, pred)
    fold_f1 = f1_score(y_test, pred)

    roc_curves.append({
        "fold": fold,
        "fpr": fpr,
        "tpr": tpr,
        "auc": fold_auc,
    })

    pr_curves.append({
        "fold": fold,
        "precision": precision,
        "recall": recall,
        "ap": fold_ap,
    })

    fold_rows.append({
        "fold": fold,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_pos_train": int(y_train.sum()),
        "n_neg_train": int((1 - y_train).sum()),
        "n_pos_test": int(y_test.sum()),
        "n_neg_test": int((1 - y_test).sum()),
        "roc_auc": fold_auc,
        "average_precision": fold_ap,
        "accuracy": fold_acc,
        "f1": fold_f1,
    })

    for col, coef in zip(feature_cols, model.coef_[0]):
        coef_rows.append({
            "fold": fold,
            "feature": col,
            "display_name": DISPLAY_NAMES.get(col, col),
            "coef": coef,
            "abs_coef": abs(coef),
        })

    print(f"\nFold {fold}")
    print(f"  Train: pos={int(y_train.sum())}, neg={int((1-y_train).sum())}, n={len(train_idx)}")
    print(f"  Test : pos={int(y_test.sum())}, neg={int((1-y_test).sum())}, n={len(test_idx)}")
    print(f"  ROC-AUC={fold_auc:.3f}, AP={fold_ap:.3f}, Acc={fold_acc:.3f}, F1={fold_f1:.3f}")

# ============================================================
# Overall OOF metrics
# ============================================================

overall_auc = roc_auc_score(y, oof_score)
overall_ap = average_precision_score(y, oof_score)
overall_acc = accuracy_score(y, oof_pred)
overall_f1 = f1_score(y, oof_pred)

print("\n=== Overall OOF metrics ===")
print(f"ROC-AUC = {overall_auc:.3f}")
print(f"AP      = {overall_ap:.3f}")
print(f"Acc     = {overall_acc:.3f}")
print(f"F1      = {overall_f1:.3f}")

# ============================================================
# Save tables
# ============================================================

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT_DIR / "fixed5_per_fold_metrics.csv", index=False)

coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(OUT_DIR / "fixed5_per_fold_coefficients.csv", index=False)

coef_summary = (
    coef_df
    .groupby(["feature", "display_name"], as_index=False)
    .agg(
        mean_coef=("coef", "mean"),
        sd_coef=("coef", "std"),
        mean_abs_coef=("abs_coef", "mean"),
    )
    .sort_values("mean_abs_coef", ascending=False)
)
coef_summary.to_csv(OUT_DIR / "fixed5_coefficient_summary.csv", index=False)

pred_df = df.copy()
pred_df["fixed5_oof_score"] = oof_score
pred_df["fixed5_oof_pred"] = oof_pred
pred_df.to_csv(OUT_DIR / "fixed5_oof_predictions.csv", index=False)

with open(OUT_DIR / "fixed5_oof_summary.txt", "w") as f:
    f.write("Fixed 5-feature stratified 5-fold out-of-fold evaluation\n")
    f.write("\nFeatures:\n")
    for c in feature_cols:
        f.write(f"- {DISPLAY_NAMES.get(c, c)} ({c})\n")
    f.write("\nOverall OOF metrics:\n")
    f.write(f"ROC-AUC = {overall_auc:.3f}\n")
    f.write(f"AP = {overall_ap:.3f}\n")
    f.write(f"Accuracy = {overall_acc:.3f}\n")
    f.write(f"F1 = {overall_f1:.3f}\n")

# ============================================================
# Plot ROC: 5 folds + overall OOF
# ============================================================

plt.figure(figsize=(7.4, 6.2))

for item in roc_curves:
    plt.plot(
        item["fpr"],
        item["tpr"],
        linewidth=1.8,
        alpha=0.85,
        label=f"Fold {item['fold']} AUC = {item['auc']:.3f}",
    )

# Overall OOF ROC
fpr_oof, tpr_oof, _ = roc_curve(y, oof_score)
plt.plot(
    fpr_oof,
    tpr_oof,
    color="black",
    linewidth=2.8,
    label=f"Overall OOF AUC = {overall_auc:.3f}",
)

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Chance")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Per-fold ROC Curves: Fixed 5-feature Logistic Regression")
plt.legend(loc="lower right", fontsize=8.5)
plt.grid(alpha=0.3)
plt.tight_layout()

roc_out = OUT_DIR / "fixed5_per_fold_roc_curves.png"
plt.savefig(roc_out, dpi=300)
plt.close()

# ============================================================
# Plot PR: 5 folds + overall OOF
# ============================================================

plt.figure(figsize=(7.4, 6.2))

baseline = y.mean()

for item in pr_curves:
    plt.plot(
        item["recall"],
        item["precision"],
        linewidth=1.8,
        alpha=0.85,
        label=f"Fold {item['fold']} AP = {item['ap']:.3f}",
    )

precision_oof, recall_oof, _ = precision_recall_curve(y, oof_score)
plt.plot(
    recall_oof,
    precision_oof,
    color="black",
    linewidth=2.8,
    label=f"Overall OOF AP = {overall_ap:.3f}",
)

plt.axhline(
    baseline,
    linestyle="--",
    linewidth=1.5,
    label=f"Baseline = {baseline:.2f}",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Per-fold PR Curves: Fixed 5-feature Logistic Regression")
plt.legend(loc="lower left", fontsize=8.5)
plt.grid(alpha=0.3)
plt.tight_layout()

pr_out = OUT_DIR / "fixed5_per_fold_pr_curves.png"
plt.savefig(pr_out, dpi=300)
plt.close()

# ============================================================
# Plot OOF score distribution
# ============================================================

plot_df = pd.DataFrame({
    "group": np.where(y == 1, "GMDB positive", "GMDB negative"),
    "score": oof_score,
    "y": y,
})

group_order = ["GMDB negative", "GMDB positive"]
data = [plot_df.loc[plot_df["group"] == g, "score"].values for g in group_order]

rng = np.random.default_rng(42)

plt.figure(figsize=(6.8, 5.4))
plt.boxplot(data, tick_labels=group_order, showfliers=False)

for i, vals in enumerate(data, start=1):
    plt.scatter(
        rng.normal(i, 0.045, size=len(vals)),
        vals,
        alpha=0.55,
        s=24,
    )

plt.axhline(0.5, linestyle="--", linewidth=1)
plt.ylabel("Out-of-fold predicted probability")
plt.title(
    "Fixed 5-feature OOF model score distribution\n"
    f"ROC-AUC={overall_auc:.3f}, AP={overall_ap:.3f}"
)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

score_out = OUT_DIR / "fixed5_oof_score_distribution.png"
plt.savefig(score_out, dpi=300)
plt.close()

print("\nSaved:")
print(roc_out)
print(pr_out)
print(score_out)
print(OUT_DIR / "fixed5_per_fold_metrics.csv")
print(OUT_DIR / "fixed5_oof_predictions.csv")
print(OUT_DIR / "fixed5_oof_summary.txt")