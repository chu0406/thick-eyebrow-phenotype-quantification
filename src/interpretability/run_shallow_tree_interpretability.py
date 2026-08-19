import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from joblib import dump


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_ANALYSIS_ROOT", Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))) / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"))

FEATURE_TABLE = BASE_DIR / "formal_model_input/merged_model_input_features.csv"

OUT_DIR = BASE_DIR / "shallow_tree_interpretability"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fixed 5 features
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
# Load data
# ============================================================

df = pd.read_csv(FEATURE_TABLE)
df = df[df["y_true"].isin([0, 1])].copy()
df["y_true"] = df["y_true"].astype(int)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

X_raw = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df["y_true"].values

print("Data shape:", df.shape)
print("Class counts:")
print(df["y_true"].value_counts().sort_index())


# ============================================================
# 5-fold OOF evaluation for shallow decision tree
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
fold_records = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y), start=1):
    X_train_raw = X_raw.iloc[train_idx]
    X_test_raw = X_raw.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Fit imputer only on training fold
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)

    # Shallow decision tree
    # Use raw feature scale so thresholds remain interpretable.
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    tree.fit(X_train, y_train)

    prob = tree.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    oof_prob[test_idx] = prob
    oof_pred[test_idx] = pred

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
    })

    print(f"\nFold {fold}")
    print(f"  AUC={fold_auc:.3f}, AP={fold_ap:.3f}, Acc={fold_acc:.3f}, F1={fold_f1:.3f}")


fold_df = pd.DataFrame(fold_records)
fold_df.to_csv(OUT_DIR / "shallow_tree_fold_metrics.csv", index=False)


# ============================================================
# Overall OOF metrics
# ============================================================

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

overall_df.to_csv(OUT_DIR / "shallow_tree_oof_metrics.csv", index=False)

df_out = df.copy()
df_out["shallow_tree_oof_score"] = oof_prob
df_out["shallow_tree_oof_pred"] = oof_pred
df_out.to_csv(OUT_DIR / "shallow_tree_oof_predictions.csv", index=False)


# ============================================================
# Train final shallow tree on all data for rule visualization
# ============================================================

final_imputer = SimpleImputer(strategy="median")
X_all = final_imputer.fit_transform(X_raw)

final_tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=12,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

final_tree.fit(X_all, y)

feature_names = [DISPLAY_NAMES.get(f, f) for f in FEATURES]

rules = export_text(
    final_tree,
    feature_names=feature_names,
    decimals=4,
)

with open(OUT_DIR / "shallow_tree_rules.txt", "w") as f:
    f.write(rules)

dump({
    "features": FEATURES,
    "display_names": DISPLAY_NAMES,
    "imputer": final_imputer,
    "tree": final_tree,
}, OUT_DIR / "final_shallow_tree_model.joblib")


# ============================================================
# Plot final tree
# ============================================================

plt.figure(figsize=(22, 10))

plot_tree(
    final_tree,
    feature_names=feature_names,
    class_names=["Annotation-negative", "Thick Eyebrow positive"],
    filled=True,
    rounded=True,
    impurity=False,
    proportion=True,
    fontsize=9,
)

plt.title("Shallow Decision Tree for Rule-Based Interpretability", fontsize=16)
plt.tight_layout()
plt.savefig(OUT_DIR / "shallow_tree.png", dpi=300)
plt.close()


# ============================================================
# Print summary
# ============================================================

print("\nDone.")
print("\nSaved files:")
print(" -", OUT_DIR / "shallow_tree_fold_metrics.csv")
print(" -", OUT_DIR / "shallow_tree_oof_metrics.csv")
print(" -", OUT_DIR / "shallow_tree_oof_predictions.csv")
print(" -", OUT_DIR / "shallow_tree_rules.txt")
print(" -", OUT_DIR / "shallow_tree.png")
print(" -", OUT_DIR / "final_shallow_tree_model.joblib")

print("\nOverall OOF metrics:")
print(overall_df)

print("\nFinal shallow tree rules:")
print(rules)