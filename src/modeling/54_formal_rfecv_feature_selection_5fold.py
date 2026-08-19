#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

"""
54_formal_rfecv_feature_selection_5fold.py

Formal RFECV feature-selection experiment for documented Thick Eyebrow classification.

Purpose:
1. Use the same 11 compact interpretable eyebrow features.
2. Compare:
   - All 11 features Logistic Regression
   - RFECV-selected Logistic Regression
3. Perform feature selection inside each training fold to avoid data leakage.
4. Evaluate by 5-fold patient-level out-of-fold predictions.

Input:
formal_model_input/merged_model_input_features.csv

Output:
formal_rfecv_feature_selection_5fold/
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV, RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================
# Paths
# ============================================================

BASE = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

EXP_DIR = (
    BASE
    / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"
)

INPUT_CSV = EXP_DIR / "formal_model_input" / "merged_model_input_features.csv"

OUT_DIR = EXP_DIR / "formal_rfecv_feature_selection_5fold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = OUT_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Compact 11 features
# ============================================================

FEATURES = [
    "pca_thickness_p95_norm_lr_mean",
    "pca_thickness_mean_norm_lr_mean",
    "pca_thickness_p95_norm_lr_absdiff",
    "mask_area_ratio_lr_mean",
    "mask_tube_ratio_lr_mean",
    "landmark_brow_length_norm_by_face_oval_width_lr_mean",
    "pca_mask_length_norm_by_face_oval_width_lr_mean",
    "local_darkness_p90_lr_max",
    "local_darkness_p95_lr_max",
    "local_darkness_mean_lr_max",
    "mask_darkness_mean_lr_max",
]

DISPLAY_NAMES = {
    "pca_thickness_p95_norm_lr_mean": "Normalized thickness (P95)",
    "pca_thickness_mean_norm_lr_mean": "Normalized thickness (mean)",
    "pca_thickness_p95_norm_lr_absdiff": "Thickness asymmetry",
    "mask_area_ratio_lr_mean": "Eyebrow mask area ratio",
    "mask_tube_ratio_lr_mean": "Mask / eyebrow tube ratio",
    "landmark_brow_length_norm_by_face_oval_width_lr_mean": "Normalized eyebrow length",
    "pca_mask_length_norm_by_face_oval_width_lr_mean": "Normalized mask length",
    "local_darkness_p90_lr_max": "Local darkness (P90)",
    "local_darkness_p95_lr_max": "Local darkness (P95)",
    "local_darkness_mean_lr_max": "Local darkness (mean)",
    "mask_darkness_mean_lr_max": "Mask darkness (mean)",
}


# ============================================================
# Settings
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

# Logistic Regression teacher / main estimator
BASE_LR = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    class_weight="balanced",
    max_iter=5000,
    random_state=RANDOM_STATE,
)

OUTER_CV = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

INNER_CV = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# Helper functions
# ============================================================

def evaluate_binary(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def build_preprocess_model(feature_names):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clone(BASE_LR)),
        ]
    )


def fit_predict_all_features(X_train, y_train, X_valid):
    pipe = build_preprocess_model(X_train.columns.tolist())
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_valid)[:, 1]
    return prob, pipe


def fit_predict_rfecv_features(X_train, y_train, X_valid):
    """
    RFECV is performed only on the training fold.
    Scaling and imputation are also fit only on the training fold.

    Because sklearn RFECV does not directly operate on a full pipeline with
    feature names as cleanly here, we manually preprocess inside the fold.
    """

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    Xtr_imp = imputer.fit_transform(X_train)
    Xtr_scaled = scaler.fit_transform(Xtr_imp)

    Xva_imp = imputer.transform(X_valid)
    Xva_scaled = scaler.transform(Xva_imp)

    rfecv = RFECV(
        estimator=clone(BASE_LR),
        step=1,
        cv=INNER_CV,
        scoring="roc_auc",
        min_features_to_select=1,
        n_jobs=-1,
    )

    rfecv.fit(Xtr_scaled, y_train)

    selected_mask = rfecv.support_
    selected_features = X_train.columns[selected_mask].tolist()

    # Train final LR on selected features in this training fold
    final_lr = clone(BASE_LR)
    final_lr.fit(Xtr_scaled[:, selected_mask], y_train)

    prob = final_lr.predict_proba(Xva_scaled[:, selected_mask])[:, 1]

    return prob, {
        "imputer": imputer,
        "scaler": scaler,
        "rfecv": rfecv,
        "final_lr": final_lr,
        "selected_features": selected_features,
        "selected_mask": selected_mask,
    }


def fit_predict_rfe_fixed_k(X_train, y_train, X_valid, k):
    """
    Optional fixed-k RFE inside each outer fold.
    Useful to compare top-3, top-5, top-7 feature subsets.
    """

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    Xtr_imp = imputer.fit_transform(X_train)
    Xtr_scaled = scaler.fit_transform(Xtr_imp)

    Xva_imp = imputer.transform(X_valid)
    Xva_scaled = scaler.transform(Xva_imp)

    rfe = RFE(
        estimator=clone(BASE_LR),
        n_features_to_select=k,
        step=1,
    )
    rfe.fit(Xtr_scaled, y_train)

    selected_mask = rfe.support_
    selected_features = X_train.columns[selected_mask].tolist()

    final_lr = clone(BASE_LR)
    final_lr.fit(Xtr_scaled[:, selected_mask], y_train)

    prob = final_lr.predict_proba(Xva_scaled[:, selected_mask])[:, 1]

    return prob, {
        "imputer": imputer,
        "scaler": scaler,
        "rfe": rfe,
        "final_lr": final_lr,
        "selected_features": selected_features,
        "selected_mask": selected_mask,
    }


def save_feature_frequency_plot(freq_df, out_png, title):
    plot_df = freq_df.sort_values(
        ["selection_count", "mean_rank"],
        ascending=[True, False],
    ).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["display_name"], plot_df["selection_count"])
    ax.set_xlabel(f"Selected folds out of {N_SPLITS}")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlim(0, N_SPLITS)
    ax.grid(axis="x", alpha=0.25)

    for i, value in enumerate(plot_df["selection_count"]):
        ax.text(value + 0.05, i, str(value), va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find input CSV:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    required = ["patient_id", "label", "y_true"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if df["patient_id"].duplicated().sum() != 0:
        raise RuntimeError(
            "Duplicate patient_id found. Formal experiment must be one image per patient."
        )

    X = df[FEATURES].copy()
    y = df["y_true"].astype(int).values

    print("=" * 90)
    print("Formal RFECV feature-selection experiment")
    print("=" * 90)
    print("\nInput:")
    print(INPUT_CSV)

    print("\nDataset:")
    print(
        df.groupby(["label", "y_true"])
        .agg(n_images=("image_id", "size"), n_patients=("patient_id", "nunique"))
        .reset_index()
        .to_string(index=False)
    )

    print("\nFeature set:")
    for i, f in enumerate(FEATURES, 1):
        print(f"{i:02d}. {DISPLAY_NAMES[f]}  [{f}]")

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    all_oof_prob = np.zeros(len(df), dtype=float)
    rfecv_oof_prob = np.zeros(len(df), dtype=float)

    fixed_k_list = [3, 5, 7, 9]
    fixed_oof = {
        k: np.zeros(len(df), dtype=float)
        for k in fixed_k_list
    }

    fold_rows = []
    rfecv_selected_rows = []
    fixed_selected_rows = []

    # --------------------------------------------------------
    # Outer 5-fold CV
    # --------------------------------------------------------

    for fold, (train_idx, valid_idx) in enumerate(OUTER_CV.split(X, y), 1):
        print("\n" + "-" * 90)
        print(f"Outer fold {fold}/{N_SPLITS}")
        print("-" * 90)

        X_train = X.iloc[train_idx].copy()
        y_train = y[train_idx]
        X_valid = X.iloc[valid_idx].copy()
        y_valid = y[valid_idx]

        # All 11 features baseline
        prob_all, model_all = fit_predict_all_features(
            X_train,
            y_train,
            X_valid,
        )
        all_oof_prob[valid_idx] = prob_all
        metrics_all = evaluate_binary(y_valid, prob_all)

        print("[All 11 features]")
        print(
            f"ROC-AUC={metrics_all['roc_auc']:.3f}, "
            f"PR-AUC={metrics_all['pr_auc']:.3f}, "
            f"F1={metrics_all['f1']:.3f}"
        )

        fold_rows.append(
            {
                "fold": fold,
                "method": "All 11 features",
                "n_features": len(FEATURES),
                "selected_features": "; ".join(FEATURES),
                **metrics_all,
            }
        )

        # RFECV selected features
        prob_rfecv, rfecv_info = fit_predict_rfecv_features(
            X_train,
            y_train,
            X_valid,
        )
        rfecv_oof_prob[valid_idx] = prob_rfecv
        metrics_rfecv = evaluate_binary(y_valid, prob_rfecv)

        selected = rfecv_info["selected_features"]

        print("[RFECV-selected features]")
        print("Selected feature count:", len(selected))
        for f in selected:
            print(" -", DISPLAY_NAMES[f])
        print(
            f"ROC-AUC={metrics_rfecv['roc_auc']:.3f}, "
            f"PR-AUC={metrics_rfecv['pr_auc']:.3f}, "
            f"F1={metrics_rfecv['f1']:.3f}"
        )

        fold_rows.append(
            {
                "fold": fold,
                "method": "RFECV selected",
                "n_features": len(selected),
                "selected_features": "; ".join(selected),
                **metrics_rfecv,
            }
        )

        ranking = rfecv_info["rfecv"].ranking_
        support = rfecv_info["selected_mask"]

        for f, rank, is_selected in zip(FEATURES, ranking, support):
            rfecv_selected_rows.append(
                {
                    "fold": fold,
                    "feature": f,
                    "display_name": DISPLAY_NAMES[f],
                    "selected": bool(is_selected),
                    "rank": int(rank),
                }
            )

        # Fixed-k RFE variants
        for k in fixed_k_list:
            prob_k, info_k = fit_predict_rfe_fixed_k(
                X_train,
                y_train,
                X_valid,
                k=k,
            )
            fixed_oof[k][valid_idx] = prob_k
            metrics_k = evaluate_binary(y_valid, prob_k)
            selected_k = info_k["selected_features"]

            fold_rows.append(
                {
                    "fold": fold,
                    "method": f"RFE top-{k}",
                    "n_features": k,
                    "selected_features": "; ".join(selected_k),
                    **metrics_k,
                }
            )

            ranking_k = info_k["rfe"].ranking_
            support_k = info_k["selected_mask"]

            for f, rank, is_selected in zip(FEATURES, ranking_k, support_k):
                fixed_selected_rows.append(
                    {
                        "fold": fold,
                        "k": k,
                        "feature": f,
                        "display_name": DISPLAY_NAMES[f],
                        "selected": bool(is_selected),
                        "rank": int(rank),
                    }
                )

    # --------------------------------------------------------
    # Overall OOF metrics
    # --------------------------------------------------------

    overall_rows = []

    metrics_all_oof = evaluate_binary(y, all_oof_prob)
    overall_rows.append(
        {
            "method": "All 11 features",
            "n_features": len(FEATURES),
            **metrics_all_oof,
        }
    )

    metrics_rfecv_oof = evaluate_binary(y, rfecv_oof_prob)
    overall_rows.append(
        {
            "method": "RFECV selected",
            "n_features": "fold-dependent",
            **metrics_rfecv_oof,
        }
    )

    for k in fixed_k_list:
        metrics_k_oof = evaluate_binary(y, fixed_oof[k])
        overall_rows.append(
            {
                "method": f"RFE top-{k}",
                "n_features": k,
                **metrics_k_oof,
            }
        )

    overall_df = pd.DataFrame(overall_rows)
    fold_df = pd.DataFrame(fold_rows)
    rfecv_sel_df = pd.DataFrame(rfecv_selected_rows)
    fixed_sel_df = pd.DataFrame(fixed_selected_rows)

    # --------------------------------------------------------
    # Feature selection summary
    # --------------------------------------------------------

    rfecv_freq = (
        rfecv_sel_df.groupby(["feature", "display_name"])
        .agg(
            selection_count=("selected", "sum"),
            mean_rank=("rank", "mean"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
        .reset_index()
        .sort_values(["selection_count", "mean_rank"], ascending=[False, True])
    )

    fixed_freq = (
        fixed_sel_df.groupby(["k", "feature", "display_name"])
        .agg(
            selection_count=("selected", "sum"),
            mean_rank=("rank", "mean"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
        .reset_index()
        .sort_values(["k", "selection_count", "mean_rank"], ascending=[True, False, True])
    )

    # --------------------------------------------------------
    # OOF prediction table
    # --------------------------------------------------------

    pred_df = df[
        [
            c for c in [
                "image_id",
                "patient_id",
                "label",
                "y_true",
                "disease_label",
                "internal_syndrome_name",
                "disorder_names",
                "image_path",
            ]
            if c in df.columns
        ]
    ].copy()

    pred_df["prob_all_11_features"] = all_oof_prob
    pred_df["pred_all_11_features"] = (all_oof_prob >= 0.5).astype(int)

    pred_df["prob_rfecv_selected"] = rfecv_oof_prob
    pred_df["pred_rfecv_selected"] = (rfecv_oof_prob >= 0.5).astype(int)

    for k in fixed_k_list:
        pred_df[f"prob_rfe_top_{k}"] = fixed_oof[k]
        pred_df[f"pred_rfe_top_{k}"] = (fixed_oof[k] >= 0.5).astype(int)

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    overall_path = TABLE_DIR / "rfecv_overall_oof_metrics.csv"
    fold_path = TABLE_DIR / "rfecv_fold_metrics.csv"
    rfecv_selected_path = TABLE_DIR / "rfecv_selected_features_by_fold.csv"
    rfecv_freq_path = TABLE_DIR / "rfecv_feature_selection_frequency.csv"
    fixed_selected_path = TABLE_DIR / "rfe_fixed_k_selected_features_by_fold.csv"
    fixed_freq_path = TABLE_DIR / "rfe_fixed_k_feature_selection_frequency.csv"
    pred_path = TABLE_DIR / "rfecv_oof_predictions.csv"

    overall_df.to_csv(overall_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    rfecv_sel_df.to_csv(rfecv_selected_path, index=False)
    rfecv_freq.to_csv(rfecv_freq_path, index=False)
    fixed_sel_df.to_csv(fixed_selected_path, index=False)
    fixed_freq.to_csv(fixed_freq_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    save_feature_frequency_plot(
        rfecv_freq,
        PLOT_DIR / "rfecv_feature_selection_frequency.png",
        "RFECV Feature Selection Frequency Across 5 Folds",
    )

    # Overall metric comparison plot
    metric_plot_df = overall_df.copy()
    metric_order = ["roc_auc", "pr_auc", "accuracy", "f1"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_plot_df))
    width = 0.18

    for i, metric in enumerate(metric_order):
        ax.bar(
            x + (i - 1.5) * width,
            metric_plot_df[metric].astype(float),
            width,
            label=metric.upper().replace("_", "-"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_plot_df["method"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("All Features vs RFE / RFECV Feature Selection", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "rfecv_model_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("Overall out-of-fold metrics")
    print("=" * 90)
    print(
        overall_df[
            [
                "method",
                "n_features",
                "roc_auc",
                "pr_auc",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "sensitivity_recall",
                "specificity",
                "f1",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("RFECV selected feature frequency")
    print("=" * 90)
    print(rfecv_freq.to_string(index=False))

    print("\nSaved outputs:")
    print(overall_path)
    print(fold_path)
    print(rfecv_selected_path)
    print(rfecv_freq_path)
    print(fixed_selected_path)
    print(fixed_freq_path)
    print(pred_path)
    print(PLOT_DIR / "rfecv_feature_selection_frequency.png")
    print(PLOT_DIR / "rfecv_model_performance_comparison.png")

    print("\nOpen output folder:")
    print(f"open {OUT_DIR}")


if __name__ == "__main__":
    main()
