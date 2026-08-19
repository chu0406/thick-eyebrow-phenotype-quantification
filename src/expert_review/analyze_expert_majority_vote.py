import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_ANALYSIS_ROOT", Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))) / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"))

FORM_CSV = BASE_DIR / "expert_blinded_review_rfe_top5/google_form_latest.csv"
KEY_CSV = BASE_DIR / "expert_blinded_review_rfe_top5/answer_key_do_not_share.csv"
OUT_DIR = BASE_DIR / "expert_blinded_review_rfe_top5/majority_vote_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_answer(x):
    if pd.isna(x):
        return None

    s = str(x).strip().lower()

    if s in ["yes", "y", "是", "有", "visible", "positive"]:
        return "Yes"

    if s in ["no", "n", "否", "沒有", "not visible", "negative"]:
        return "No"

    if "unclear" in s or "cannot" in s or "不確定" in s or "無法" in s or "不清楚" in s:
        return "Unclear"

    return None


def compute_majority(answers):
    """
    Return:
    - expert_label: Yes / No / Ambiguous / Missing
    - majority_ratio
    - consensus_level
    - yes/no/unclear counts
    - tie flag
    """
    answers = [a for a in answers if a in ["Yes", "No", "Unclear"]]

    yes = answers.count("Yes")
    no = answers.count("No")
    unclear = answers.count("Unclear")
    total = yes + no + unclear

    if total == 0:
        return {
            "expert_label": "Missing",
            "majority_ratio": 0.0,
            "consensus_level": "Missing",
            "yes": yes,
            "no": no,
            "unclear": unclear,
            "is_tie": False,
        }

    vote_counts = {
        "Yes": yes,
        "No": no,
        "Unclear": unclear,
    }

    max_vote = max(vote_counts.values())
    winners = [k for k, v in vote_counts.items() if v == max_vote]
    ratio = max_vote / total

    # Tie: no unique majority
    if len(winners) > 1:
        return {
            "expert_label": "Ambiguous",
            "majority_ratio": ratio,
            "consensus_level": "Tie",
            "yes": yes,
            "no": no,
            "unclear": unclear,
            "is_tie": True,
        }

    winner = winners[0]

    # If Unclear is the unique majority, keep it as ambiguous for binary metrics
    if winner == "Unclear":
        return {
            "expert_label": "Ambiguous",
            "majority_ratio": ratio,
            "consensus_level": "Unclear majority",
            "yes": yes,
            "no": no,
            "unclear": unclear,
            "is_tie": False,
        }

    if ratio == 1.0:
        consensus = "Unanimous"
    elif ratio >= 0.75:
        consensus = "Strong"
    else:
        consensus = "Weak"

    return {
        "expert_label": winner,
        "majority_ratio": ratio,
        "consensus_level": consensus,
        "yes": yes,
        "no": no,
        "unclear": unclear,
        "is_tie": False,
    }


# =========================
# Load data
# =========================

form_df = pd.read_csv(FORM_CSV)
key_df = pd.read_csv(KEY_CSV)

form_df = form_df.dropna(how="all").reset_index(drop=True)

print(f"Loaded {len(form_df)} reviewer responses.")



if len(form_df) % 2 == 0:
    print("[INFO] Even number of reviewers detected. Tie-breaking will remove the first reviewer only when no unique majority exists.")
else:
    print("[INFO] Odd number of reviewers detected. Tie-breaking is unlikely unless three-way tie occurs.")


rows = []
long_rows = []

# Google Form columns are 1,2,3,...,20
for i in range(1, 21):
    col = str(i)
    review_id = f"R{i:03d}"

    if col not in form_df.columns:
        raise ValueError(
            f"Missing column {col} in Google Form CSV. "
            f"Available columns: {form_df.columns.tolist()}"
        )

    raw_answers = form_df[col].map(normalize_answer).tolist()

    # First pass: use all reviewers
    first_result = compute_majority(raw_answers)

    tie_resolved_by_dropping_first = False
    final_answers = raw_answers
    final_result = first_result

    # If tied with 8 reviewers, remove first reviewer and recompute
    if first_result["is_tie"]:
        final_answers = raw_answers[1:]
        final_result = compute_majority(final_answers)
        tie_resolved_by_dropping_first = True

    rows.append({
        "review_id": review_id,
        "question_number": i,

        "n_votes_original": sum(a in ["Yes", "No", "Unclear"] for a in raw_answers),
        "yes_votes_original": first_result["yes"],
        "no_votes_original": first_result["no"],
        "unclear_votes_original": first_result["unclear"],
        "original_majority_label": first_result["expert_label"],
        "original_majority_agreement_ratio": first_result["majority_ratio"],
        "original_consensus_level": first_result["consensus_level"],

        "tie_resolved_by_dropping_first_reviewer": tie_resolved_by_dropping_first,

        "n_votes_final": sum(a in ["Yes", "No", "Unclear"] for a in final_answers),
        "yes_votes": final_result["yes"],
        "no_votes": final_result["no"],
        "unclear_votes": final_result["unclear"],
        "expert_majority_label": final_result["expert_label"],
        "majority_agreement_ratio": final_result["majority_ratio"],
        "consensus_level": final_result["consensus_level"],
    })

    for reviewer_idx, ans in enumerate(raw_answers, start=1):
        long_rows.append({
            "review_id": review_id,
            "question_number": i,
            "reviewer_index": reviewer_idx,
            "answer": ans,
            "used_in_final_vote": not (tie_resolved_by_dropping_first and reviewer_idx == 1),
        })

vote_df = pd.DataFrame(rows)
long_df = pd.DataFrame(long_rows)

# =========================
# Merge answer key
# =========================

merged = vote_df.merge(key_df, on="review_id", how="left")

if merged["model_prediction"].isna().any():
    missing = merged[merged["model_prediction"].isna()]["review_id"].tolist()
    raise ValueError(f"These review IDs were not found in answer key: {missing}")

merged["model_label"] = merged["model_prediction"].map({1: "Yes", 0: "No"})

# Binary expert labels only
metric_df = merged[merged["expert_majority_label"].isin(["Yes", "No"])].copy()
metric_df["expert_binary"] = metric_df["expert_majority_label"].map({"Yes": 1, "No": 0})
metric_df["model_binary"] = metric_df["model_label"].map({"Yes": 1, "No": 0})
metric_df["model_score"] = metric_df["model_probability"].astype(float)

# =========================
# Metrics
# =========================

y_true = metric_df["expert_binary"]
y_pred = metric_df["model_binary"]
y_score = metric_df["model_score"]

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

roc_auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None
ap = average_precision_score(y_true, y_score) if len(set(y_true)) == 2 else None

# labels=[1,0] gives:
# [[expert Yes/model Yes, expert Yes/model No],
#  [expert No/model Yes, expert No/model No]]
cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

metrics_df = pd.DataFrame([{
    "n_reviewers_original": len(form_df),
    "n_images_total": len(merged),
    "n_images_used_for_binary_metrics": len(metric_df),
    "n_images_excluded_ambiguous": len(merged) - len(metric_df),
    "n_images_tie_resolved_by_dropping_first_reviewer": int(merged["tie_resolved_by_dropping_first_reviewer"].sum()),
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "average_precision": ap,
    "TP_expertYes_modelYes": cm[0, 0],
    "FN_expertYes_modelNo": cm[0, 1],
    "FP_expertNo_modelYes": cm[1, 0],
    "TN_expertNo_modelNo": cm[1, 1],
}])

consensus_summary = (
    merged["consensus_level"]
    .value_counts()
    .rename_axis("consensus_level")
    .reset_index(name="n_images")
)
consensus_summary["percentage"] = consensus_summary["n_images"] / len(merged) * 100

expert_label_summary = (
    merged["expert_majority_label"]
    .value_counts()
    .rename_axis("expert_majority_label")
    .reset_index(name="n_images")
)
expert_label_summary["percentage"] = expert_label_summary["n_images"] / len(merged) * 100

case_type_summary = (
    merged.groupby(["case_type", "expert_majority_label"])
    .size()
    .reset_index(name="n_images")
)

merged["model_agrees_with_expert"] = merged.apply(
    lambda r: (
        r["model_label"] == r["expert_majority_label"]
        if r["expert_majority_label"] in ["Yes", "No"]
        else None
    ),
    axis=1
)

# =========================
# Save outputs
# =========================

vote_df.to_csv(OUT_DIR / "per_image_expert_votes.csv", index=False)
long_df.to_csv(OUT_DIR / "individual_votes_long_format.csv", index=False)
merged.to_csv(OUT_DIR / "per_image_majority_vote_with_answer_key.csv", index=False)
metrics_df.to_csv(OUT_DIR / "model_vs_expert_majority_metrics.csv", index=False)
consensus_summary.to_csv(OUT_DIR / "consensus_summary.csv", index=False)
expert_label_summary.to_csv(OUT_DIR / "expert_label_summary.csv", index=False)
case_type_summary.to_csv(OUT_DIR / "case_type_by_expert_label_summary.csv", index=False)

print("\nSaved outputs to:")
print(OUT_DIR)

print("\nExpert label summary:")
print(expert_label_summary.to_string(index=False))

print("\nConsensus summary:")
print(consensus_summary.to_string(index=False))

print("\nModel vs expert majority metrics:")
print(metrics_df.T.to_string())

print("\nPer-image summary:")
print(merged[[
    "review_id",
    "case_type",
    "model_label",
    "expert_majority_label",
    "yes_votes_original",
    "no_votes_original",
    "unclear_votes_original",
    "tie_resolved_by_dropping_first_reviewer",
    "yes_votes",
    "no_votes",
    "unclear_votes",
    "majority_agreement_ratio",
    "consensus_level",
    "model_agrees_with_expert"
]].to_string(index=False))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# =========================
# Plot ROC and PR curves
# =========================

if len(metric_df) > 0 and len(set(metric_df["expert_binary"])) == 2:
    y_true = metric_df["expert_binary"].astype(int)
    y_score = metric_df["model_score"].astype(float)

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc_value = auc(fpr, tpr)

    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {roc_auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve: Model vs Expert Majority Label")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "expert_review_roc_curve.png", dpi=300)
    plt.close()

    # Precision-recall curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc_value = auc(recall_curve, precision_curve)

    plt.figure(figsize=(5.5, 5))
    plt.plot(recall_curve, precision_curve, linewidth=2, label=f"AP = {average_precision_score(y_true, y_score):.3f}")
    baseline = y_true.mean()
    plt.axhline(baseline, linestyle="--", linewidth=1, label=f"Baseline = {baseline:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: Model vs Expert Majority Label")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "expert_review_pr_curve.png", dpi=300)
    plt.close()

    # Save curve coordinates
    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": roc_thresholds
    })
    roc_df.to_csv(OUT_DIR / "expert_review_roc_curve_points.csv", index=False)

    pr_df = pd.DataFrame({
        "recall": recall_curve,
        "precision": precision_curve
    })
    pr_df.to_csv(OUT_DIR / "expert_review_pr_curve_points.csv", index=False)

    print("\nSaved ROC / PR curve figures:")
    print(OUT_DIR / "expert_review_roc_curve.png")
    print(OUT_DIR / "expert_review_pr_curve.png")
else:
    print("\nROC / PR curves were not generated because expert labels contain only one class or no valid binary labels.")