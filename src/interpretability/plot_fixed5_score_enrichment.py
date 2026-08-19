import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_ANALYSIS_ROOT", Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))) / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"))

PRED_CSV = BASE_DIR / "fixed5_feature_5fold_per_fold_analysis/fixed5_oof_predictions.csv"

OUT_DIR = BASE_DIR / "fixed5_feature_5fold_per_fold_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PRED_CSV)

score_col = "fixed5_oof_score"
label_col = "y_true"

if score_col not in df.columns:
    raise ValueError(f"Missing column: {score_col}")

if label_col not in df.columns:
    raise ValueError(f"Missing column: {label_col}")

df = df[df[label_col].isin([0, 1])].copy()
df[label_col] = df[label_col].astype(int)

# ============================================================
# Quartile enrichment
# ============================================================

df["score_quartile"] = pd.qcut(
    df[score_col],
    q=4,
    labels=["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"]
)

quartile_summary = (
    df.groupby("score_quartile", observed=False)
    .agg(
        positive_cases=(label_col, "sum"),
        total_cases=(label_col, "count"),
        positive_rate=(label_col, "mean"),
        score_min=(score_col, "min"),
        score_max=(score_col, "max"),
    )
    .reset_index()
)

quartile_summary.to_csv(
    OUT_DIR / "fixed5_score_quartile_positive_rate.csv",
    index=False
)

print("\nQuartile summary:")
print(quartile_summary)

plt.figure(figsize=(8, 5.5))

x = np.arange(len(quartile_summary))
rates = quartile_summary["positive_rate"].values

plt.bar(x, rates)

baseline = df[label_col].mean()
plt.axhline(
    baseline,
    linestyle="--",
    linewidth=1.8,
    label=f"Baseline = {baseline:.0%}"
)

for i, row in quartile_summary.iterrows():
    pct = row["positive_rate"] * 100
    pos = int(row["positive_cases"])
    total = int(row["total_cases"])
    plt.text(
        i,
        row["positive_rate"] + 0.035,
        f"{pct:.1f}%\n({pos}/{total})",
        ha="center",
        va="bottom",
        fontsize=12
    )

plt.xticks(x, quartile_summary["score_quartile"])
plt.ylim(0, 1.05)
plt.ylabel("Documented Thick Eyebrow positive rate")
plt.xlabel("Fixed five-feature OOF score quartile")
plt.title("Positive enrichment across fixed five-feature score quartiles")
plt.legend(loc="upper left")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

out_png = OUT_DIR / "fixed5_score_quartile_positive_rate.png"
plt.savefig(out_png, dpi=300)
plt.close()

print("\nSaved:")
print(out_png)

# ============================================================
# Extreme lowest 50 vs highest 50
# ============================================================

df_sorted = df.sort_values(score_col).copy()

lowest = df_sorted.head(50)
highest = df_sorted.tail(50)

extreme_summary = pd.DataFrame([
    {
        "score_group": "Lowest 50 scores",
        "positive_cases": int(lowest[label_col].sum()),
        "total_cases": len(lowest),
        "positive_rate": lowest[label_col].mean(),
        "score_min": lowest[score_col].min(),
        "score_max": lowest[score_col].max(),
    },
    {
        "score_group": "Highest 50 scores",
        "positive_cases": int(highest[label_col].sum()),
        "total_cases": len(highest),
        "positive_rate": highest[label_col].mean(),
        "score_min": highest[score_col].min(),
        "score_max": highest[score_col].max(),
    },
])

extreme_summary.to_csv(
    OUT_DIR / "fixed5_score_extreme_positive_rate.csv",
    index=False
)

print("\nExtreme summary:")
print(extreme_summary)

plt.figure(figsize=(7.2, 5.5))

x = np.arange(len(extreme_summary))
rates = extreme_summary["positive_rate"].values

plt.bar(x, rates)
plt.axhline(
    baseline,
    linestyle="--",
    linewidth=1.8,
    label=f"Baseline = {baseline:.0%}"
)

for i, row in extreme_summary.iterrows():
    pct = row["positive_rate"] * 100
    pos = int(row["positive_cases"])
    total = int(row["total_cases"])
    plt.text(
        i,
        row["positive_rate"] + 0.035,
        f"{pct:.1f}%\n({pos}/{total})",
        ha="center",
        va="bottom",
        fontsize=12
    )

plt.xticks(x, extreme_summary["score_group"])
plt.ylim(0, 1.05)
plt.ylabel("Documented Thick Eyebrow positive rate")
plt.title("Positive enrichment in highest vs lowest fixed five-feature scores")
plt.legend(loc="upper left")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

out_png2 = OUT_DIR / "fixed5_score_extreme_positive_rate.png"
plt.savefig(out_png2, dpi=300)
plt.close()

print(out_png2)