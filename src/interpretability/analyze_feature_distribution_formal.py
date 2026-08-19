import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


# ============================================================
# 0. Paths
# ============================================================

BASE_DIR = Path(os.environ.get("THICK_EYEBROW_ANALYSIS_ROOT", Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))) / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"))

FEATURE_TABLE = BASE_DIR / "formal_model_input/merged_model_input_features.csv"
FEATURE_PANEL = BASE_DIR / "classification_compact_features_FORMAL/compact_feature_panel_used.csv"

OUT_DIR = BASE_DIR / "feature_distribution_analysis_FORMAL"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Utility functions
# ============================================================

def cohen_d(x_pos, x_neg):
    x_pos = np.asarray(x_pos, dtype=float)
    x_neg = np.asarray(x_neg, dtype=float)

    n1 = len(x_pos)
    n0 = len(x_neg)

    if n1 < 2 or n0 < 2:
        return np.nan

    s1 = np.var(x_pos, ddof=1)
    s0 = np.var(x_neg, ddof=1)

    pooled = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2))

    if pooled == 0:
        return np.nan

    return (np.mean(x_pos) - np.mean(x_neg)) / pooled


def cliffs_delta(x_pos, x_neg):
    x_pos = np.asarray(x_pos, dtype=float)
    x_neg = np.asarray(x_neg, dtype=float)

    greater = 0
    less = 0

    for xp in x_pos:
        greater += np.sum(xp > x_neg)
        less += np.sum(xp < x_neg)

    total = len(x_pos) * len(x_neg)

    if total == 0:
        return np.nan

    return (greater - less) / total


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    order = np.argsort(pvals)
    ranked = pvals[order]

    adjusted = np.empty(n, dtype=float)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        prev = min(prev, value)
        adjusted[order[i]] = prev

    return np.minimum(adjusted, 1.0)


def short_name(name):
    name = str(name)
    name = name.replace("Normalized ", "Norm. ")
    name = name.replace("Eyebrow ", "")
    name = name.replace(" / eyebrow tube ratio", "/tube ratio")
    name = name.replace("Local darkness", "Darkness")
    return name


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(FEATURE_TABLE)
panel = pd.read_csv(FEATURE_PANEL)

print("Feature table:", FEATURE_TABLE)
print("Feature panel:", FEATURE_PANEL)
print("Feature table shape:", df.shape)
print("Feature panel:")
print(panel.to_string(index=False))

# label: y_true should be 1 = documented Thick Eyebrow, 0 = no documented annotation
if "y_true" not in df.columns:
    raise ValueError("Cannot find y_true column in feature table.")

df = df[df["y_true"].isin([0, 1])].copy()

print("\nGroup counts:")
print(df["y_true"].value_counts().sort_index())


# ============================================================
# 3. Get formal features
# ============================================================

required_cols = ["feature", "display_name"]
for c in required_cols:
    if c not in panel.columns:
        raise ValueError(f"Missing column '{c}' in {FEATURE_PANEL}")

features = []

for _, row in panel.iterrows():
    feature_col = row["feature"]
    display_name = row["display_name"]

    if feature_col not in df.columns:
        raise ValueError(
            f"Feature column not found in merged feature table: {feature_col}\n"
            f"Please check compact_feature_panel_used.csv"
        )

    features.append({
        "feature": feature_col,
        "display_name": display_name,
        "group": row["group"] if "group" in panel.columns else "",
        "concept": row["concept"] if "concept" in panel.columns else "",
    })

print("\nUsing formal feature columns:")
for f in features:
    print(f"{f['display_name']} -> {f['feature']}")


# ============================================================
# 4. Distribution statistics
# ============================================================

rows = []

for f in features:
    col = f["feature"]
    display_name = f["display_name"]

    values = pd.to_numeric(df[col], errors="coerce")
    tmp = pd.DataFrame({
        "y_true": df["y_true"],
        "value": values
    }).dropna()

    pos = tmp.loc[tmp["y_true"] == 1, "value"].values
    neg = tmp.loc[tmp["y_true"] == 0, "value"].values

    u_stat, p_value = mannwhitneyu(pos, neg, alternative="two-sided")

    d = cohen_d(pos, neg)
    delta = cliffs_delta(pos, neg)

    auc_raw = roc_auc_score(tmp["y_true"], tmp["value"])
    auc_directional = max(auc_raw, 1 - auc_raw)

    rows.append({
        "concept": f["concept"],
        "group": f["group"],
        "feature": col,
        "display_name": display_name,

        "n_positive": len(pos),
        "n_negative": len(neg),

        "positive_mean": np.mean(pos),
        "negative_mean": np.mean(neg),
        "mean_difference_pos_minus_neg": np.mean(pos) - np.mean(neg),

        "positive_sd": np.std(pos, ddof=1),
        "negative_sd": np.std(neg, ddof=1),

        "positive_median": np.median(pos),
        "negative_median": np.median(neg),
        "median_difference_pos_minus_neg": np.median(pos) - np.median(neg),

        "positive_q1": np.percentile(pos, 25),
        "positive_q3": np.percentile(pos, 75),
        "negative_q1": np.percentile(neg, 25),
        "negative_q3": np.percentile(neg, 75),

        "mannwhitney_u": u_stat,
        "p_value": p_value,
        "cohen_d": d,
        "cliffs_delta": delta,
        "univariate_auc_raw": auc_raw,
        "univariate_auc_directional": auc_directional,

        "higher_group_by_mean": "Positive" if np.mean(pos) > np.mean(neg) else "Negative",
    })

summary = pd.DataFrame(rows)
summary["fdr_bh"] = benjamini_hochberg(summary["p_value"].fillna(1.0).values)

summary = summary.sort_values("univariate_auc_directional", ascending=False)

summary_path = OUT_DIR / "feature_distribution_summary_FORMAL.csv"
summary.to_csv(summary_path, index=False)

print("\nSaved:")
print(summary_path)


# ============================================================
# 5. Plot all formal features: boxplots
# ============================================================

n_features = len(features)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
axes = np.array(axes).reshape(-1)

rng = np.random.default_rng(42)

for ax, f in zip(axes, features):
    col = f["feature"]
    display_name = f["display_name"]

    values = pd.to_numeric(df[col], errors="coerce")

    neg = values[df["y_true"] == 0].dropna().values
    pos = values[df["y_true"] == 1].dropna().values

    ax.boxplot(
        [neg, pos],
        tick_labels=["No documented\nannotation", "Documented\nThick Eyebrow"],
        showfliers=False
    )

    ax.scatter(rng.normal(1, 0.04, size=len(neg)), neg, alpha=0.25, s=12)
    ax.scatter(rng.normal(2, 0.04, size=len(pos)), pos, alpha=0.25, s=12)

    stat = summary.loc[summary["feature"] == col].iloc[0]

    ax.set_title(
        f"{short_name(display_name)}\n"
        f"d={stat['cohen_d']:.2f}, "
        f"δ={stat['cliffs_delta']:.2f}, "
        f"FDR={stat['fdr_bh']:.2g}"
    )

    ax.set_ylabel("Feature value")
    ax.grid(axis="y", alpha=0.3)

for ax in axes[n_features:]:
    ax.axis("off")

fig.suptitle(
    "Formal eyebrow feature distributions by GMDB annotation group",
    fontsize=16
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

all_boxplot_path = OUT_DIR / "feature_distribution_all_FORMAL_boxplots.png"
fig.savefig(all_boxplot_path, dpi=300)
plt.close(fig)

print("Saved:")
print(all_boxplot_path)


# ============================================================
# 6. Plot top stable features
# ============================================================

stable_keywords = [
    "local_darkness_p95",
    "mask_tube_ratio",
    "mask_area_ratio",
    "pca_mask_length",
]

stable_features = []
for f in features:
    if any(k in f["feature"] for k in stable_keywords):
        stable_features.append(f)

# remove duplicates while preserving order
seen = set()
stable_features_unique = []
for f in stable_features:
    if f["feature"] not in seen:
        stable_features_unique.append(f)
        seen.add(f["feature"])

stable_features = stable_features_unique[:4]

if len(stable_features) > 0:
    fig, axes = plt.subplots(1, len(stable_features), figsize=(5 * len(stable_features), 5))

    if len(stable_features) == 1:
        axes = [axes]

    for ax, f in zip(axes, stable_features):
        col = f["feature"]
        display_name = f["display_name"]

        values = pd.to_numeric(df[col], errors="coerce")

        neg = values[df["y_true"] == 0].dropna().values
        pos = values[df["y_true"] == 1].dropna().values

        ax.boxplot(
            [neg, pos],
            tick_labels=["No documented\nannotation", "Documented\nThick Eyebrow"],
            showfliers=False
        )

        ax.scatter(rng.normal(1, 0.04, size=len(neg)), neg, alpha=0.25, s=12)
        ax.scatter(rng.normal(2, 0.04, size=len(pos)), pos, alpha=0.25, s=12)

        stat = summary.loc[summary["feature"] == col].iloc[0]

        ax.set_title(
            f"{short_name(display_name)}\n"
            f"d={stat['cohen_d']:.2f}, "
            f"δ={stat['cliffs_delta']:.2f}, "
            f"FDR={stat['fdr_bh']:.2g}"
        )

        ax.set_ylabel("Feature value")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Distributions of stable selected features", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    stable_path = OUT_DIR / "top_stable_feature_boxplots_FORMAL.png"
    fig.savefig(stable_path, dpi=300)
    plt.close(fig)

    print("Saved:")
    print(stable_path)


# ============================================================
# 7. Effect size bar plot
# ============================================================

effect_df = summary.copy()
effect_df["abs_delta"] = effect_df["cliffs_delta"].abs()
effect_df = effect_df.sort_values("abs_delta", ascending=True)

fig, ax = plt.subplots(figsize=(10, 0.55 * len(effect_df) + 2))

ax.barh(effect_df["display_name"], effect_df["cliffs_delta"])
ax.axvline(0, linestyle="--", linewidth=1)
ax.set_xlabel("Cliff's delta\npositive value = higher in documented Thick Eyebrow group")
ax.set_title("Effect size of formal eyebrow features")
ax.grid(axis="x", alpha=0.3)

fig.tight_layout()

effect_path = OUT_DIR / "feature_effect_size_bar_FORMAL.png"
fig.savefig(effect_path, dpi=300)
plt.close(fig)

print("Saved:")
print(effect_path)


# ============================================================
# 8. Feature correlation heatmap
# ============================================================

feature_cols = [f["feature"] for f in features]
feature_names = [f["display_name"] for f in features]

feature_matrix = df[feature_cols].apply(pd.to_numeric, errors="coerce")
corr = feature_matrix.corr(method="spearman")

fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(corr.values, vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(feature_names)))
ax.set_yticks(np.arange(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(feature_names, fontsize=8)
ax.set_title("Spearman correlation among formal eyebrow features")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()

corr_path = OUT_DIR / "feature_correlation_heatmap_FORMAL.png"
fig.savefig(corr_path, dpi=300)
plt.close(fig)

print("Saved:")
print(corr_path)


# ============================================================
# 9. Compact printout
# ============================================================

print("\n=== Compact summary for slides ===")
print(
    summary[
        [
            "display_name",
            "positive_median",
            "negative_median",
            "median_difference_pos_minus_neg",
            "cohen_d",
            "cliffs_delta",
            "p_value",
            "fdr_bh",
            "univariate_auc_directional",
            "higher_group_by_mean",
        ]
    ].to_string(index=False)
)