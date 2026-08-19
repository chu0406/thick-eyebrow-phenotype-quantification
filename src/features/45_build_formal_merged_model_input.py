#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

"""
45_build_formal_merged_model_input.py

Merge the completed formal image-level feature tables into one modeling table.

Formal cohort:
- 137 Manual_Clean_Thick_Eyebrow patients
- 137 QC_Usable_No_Documented_Thick_or_Synophrys_GMDB patients

Safety:
- Merge is one-to-one at image_id level.
- Reject duplicated image_id or patient_id.
- Preserve only one copy of metadata columns.
"""

from pathlib import Path
import pandas as pd

# ============================================================
# Paths
# ============================================================

BASE = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

EXP_DIR = (
    BASE
    / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"
)

FEATURE_DIR = EXP_DIR / "formal_features_complete"

OUT_DIR = EXP_DIR / "formal_model_input"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "merged_model_input_features.csv"
SUMMARY_CSV = OUT_DIR / "merged_model_input_summary.csv"

# ============================================================
# Input feature tables
# ============================================================

MANIFEST_CSV = FEATURE_DIR / "image_manifest.csv"

FEATURE_FILES = [
    "eyebrow_thickness_image_features.csv",
    "eyebrow_length_image_features.csv",
    "eyebrow_thickness_eye_line_width_image_features.csv",
    "eyebrow_darkness_image_features.csv",
    "eyebrow_local_density_image_features.csv",
]

EXPECTED_COUNTS = {
    "Manual_Clean_Thick_Eyebrow": 137,
    "QC_Usable_No_Documented_Thick_or_Synophrys_GMDB": 137,
}

# Columns already supplied by manifest and not needed repeatedly from feature CSVs.
META_COLUMNS = {
    "patient_id",
    "image_path",
    "filename",
    "label",
    "y_true",
    "source",
    "image_exists",
    "disease_label",
    "internal_syndrome_name",
    "disorder_names",
    "omim_ids",
    "present_features",
    "absent_features",
    "age_year",
    "age_month",
    "gender",
    "ethnicity",
    "has_thick_eyebrow",
    "has_synophrys",
    "image_readable",
    "usable",
    "curation_status",
}


def normalize_image_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )


def check_table(df: pd.DataFrame, table_name: str):
    required = ["image_id", "patient_id", "label"]  # y_true is supplied by the formal manifest only
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{table_name} missing required columns: {missing}")

    df = df.copy()
    df["image_id"] = normalize_image_id(df["image_id"])

    if df["image_id"].duplicated().any():
        dup = int(df["image_id"].duplicated().sum())
        raise RuntimeError(f"{table_name} has duplicated image_id rows: {dup}")

    if df["patient_id"].duplicated().any():
        dup = int(df["patient_id"].duplicated().sum())
        raise RuntimeError(f"{table_name} has duplicated patient_id rows: {dup}")

    counts = df.groupby("label").size()

    for label, expected_n in EXPECTED_COUNTS.items():
        actual_n = int(counts.get(label, 0))
        if actual_n != expected_n:
            raise RuntimeError(
                f"{table_name}: {label} expected {expected_n}, got {actual_n}"
            )

    return df


def main():
    required_files = [MANIFEST_CSV] + [FEATURE_DIR / f for f in FEATURE_FILES]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Cannot find required formal feature file:\n{path}")

    base = pd.read_csv(MANIFEST_CSV, low_memory=False)
    base = check_table(base, "image_manifest.csv")

    print("=" * 90)
    print("Formal manifest")
    print("=" * 90)
    print(base.groupby(["label", "y_true"]).size())

    merged = base.copy()

    for filename in FEATURE_FILES:
        path = FEATURE_DIR / filename
        feat = pd.read_csv(path, low_memory=False)
        feat = check_table(feat, filename)

        feature_columns = [
            c for c in feat.columns
            if c not in META_COLUMNS and c != "image_id"
        ]

        # Remove columns already merged, except for merge key.
        feature_columns = [
            c for c in feature_columns
            if c not in merged.columns
        ]

        add_df = feat[["image_id"] + feature_columns].copy()

        before = merged.shape
        merged = merged.merge(
            add_df,
            on="image_id",
            how="left",
            validate="one_to_one",
        )
        after = merged.shape

        print(f"\n[MERGE] {filename}")
        print("Added feature columns:", len(feature_columns))
        print("Shape:", before, "->", after)

    # Final checks
    if len(merged) != 274:
        raise RuntimeError(f"Expected 274 merged rows, got {len(merged)}")

    if merged["patient_id"].duplicated().any():
        raise RuntimeError("Merged model input contains duplicated patients.")

    summary = (
        merged.groupby(["label", "y_true"])
        .agg(
            n_images=("image_id", "size"),
            n_patients=("patient_id", "nunique"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_CSV, index=False)
    merged.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 90)
    print("Final formal merged model input")
    print("=" * 90)
    print(summary.to_string(index=False))
    print("\nShape:", merged.shape)
    print("Duplicate patients:", int(merged["patient_id"].duplicated().sum()))

    print("\nSaved:")
    print(OUT_CSV)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
