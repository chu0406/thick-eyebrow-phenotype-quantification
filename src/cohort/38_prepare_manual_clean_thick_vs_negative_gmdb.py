#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

"""
38_prepare_manual_clean_thick_vs_negative_gmdb.py

Prepare the formal manually curated, patient-balanced main experiment:

Positive:
    Manually curated GMDB images with documented Thick Eyebrow annotation.
    One clean image per patient.

Negative:
    GMDB patients without documented Thick Eyebrow or Synophrys annotation.
    One randomly sampled image per patient.

Output:
    Formal manifest for re-running feature extraction and classification.
"""

from pathlib import Path
import re
import pandas as pd

# ============================================================
# Configuration
# ============================================================

BASE = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

CLEAN_POSITIVE_CSV = (
    BASE
    / "thick_eyebrow_clean_positive_patient_check"
    / "manual_clean_positive_one_image_per_patient.csv"
)

METADATA_TSV = (
    BASE
    / "gmdb_metadata"
    / "image_metadata_v1.1.0.tsv"
)

IMAGE_DIR = Path(os.environ.get("GMDB_IMAGE_DIR", str(Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))) / "data" / "gmdb_images")))

OUT_DIR = (
    BASE
    / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MANIFEST = OUT_DIR / "manifest_manual_clean_thick_vs_negative_gmdb.csv"
OUTPUT_SUMMARY = OUT_DIR / "dataset_summary_manual_clean_patient_balanced.csv"
OUTPUT_NEGATIVE_POOL = OUT_DIR / "eligible_negative_patient_pool.csv"
OUTPUT_POSITIVE_DISEASES = OUT_DIR / "manual_clean_positive_disease_distribution.csv"

THICK_HPO = "HP:0000574"
SYNOPHRYS_HPO = "HP:0000664"

RANDOM_STATE = 42


# ============================================================
# Helpers
# ============================================================

def extract_hpo_terms(value) -> set[str]:
    if pd.isna(value):
        return set()
    return set(re.findall(r"HP:\d{7}", str(value)))


def normalize_image_id(value) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).replace("_aligned", "").strip()


def find_original_image_path(image_id) -> str:
    image_id = normalize_image_id(image_id)

    for suffix in [".jpg", ".jpeg", ".png"]:
        path = IMAGE_DIR / f"{image_id}{suffix}"
        if path.exists():
            return str(path)

    return str(IMAGE_DIR / f"{image_id}.jpg")


def choose_disease_label(row: pd.Series) -> str:
    for col in ["internal_syndrome_name", "disorder_names", "omim_ids"]:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value and value.lower() not in {"nan", "none"}:
                return value
    return "Unknown disease"


# ============================================================
# Main
# ============================================================

def main():
    if not CLEAN_POSITIVE_CSV.exists():
        raise FileNotFoundError(f"Cannot find clean positive CSV: {CLEAN_POSITIVE_CSV}")

    if not METADATA_TSV.exists():
        raise FileNotFoundError(f"Cannot find metadata TSV: {METADATA_TSV}")

    positive = pd.read_csv(CLEAN_POSITIVE_CSV, low_memory=False)
    metadata = pd.read_csv(METADATA_TSV, sep="\t", low_memory=False)

    required_positive = ["image_id", "patient_id", "clean_image_path"]
    missing_positive = [c for c in required_positive if c not in positive.columns]
    if missing_positive:
        raise RuntimeError(f"Positive CSV missing columns: {missing_positive}")

    required_metadata = ["image_id", "patient_id", "present_features"]
    missing_metadata = [c for c in required_metadata if c not in metadata.columns]
    if missing_metadata:
        raise RuntimeError(f"Metadata missing columns: {missing_metadata}")

    # --------------------------------------------------------
    # Prepare positive group: already manually curated and
    # already reduced to one clean image per patient.
    # --------------------------------------------------------
    positive = positive.copy()
    positive["image_path"] = positive["clean_image_path"].astype(str)
    positive["filename"] = positive["image_path"].apply(lambda x: Path(x).name)
    positive["image_exists"] = positive["image_path"].apply(lambda x: Path(x).exists())
    positive["label"] = "Manual_Clean_Thick_Eyebrow"
    positive["y_true"] = 1
    positive["source"] = "GMDB"
    positive["image_readable"] = True
    positive["usable"] = True
    positive["has_thick_eyebrow"] = True

    if "has_synophrys" not in positive.columns:
        positive["has_synophrys"] = positive["present_features"].apply(
            lambda value: SYNOPHRYS_HPO in extract_hpo_terms(value)
        )

    positive["disease_label"] = positive.apply(choose_disease_label, axis=1)

    if not positive["image_exists"].all():
        missing = positive[~positive["image_exists"]]
        raise RuntimeError(
            f"Some manual clean positive images do not exist: {len(missing)}"
        )

    if positive["patient_id"].duplicated().any():
        raise RuntimeError("Positive clean cohort still contains duplicated patients.")

    n_positive = positive["patient_id"].nunique()

    # --------------------------------------------------------
    # Prepare full metadata for negative sampling.
    # Exclude any patient who has Thick Eyebrow or Synophrys
    # annotation in any of their images.
    # --------------------------------------------------------
    metadata["hpo_set"] = metadata["present_features"].apply(extract_hpo_terms)
    metadata["has_thick_eyebrow"] = metadata["hpo_set"].apply(
        lambda terms: THICK_HPO in terms
    )
    metadata["has_synophrys"] = metadata["hpo_set"].apply(
        lambda terms: SYNOPHRYS_HPO in terms
    )

    excluded_patient_ids = set(
        metadata.loc[
            metadata["has_thick_eyebrow"] | metadata["has_synophrys"],
            "patient_id"
        ].dropna().tolist()
    )

    negative_pool = metadata[
        ~metadata["patient_id"].isin(excluded_patient_ids)
    ].copy()

    negative_pool["image_path"] = negative_pool["image_id"].apply(find_original_image_path)
    negative_pool["image_exists"] = negative_pool["image_path"].apply(
        lambda x: Path(x).exists()
    )

    negative_pool = negative_pool[negative_pool["image_exists"]].copy()

    # Keep one candidate image per negative patient.
    negative_patient_pool = (
        negative_pool
        .sample(frac=1, random_state=RANDOM_STATE)
        .drop_duplicates(subset=["patient_id"], keep="first")
        .copy()
        .reset_index(drop=True)
    )

    if len(negative_patient_pool) < n_positive:
        raise RuntimeError(
            f"Not enough eligible negative patients: need {n_positive}, "
            f"available {len(negative_patient_pool)}"
        )

    negative = negative_patient_pool.sample(
        n=n_positive,
        replace=False,
        random_state=RANDOM_STATE
    ).copy()

    negative["filename"] = negative["image_path"].apply(lambda x: Path(x).name)
    negative["label"] = "No_Documented_Thick_or_Synophrys_GMDB"
    negative["y_true"] = 0
    negative["source"] = "GMDB"
    negative["image_readable"] = True
    negative["usable"] = True
    negative["disease_label"] = negative.apply(choose_disease_label, axis=1)

    # --------------------------------------------------------
    # Combine formal experiment manifest.
    # --------------------------------------------------------
    keep_cols = [
        "image_id",
        "patient_id",
        "image_path",
        "filename",
        "image_exists",
        "label",
        "y_true",
        "source",
        "disease_label",
        "internal_syndrome_name",
        "disorder_names",
        "omim_ids",
        "present_features",
        "age_year",
        "age_month",
        "gender",
        "ethnicity",
        "has_thick_eyebrow",
        "has_synophrys",
        "image_readable",
        "usable",
    ]

    keep_cols = [
        col for col in keep_cols
        if col in positive.columns and col in negative.columns
    ]

    manifest = pd.concat(
        [positive[keep_cols], negative[keep_cols]],
        ignore_index=True
    )

    manifest = manifest.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    if manifest["patient_id"].duplicated().any():
        raise RuntimeError("Final formal manifest contains duplicated patients.")

    summary = (
        manifest.groupby(["label", "y_true"])
        .agg(
            n_images=("image_path", "size"),
            n_patients=("patient_id", "nunique"),
            n_existing_images=("image_exists", "sum"),
        )
        .reset_index()
    )

    positive_disease_summary = (
        positive.groupby("disease_label", dropna=False)
        .agg(
            n_images=("image_path", "size"),
            n_patients=("patient_id", "nunique"),
        )
        .reset_index()
        .sort_values(["n_patients", "n_images"], ascending=False)
    )

    manifest.to_csv(OUTPUT_MANIFEST, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    negative_patient_pool.to_csv(OUTPUT_NEGATIVE_POOL, index=False)
    positive_disease_summary.to_csv(OUTPUT_POSITIVE_DISEASES, index=False)

    print("=" * 80)
    print("Formal manual-clean patient-balanced experiment")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nPositive disease distribution:")
    print(positive_disease_summary.head(20).to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_MANIFEST)
    print(OUTPUT_SUMMARY)
    print(OUTPUT_NEGATIVE_POOL)
    print(OUTPUT_POSITIVE_DISEASES)

    print("\nNext:")
    print("Use OUTPUT_MANIFEST as the input for a new feature extraction run.")


if __name__ == "__main__":
    main()
