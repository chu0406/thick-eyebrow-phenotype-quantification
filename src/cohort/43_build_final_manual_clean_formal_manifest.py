#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

"""
43_build_final_manual_clean_formal_manifest.py

Build the final formal manifest for the main experiment:

Positive:
    137 manually curated documented Thick Eyebrow GMDB patients.

Negative:
    137 technically usable GMDB patients without documented
    Thick Eyebrow / Synophrys annotation.

Design:
- One image per patient.
- Balanced at patient level.
- Both groups from GMDB.
- Both groups quality-controlled for eyebrow visibility.
"""

from pathlib import Path
import re
import pandas as pd

# ============================================================
# Configuration
# ============================================================

BASE = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

EXP_DIR = (
    BASE
    / "thick_eyebrow_vs_nonthick_gmdb_manual_clean_patient_balanced"
)

POSITIVE_CSV = (
    BASE
    / "thick_eyebrow_clean_positive_patient_check"
    / "manual_clean_positive_one_image_per_patient.csv"
)

NEGATIVE_CSV = (
    EXP_DIR
    / "negative_manual_qc"
    / "formal_negative_137_technically_usable.csv"
)

ALL_NEGATIVE_KEEP_CSV = (
    EXP_DIR
    / "negative_manual_qc"
    / "negative_qc_keep_round1_to_round4_CONFIRMED.csv"
)

METADATA_TSV = (
    BASE
    / "gmdb_metadata"
    / "image_metadata_v1.1.0.tsv"
)

OUT_DIR = EXP_DIR / "formal_final_manifest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MANIFEST_CSV = OUT_DIR / "final_manifest_manual_clean_thick_vs_qc_negative.csv"
FINAL_SUMMARY_CSV = OUT_DIR / "final_dataset_summary.csv"
POSITIVE_DISEASE_CSV = OUT_DIR / "final_positive_disease_distribution.csv"
NEGATIVE_RESERVE_CSV = OUT_DIR / "unused_qc_passed_negative_reserve.csv"

THICK_HPO = "HP:0000574"
SYNOPHRYS_HPO = "HP:0000664"


# ============================================================
# Helpers
# ============================================================

def normalize_image_id(value) -> str:
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).replace("_aligned", "").strip()


def extract_hpo_terms(value) -> set[str]:
    if pd.isna(value):
        return set()
    return set(re.findall(r"HP:\d{7}", str(value)))


def choose_disease_label(row: pd.Series) -> str:
    for col in ["internal_syndrome_name", "disorder_names", "omim_ids"]:
        if col in row.index and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text and text.lower() not in {"nan", "none"}:
                return text
    return "Unknown disease"


def merge_selected_with_metadata(
    selected: pd.DataFrame,
    selected_path_col: str,
    metadata: pd.DataFrame,
    label: str,
    y_true: int,
    curation_status: str,
) -> pd.DataFrame:
    if "image_id" not in selected.columns:
        raise RuntimeError(f"Selected CSV does not contain image_id: {label}")

    if selected_path_col not in selected.columns:
        raise RuntimeError(
            f"Selected CSV does not contain required path column "
            f"'{selected_path_col}': {label}"
        )

    selected = selected.copy()
    selected["image_id_str"] = selected["image_id"].apply(normalize_image_id)
    selected["selected_image_path"] = selected[selected_path_col].astype(str)

    selected_min = selected[
        ["image_id_str", "selected_image_path"]
    ].drop_duplicates(subset=["image_id_str"])

    merged = selected_min.merge(
        metadata,
        on="image_id_str",
        how="left",
        validate="one_to_one",
    )

    if merged["patient_id"].isna().any():
        n_missing = int(merged["patient_id"].isna().sum())
        raise RuntimeError(
            f"{label}: {n_missing} selected images could not be matched to metadata."
        )

    merged["image_path"] = merged["selected_image_path"]
    merged["filename"] = merged["image_path"].apply(
        lambda value: Path(str(value)).name
    )
    merged["image_exists"] = merged["image_path"].apply(
        lambda value: Path(str(value)).exists()
    )

    if not merged["image_exists"].all():
        missing = merged[~merged["image_exists"]][["image_id", "image_path"]]
        print(missing.to_string(index=False))
        raise RuntimeError(f"{label}: some selected image files do not exist.")

    merged["hpo_set"] = merged["present_features"].apply(extract_hpo_terms)
    merged["has_thick_eyebrow"] = merged["hpo_set"].apply(
        lambda terms: THICK_HPO in terms
    )
    merged["has_synophrys"] = merged["hpo_set"].apply(
        lambda terms: SYNOPHRYS_HPO in terms
    )

    merged["label"] = label
    merged["y_true"] = y_true
    merged["source"] = "GMDB"
    merged["image_readable"] = True
    merged["usable"] = True
    merged["curation_status"] = curation_status
    merged["disease_label"] = merged.apply(choose_disease_label, axis=1)

    return merged


# ============================================================
# Main
# ============================================================

def main():
    for required_file in [
        POSITIVE_CSV,
        NEGATIVE_CSV,
        ALL_NEGATIVE_KEEP_CSV,
        METADATA_TSV,
    ]:
        if not required_file.exists():
            raise FileNotFoundError(f"Cannot find required file:\n{required_file}")

    positive_selected = pd.read_csv(POSITIVE_CSV, low_memory=False)
    negative_selected = pd.read_csv(NEGATIVE_CSV, low_memory=False)
    all_negative_keep = pd.read_csv(ALL_NEGATIVE_KEEP_CSV, low_memory=False)

    metadata = pd.read_csv(METADATA_TSV, sep="\t", low_memory=False)
    metadata["image_id_str"] = metadata["image_id"].apply(normalize_image_id)
    metadata = metadata.drop_duplicates(subset=["image_id_str"], keep="first")

    positive = merge_selected_with_metadata(
        selected=positive_selected,
        selected_path_col="clean_image_path",
        metadata=metadata,
        label="Manual_Clean_Thick_Eyebrow",
        y_true=1,
        curation_status="Manual clean positive: eyebrow technically usable",
    )

    negative = merge_selected_with_metadata(
        selected=negative_selected,
        selected_path_col="image_path",
        metadata=metadata,
        label="QC_Usable_No_Documented_Thick_or_Synophrys_GMDB",
        y_true=0,
        curation_status="Manual QC negative: eyebrow technically usable",
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if len(positive) != 137:
        raise RuntimeError(f"Expected 137 positive patients, found {len(positive)}.")

    if len(negative) != 137:
        raise RuntimeError(f"Expected 137 negative patients, found {len(negative)}.")

    if positive["patient_id"].duplicated().any():
        raise RuntimeError("Positive group contains duplicated patient_id.")

    if negative["patient_id"].duplicated().any():
        raise RuntimeError("Negative group contains duplicated patient_id.")

    overlap_patients = set(positive["patient_id"]) & set(negative["patient_id"])
    if overlap_patients:
        raise RuntimeError(
            f"Positive and negative groups share {len(overlap_patients)} patients."
        )

    positive_without_thick = positive[~positive["has_thick_eyebrow"]]
    if len(positive_without_thick) > 0:
        raise RuntimeError(
            f"Positive group contains {len(positive_without_thick)} cases "
            "without Thick Eyebrow HPO."
        )

    invalid_negative = negative[
        negative["has_thick_eyebrow"] | negative["has_synophrys"]
    ]
    if len(invalid_negative) > 0:
        raise RuntimeError(
            f"Negative group contains {len(invalid_negative)} cases with "
            "Thick Eyebrow or Synophrys HPO."
        )

    # --------------------------------------------------------
    # Final manifest
    # --------------------------------------------------------

    output_cols = [
        "image_id",
        "patient_id",
        "image_path",
        "filename",
        "image_exists",
        "label",
        "y_true",
        "source",
        "curation_status",
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
    ]

    output_cols = [
        col for col in output_cols
        if col in positive.columns and col in negative.columns
    ]

    final_manifest = pd.concat(
        [positive[output_cols], negative[output_cols]],
        ignore_index=True,
    )

    final_manifest = final_manifest.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    summary = (
        final_manifest.groupby(["label", "y_true"])
        .agg(
            n_images=("image_path", "size"),
            n_patients=("patient_id", "nunique"),
            n_existing_images=("image_exists", "sum"),
        )
        .reset_index()
    )

    disease_summary = (
        positive.groupby("disease_label", dropna=False)
        .agg(
            n_images=("image_path", "size"),
            n_patients=("patient_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["n_patients", "n_images"],
            ascending=False,
        )
    )

    selected_negative_patient_ids = set(negative["patient_id"])
    reserve_negative = all_negative_keep[
        ~all_negative_keep["patient_id"].isin(selected_negative_patient_ids)
    ].copy()

    final_manifest.to_csv(FINAL_MANIFEST_CSV, index=False)
    summary.to_csv(FINAL_SUMMARY_CSV, index=False)
    disease_summary.to_csv(POSITIVE_DISEASE_CSV, index=False)
    reserve_negative.to_csv(NEGATIVE_RESERVE_CSV, index=False)

    print("=" * 90)
    print("FINAL FORMAL MANUAL-CLEAN PATIENT-BALANCED DATASET")
    print("=" * 90)
    print(summary.to_string(index=False))

    print("\nTotal final images:", len(final_manifest))
    print("Total final patients:", final_manifest["patient_id"].nunique())
    print("Missing image files:", int((~final_manifest["image_exists"]).sum()))
    print("Unused QC-passed negative reserve:", len(reserve_negative))

    print("\nPositive disease distribution:")
    print(disease_summary.head(20).to_string(index=False))

    print("\nSaved:")
    print(FINAL_MANIFEST_CSV)
    print(FINAL_SUMMARY_CSV)
    print(POSITIVE_DISEASE_CSV)
    print(NEGATIVE_RESERVE_CSV)


if __name__ == "__main__":
    main()
