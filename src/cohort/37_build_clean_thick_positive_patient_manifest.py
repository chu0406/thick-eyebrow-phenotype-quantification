#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from pathlib import Path
import pandas as pd

BASE = Path(os.environ.get("THICK_EYEBROW_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

CLEAN_DIR = BASE / "thick_eyebrow_p"

# 原始 GMDB metadata；若你的檔案路徑不同，只改這一行
METADATA_CSV = (
    BASE
    / "gmdb_metadata"
    / "image_metadata_v1.1.0.tsv"
)

OUT_DIR = BASE / "thick_eyebrow_clean_positive_patient_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def normalize_image_id(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("_aligned", "")
    return str(stem)


def main():
    if not CLEAN_DIR.exists():
        raise FileNotFoundError(f"Cannot find clean image folder: {CLEAN_DIR}")

    if not METADATA_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find metadata: {METADATA_CSV}\n"
            "Please replace METADATA_CSV with your actual metadata TSV path."
        )

    clean_files = [
        p for p in CLEAN_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    clean_file_df = pd.DataFrame({
        "clean_image_path": [str(p) for p in clean_files],
        "image_id_str": [normalize_image_id(p) for p in clean_files],
    })

    metadata = pd.read_csv(METADATA_CSV, sep="\t", low_memory=False)
    metadata["image_id_str"] = (
        metadata["image_id"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    clean_merged = clean_file_df.merge(
        metadata,
        on="image_id_str",
        how="left",
        validate="one_to_one",
    )

    missing_metadata = clean_merged["patient_id"].isna().sum()

    if missing_metadata > 0:
        print("[WARNING] Clean files without metadata match:", missing_metadata)

    clean_valid = clean_merged.dropna(subset=["patient_id"]).copy()

    print("=" * 80)
    print("Manual clean Thick Eyebrow image set")
    print("=" * 80)
    print("Clean image files:", len(clean_file_df))
    print("Images matched to metadata:", len(clean_valid))
    print("Unique patients:", clean_valid["patient_id"].nunique())
    print("Duplicated patient images:", len(clean_valid) - clean_valid["patient_id"].nunique())

    # Choose one clean image per patient, rather than choosing before curation.
    clean_patient_level = (
        clean_valid
        .sample(frac=1, random_state=RANDOM_STATE)
        .drop_duplicates(subset=["patient_id"], keep="first")
        .copy()
        .reset_index(drop=True)
    )

    disease_col = None
    for col in ["internal_syndrome_name", "disorder_names"]:
        if col in clean_patient_level.columns:
            disease_col = col
            break

    if disease_col:
        disease_summary = (
            clean_patient_level
            .assign(disease=clean_patient_level[disease_col].fillna("Unknown"))
            .groupby("disease", dropna=False)
            .agg(
                n_images=("image_id", "size"),
                n_patients=("patient_id", "nunique"),
            )
            .reset_index()
            .sort_values(["n_patients", "n_images"], ascending=False)
        )
    else:
        disease_summary = pd.DataFrame()

    clean_merged.to_csv(
        OUT_DIR / "manual_clean_184_images_with_metadata.csv",
        index=False,
    )

    clean_patient_level.to_csv(
        OUT_DIR / "manual_clean_positive_one_image_per_patient.csv",
        index=False,
    )

    if not disease_summary.empty:
        disease_summary.to_csv(
            OUT_DIR / "manual_clean_positive_disease_distribution.csv",
            index=False,
        )

        print("\nDisease distribution after one-image-per-patient:")
        print(disease_summary.head(20).to_string(index=False))

    print("\nSaved outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
