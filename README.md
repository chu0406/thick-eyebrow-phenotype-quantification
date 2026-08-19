# Interpretable Quantification of the "Thick Eyebrow" Phenotype

This repository contains the computational analysis code used for
interpretable patient-level classification of the documented
"Thick eyebrow" facial phenotype from rare-disease facial images.

The analysis was developed for the master's thesis project on
image-derived eyebrow phenotype quantification.

---

## Study overview

The computational workflow consists of:

1. Manual cohort curation
2. Patient-level cohort construction
3. Facial landmark detection and face alignment
4. Eyebrow ROI extraction
5. Eyebrow tube construction
6. Relative-darkness-based eyebrow mask generation
7. Interpretable feature extraction
8. Feature-selection experiments
9. Fixed five-feature logistic regression
10. Patient-level five-fold out-of-fold validation
11. Instance-level prediction explanation
12. Alternative interpretable models
13. Blinded expert review

---

## Final cohort

The final balanced cohort contained:

- 137 patients with documented "Thick eyebrow"
- 137 annotation-negative patients without documented
  "Thick eyebrow" or "Synophrys"

One image was retained per patient.

Both groups underwent manual image-quality review for technical
usability and eyebrow visibility.

Raw facial images and identifiable patient-level information are
not included in this repository.

---

## Primary model

The primary classifier was logistic regression using five fixed,
interpretable eyebrow features:

1. Mask / eyebrow tube ratio
2. Eyebrow mask area ratio
3. Local darkness P95
4. PCA mask length normalized by cheek-to-cheek face width
5. Mean PCA eyebrow thickness

Evaluation used patient-level five-fold cross-validation with
out-of-fold predictions.

---

## Repository structure

```text
src/
├── cohort/
│   ├── 37_build_clean_thick_positive_patient_manifest.py
│   ├── 38_prepare_manual_clean_thick_vs_negative_gmdb.py
│   └── 43_build_final_manual_clean_formal_manifest.py
│
├── preprocessing/
│   ├── 32_02_face_align_thick_vs_nonthick.py
│   ├── 32_03_extract_eyebrow_roi_thick_vs_nonthick.py
│   ├── 32_04_generate_eyebrow_mask_v2_thick_vs_nonthick.py
│   └── 32_05_measure_eyebrow_thickness_thick_vs_nonthick.py
│
├── features/
│   ├── 44_07_add_local_density_features_FORMAL.py
│   ├── 44_12_add_eyebrow_length_features_FORMAL.py
│   ├── 44_13_eye_line_face_width_experiment_FORMAL.py
│   ├── 44_14_add_darkness_features_FORMAL.py
│   └── 45_build_formal_merged_model_input.py
│
├── modeling/
│   ├── 54_formal_rfecv_feature_selection_5fold.py
│   └── run_fixed5_instance_explainability.py
│
├── interpretability/
│   ├── run_shallow_tree_interpretability.py
│   ├── run_xgboost_shap_interpretability.py
│   ├── analyze_feature_distribution_formal.py
│   ├── plot_fixed5_per_fold_roc_pr.py
│   └── plot_fixed5_score_enrichment.py
│
└── expert_review/
    └── analyze_expert_majority_vote.py
