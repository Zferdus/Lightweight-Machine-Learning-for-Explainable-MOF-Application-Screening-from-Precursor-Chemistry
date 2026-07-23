# Lightweight and Explainable Machine Learning for MOF Precursor-Based Screening

This repository contains the corrected and reproducible pipeline associated with the study **“Lightweight and Explainable Machine Learning for Metal–Organic Framework Application Screening Using Precursor Chemistry.”**

The implementation performs **precursor-based high-uptake proxy screening** for CO2 capture and CH4 storage. It does not claim to predict complete industrial application suitability.

## Research question

How much predictive information can be recovered from the metal-node and organic-linker precursors available at synthesis time, without using powder X-ray diffraction, a resolved crystal structure, or GPU-based deep learning?

## Dataset

The workflow uses six row-aligned precursor–property CSV files derived from CoRE MOF 2019 and distributed with the XRayPro resources.

```text
Raw records:       8,571
Modeling records:  7,870
Excluded records:    701
```

Records with unparseable or invalid precursor/linker representations are explicitly audited and excluded rather than silently converted to zero-valued features.

The two binary proxy targets identify the top quartile of:

- low-pressure CO2 uptake;
- high-pressure CH4 uptake.

Each target cutoff is calculated using the training fold only and then applied unchanged to validation and test data.

## Features

The corrected pipeline generates three main representations:

```text
Precursor-only features:        320
Descriptor-only features:        17
Precursor + descriptor features: 337
```

Feature groups include:

- 256-bit Morgan linker fingerprints;
- metal-element indicators;
- metal electronegativity, covalent radius, atomic weight, and atomic number;
- RDKit physicochemical linker descriptors;
- parsed precursor composition and fragment-count features.

## Models and baselines

Classical CPU-only classifiers:

- Logistic Regression
- Random Forest
- XGBoost

Baselines:

- class-prior baseline;
- linker-only Tanimoto k-nearest-neighbor baseline;
- metal-aware precursor-similarity baseline.

Hyperparameters and classification thresholds are selected using the validation fold only.

## Leakage-aware evaluation

Three split strategies are implemented:

1. **Scaffold split — primary evaluation**
2. **Group-aware random split — comparison evaluation**
3. **Metal-element holdout — unseen-metal evaluation**

The primary scaffold split is balanced across all three seeds:

```text
Train:      5,510
Validation: 1,180
Test:       1,180
Seeds:      42, 7, 123
```

For every primary scaffold split:

```text
Train–test precursor overlap: 0
Train–test linker overlap:    0
Train–test scaffold overlap:  0
```

The metal-element holdout evaluation separately reports combined, partially unseen, and fully unseen subsets.

## Final primary results

Three-seed scaffold-split results for the best mean-AUC configuration:

| Application | Feature group | Model | ROC-AUC | F1 | PR-AUC |
|---|---|---|---:|---:|---:|
| CO2 capture proxy | Precursor + descriptor | XGBoost | 0.794 ± 0.020 | 0.565 ± 0.028 | 0.580 ± 0.049 |
| CH4 storage proxy | Precursor + descriptor | XGBoost | 0.828 ± 0.006 | 0.582 ± 0.006 | 0.677 ± 0.022 |

The corrected results are stronger and more stable than the earlier draft results while using a stricter and explicitly audited evaluation protocol.

## Calibration finding

Platt scaling did **not** improve probability calibration in the final run:

| Application | Brier before | Brier after | ECE before | ECE after |
|---|---:|---:|---:|---:|
| CO2 capture proxy | 0.1538 | 0.1550 | 0.0430 | 0.0567 |
| CH4 storage proxy | 0.1422 | 0.1471 | 0.0519 | 0.0745 |

Raw and Platt-scaled probabilities are therefore compared explicitly. Calibrated probabilities should not be presented as uniformly improved.

## Explainability

The pipeline performs:

- train-only SHAP feature selection separately for each seed;
- post-hoc SHAP interpretation of the selected deployment model;
- decoded fingerprint-substructure reporting when available;
- error analysis for unusual metals, multiple-linker precursors, out-of-domain linker sizes, and unexplained errors.

The same selected deployment model is reused consistently for screening, SHAP interpretation, calibration assessment, and error analysis.

## Low-compute profile

Final six-run mean measurements:

| Model | Tuning + fit time | Inference time | Serialized size | GPU |
|---|---:|---:|---:|---:|
| Logistic Regression | 6.37 ± 0.88 s | 4.61 µs/sample | 14.9 KB | No |
| Random Forest | 6.80 ± 0.82 s | 50.78 µs/sample | 20.5 MB | No |
| XGBoost | 3.67 ± 0.84 s | 43.62 µs/sample | 306.3 KB | No |

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── RUN_FULL_PIPELINE_ONE_CELL.py
├── Paper2_Full_Corrected_Colab.ipynb
├── src/
│   ├── run_pipeline.py
│   ├── data_prep.py
│   ├── featurize.py
│   ├── splits.py
│   ├── models.py
│   ├── baseline.py
│   ├── confidence_screen.py
│   ├── shap_analysis.py
│   ├── error_analysis.py
│   └── make_figures.py
├── tests/
│   ├── test_integrity.py
│   └── test_model_smoke.py
├── data/
│   ├── raw/
│   └── processed/
└── results/
    ├── tables/
    ├── figures/
    │   ├── generated/
    │   └── publication/
    ├── audits/
    └── models/
```

Use `results/figures/publication/` for the revised publication-ready PNG and PDF files and retain the automatically generated figures separately if both are uploaded.

## Installation

Python 3.10 or newer is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

RDKit installation may be easier in a clean Conda environment on some systems. The supplied Google Colab notebook is the simplest reproducible execution route.

## Preflight checks

Run the integrity and smoke tests before the full experiment:

```bash
python src/run_checks.py
python tests/test_model_smoke.py
```

## Run the complete pipeline

```bash
python src/run_pipeline.py
```

The pipeline performs all nine stages:

1. data preparation and parsing audit;
2. feature construction;
3. split generation and leakage audit;
4. model ablation, baselines, and deployment selection;
5. low-compute profiling;
6. calibration and confidence screening;
7. post-hoc SHAP interpretation;
8. error analysis;
9. figure generation and reproducibility manifest creation.

The corrected source already includes the deployment split-index serialization fix and modern binary-class SHAP output normalization. No separate hotfix script is required.

## Main outputs

### Result tables

- `results_per_seed.csv`
- `ablation_results_full.csv`
- `predictions_per_run.csv`
- `deployment_model_summary.csv`
- `compute_comparison.csv`
- `calibration_summary.csv`
- `calibration_curve.csv`
- `confidence_tier_audit.csv`
- `shap_trend_co2_capture.csv`
- `shap_trend_ch4_storage.csv`
- `error_analysis_co2_capture.csv`
- `error_analysis_ch4_storage.csv`

### Audits

- raw-data checksums;
- parsing audit;
- feature-schema audit;
- saved split indices;
- precursor/linker/scaffold overlap audit;
- per-seed SHAP-selected features;
- precursor label-conflict ceiling audit;
- package versions and output SHA-256 hashes.

### Publication figures

The main manuscript figure set should include:

1. methodology workflow;
2. CO2 and CH4 primary scaffold-split AUC panels;
3. learned-model versus similarity-baseline panels;
4. CO2 and CH4 SHAP-importance panels;
5. raw-versus-Platt calibration panels.

Metal-holdout, split-audit, and low-compute results may be placed in compact tables or supplementary/repository figures depending on the conference page limit.

## Data provenance

The precursor–property files originate from the XRayPro resources and are derived from CoRE MOF 2019:

- XRayPro code: https://github.com/AI4ChemS/XRayPro
- XRayPro data archive: https://zenodo.org/records/14908210
- CoRE MOF 2019 publication: Chung et al., *Journal of Chemical & Engineering Data*, 2019

Review and comply with the applicable upstream licenses before redistributing third-party raw data.

## Reproducibility

The final run used:

```text
Pipeline version: 3.0.0
Python:           3.12.13
Seeds:            42, 7, 123
Execution:        CPU only
```

Exact package versions and output hashes are stored in `results/audits/run_manifest.json`.

## Limitations

- Labels are top-quartile uptake proxies, not industrial application cutoffs.
- Precursor-only inputs cannot uniquely resolve MOFs that share precursor strings but differ structurally.
- No independent cross-database validation is included in the present study.
- Platt scaling did not improve calibration in the final run.
- Metal-holdout comparisons may have different test compositions and should not be interpreted as directly equivalent to the primary scaffold split.

## Citation

Citation information will be added after publication. Until then, cite the associated manuscript, the upstream datasets, and the repository release used for the analysis.
