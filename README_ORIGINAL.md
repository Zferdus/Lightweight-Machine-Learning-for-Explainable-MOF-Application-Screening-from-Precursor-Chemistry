# MOF Screening V3 — Leakage-Audited Corrected Pipeline

This folder is a **self-contained update of the supplied project**. It includes the six raw XRayPro/CoRE-derived CSV files, corrected source code, fast integrity tests, a Colab runner, and the previous V2 tables archived separately.

The V3 run must generate new final results. Do **not** use `results_v2_archive/` as the final paper result.

## What V3 fixes

1. **Balanced scaffold split**
   - Acyclic linkers no longer collapse into one empty-scaffold group.
   - Whole scaffold groups are greedily assigned to approximately 70/15/15 train/validation/test partitions.
   - Exact precursor, linker, and scaffold overlap audits are saved for every seed.

2. **Per-seed SHAP feature selection**
   - Top features are selected separately from each seed's training fold.
   - Seed 42 feature choices are not reused for seeds 7 and 123.

3. **Metal-holdout leakage fix**
   - Held-out metal elements are absent from training.
   - Train/validation are grouped by canonical precursor.
   - Combined, partial-unseen, and fully-unseen metal test subsets are evaluated separately.

4. **Correct degeneracy analysis**
   - Raw property variation is reported separately from classification-label conflict.
   - For each split and application, V3 reports conflicting precursor-label groups and a majority-label upper bound.

5. **Improved precursor parser and parsing audit**
   - Any fragment containing a recognized MOF metal is treated as a node fragment even when carbon is present.
   - Auxiliary and ambiguous fragments are retained and audited.
   - Multi-linker and parse-failure counts are saved.

6. **No data-fitted metal vocabulary**
   - The former global “top-15 metals” one-hot vocabulary is replaced by a fixed indicator schema covering all recognized metals.

7. **Real multi-seed behavior**
   - Both split seed and model `random_state` use the current seed.
   - Aggregate SD uses sample SD (`ddof=1`) when more than one seed is available.
   - Seed-level metrics and predictions are saved.

8. **Validation-selected classification threshold**
   - Hyperparameters are selected by validation ROC-AUC.
   - The probability threshold for F1 is then selected on validation predictions only and frozen for test evaluation.

9. **Stronger baselines**
   - Class-prior baseline.
   - Validation-tuned linker-only Tanimoto kNN.
   - Validation-tuned metal-aware precursor-similarity kNN.

10. **Selected model reused downstream**
    - The validation-selected precursor+descriptor deployment model is saved.
    - Calibration, confidence screening, SHAP interpretation, and error analysis reuse that saved model.

11. **Better calibration and compute reporting**
    - Brier score and expected calibration error are reported before/after calibration.
    - Confidence-tier observed positive rates are audited.
    - Process RSS sampling replaces `tracemalloc` for native-library memory.

12. **Reproducibility outputs**
    - Raw file SHA-256 manifest.
    - Split-index JSON files.
    - Feature schema audit.
    - Per-seed predictions.
    - Saved deployment/calibrated models.
    - Environment/run manifest and automatically generated figures.

## Important scientific wording

The targets are **top-quartile uptake proxy labels**, not complete industrial application-suitability labels. Recommended manuscript wording:

- “CO2 high-uptake screening proxy”
- “CH4 high-uptake screening proxy”
- “precursor-only proxy screening”

Do not claim that precursor chemistry alone proves full application suitability.

## Fast integrity check

From the project root:

```bash
python src/run_checks.py
python tests/test_model_smoke.py
```

The packaged project passed both checks before delivery.

## Full CPU run

```bash
python -m venv venv
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
python src/run_pipeline.py
```

No GPU is required. The full run performs many validation-tuned fits, three scaffold seeds, SHAP-selected ablations, three baselines, comparison splits, calibration, SHAP interpretation, error analysis, and figure generation. Runtime depends heavily on CPU and may be roughly **45–120 minutes**. The pipeline uses one worker by default for deterministic behavior and to avoid OpenMP deadlocks on hosted runtimes.

## Main new outputs

```text
results/
├── tables/
│   ├── results_per_seed.csv
│   ├── ablation_results_full.csv
│   ├── predictions_per_run.csv
│   ├── deployment_model_summary.csv
│   ├── calibration_summary.csv
│   ├── calibration_curve.csv
│   ├── confidence_tier_audit.csv
│   ├── confidence_screen_<application>.csv
│   ├── shap_trend_<application>.csv
│   ├── error_analysis_<application>.csv
│   └── compute_comparison.csv
├── audits/
│   ├── split_audit.csv
│   ├── split_indices_*.json
│   ├── label_conflict_ceiling_audit.csv
│   ├── shap_selected_features_per_seed.csv
│   ├── precursor_parsing_audit.csv
│   ├── raw_data_manifest.csv
│   └── run_manifest.json
├── models/
│   ├── deployment_<application>.joblib
│   └── calibrated_<application>.joblib
└── figures/
    ├── primary_auc_<application>.png/.pdf
    ├── baseline_comparison.png/.pdf
    ├── scaffold_split_balance.png/.pdf
    └── calibration_reliability.png/.pdf
```

## Verified packaged-data audit

The supplied data produced:

- 8,571 raw aligned rows.
- 7,870 modeling rows after the stricter parser excluded 701 unparseable/empty linker rows.
- Exactly balanced scaffold splits for each packaged seed: 5,510 train, 1,180 validation, 1,180 test.
- Zero canonical-precursor overlap between scaffold train and test.
- Zero canonical-linker overlap between scaffold train and test.
- Zero scaffold-group overlap between scaffold train and test.

Because the parser is stricter than V2, the modeling count differs from the older paper draft’s 7,966. The manuscript must be updated using the V3 rerun outputs rather than retaining the older count.


## Colab-ready execution

Use `MOF_Screening_V3_Colab_READY.ipynb`. It mounts Drive, locates or uploads the full project ZIP, extracts the project to a fixed Drive path, installs dependencies, runs preflight checks, archives stale outputs, executes the CPU pipeline, verifies figures/tables/models, and downloads a final reproducibility package.
