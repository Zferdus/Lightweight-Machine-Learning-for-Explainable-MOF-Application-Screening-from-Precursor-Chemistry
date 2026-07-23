"""Central configuration for the leakage-audited V3 pipeline."""
from __future__ import annotations

import os

PIPELINE_VERSION = "3.0.0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
RESULTS_TABLES = os.path.join(RESULTS, "tables")
RESULTS_FIGURES = os.path.join(RESULTS, "figures")
RESULTS_MODELS = os.path.join(RESULTS, "models")
RESULTS_AUDITS = os.path.join(RESULTS, "audits")

RAW_FILES = {
    "co2_uptake_lp": "core_uptake.csv",
    "ch4_uptake_hp": "core_ch4uptake_highP.csv",
    "logKH_CO2": "core_logKH_CO2.csv",
    "logKH_CH4": "core_logKH_CH4.csv",
    "pore_diameter": "core_di.csv",
    "density": "core_density.csv",
}

SEED = 42
SEEDS = [42, 7, 123]
N_JOBS = 1  # deterministic and avoids OpenMP deadlocks on hosted runtimes

APPLICATIONS = {
    "co2_capture": {
        "source_property": "co2_uptake_lp",
        "percentile_cutoff": 75,
        "higher_is_better": True,
        "display_name": "CO2 high-uptake screening",
    },
    "ch4_storage": {
        "source_property": "ch4_uptake_hp",
        "percentile_cutoff": 75,
        "higher_is_better": True,
        "display_name": "CH4 high-uptake screening",
    },
}

MORGAN_RADIUS = 2
MORGAN_NBITS = 256
SHAP_TOP_K = 15
TOP_METAL_ONEHOT = 15

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
BASE_FEATURE_GROUPS = ["precursor_only", "descriptor_only", "precursor_descriptor"]

TEST_RATIO = 0.15
VALID_RATIO = 0.15
EXCLUDE_UNPARSEABLE_LINKERS = True

HYPERPARAM_GRIDS = {
    "random_forest": [
        {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 1},
        {"n_estimators": 250, "max_depth": 16, "min_samples_leaf": 1},
        {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 2},
    ],
    "xgboost": [
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.10},
        {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.10},
    ],
    "logistic_regression": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
}

METAL_ELEMENT_HOLDOUT_FRAC = 0.20
CALIBRATION_METHOD = "sigmoid"
ENRICHMENT_TOP_FRACTIONS = [0.10, 0.25]

# Classification threshold is selected on validation predictions only.
THRESHOLD_GRID = [round(x, 3) for x in __import__("numpy").linspace(0.05, 0.95, 181)]
THRESHOLD_SELECTION_METRIC = "f1"

# Similarity baselines are tuned on validation data.
KNN_K_GRID = [1, 3, 5, 7, 11]
METAL_SIMILARITY_WEIGHT_GRID = [0.0, 0.25, 0.5, 1.0]

# Greedy scaffold-group assignment target tolerance is audited, not assumed.
SPLIT_SIZE_TOLERANCE = 0.04

# Confidence tiers are descriptive and are evaluated after calibration.
CONFIDENCE_LOW = 0.40
CONFIDENCE_HIGH = 0.70

# Avoid an expensive SHAP pass over every training row.
SHAP_BACKGROUND_MAX = 500
SHAP_EXPLAIN_MAX = 500
