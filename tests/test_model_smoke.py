from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config
import models
import splits


def main():
    master = pd.read_csv(Path(config.DATA_PROCESSED) / "master_table.csv")
    features = pd.read_parquet(Path(config.DATA_PROCESSED) / "features_descriptor_only.parquet")
    split = splits.scaffold_split(master, seed=config.SEED)
    _, metrics, predictions = models.train_and_eval(
        "logistic_regression",
        features,
        master,
        "co2_capture",
        split,
        seed=config.SEED,
    )
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert len(predictions) == len(split["test"])
    print("MODEL SMOKE TEST PASSED")
    print({k: metrics[k] for k in ["roc_auc", "pr_auc", "f1", "selected_probability_threshold"]})


if __name__ == "__main__":
    main()
