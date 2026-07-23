from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config
import data_prep
import splits


def main():
    master_path = Path(config.DATA_PROCESSED) / "master_table.csv"
    assert master_path.exists(), "Run data_prep.py first"
    master = pd.read_csv(master_path)

    assert len(master) > 7000
    assert not master["linker_parse_failed"].any()
    assert master["mof_id"].is_unique

    for seed in config.SEEDS:
        split = splits.scaffold_split(master, seed)
        audit = splits.split_audit(master, split, "scaffold", seed)
        assert audit["precursor_overlap_train_test"] == 0
        assert audit["linker_overlap_train_test"] == 0
        assert audit["scaffold_overlap_train_test"] == 0
        assert abs(audit["train_fraction"] - 0.70) < 0.02
        assert abs(audit["valid_fraction"] - 0.15) < 0.02
        assert abs(audit["test_fraction"] - 0.15) < 0.02

        metal = splits.metal_element_holdout_split(master, seed)
        metal_audit = splits.split_audit(master, metal, "metal_element_holdout", seed)
        assert metal_audit["precursor_overlap_train_valid"] == 0
        assert metal_audit["precursor_overlap_train_test"] == 0
        assert len(metal["test_fully_unseen"]) > 0

    for group in config.BASE_FEATURE_GROUPS:
        frame = pd.read_parquet(Path(config.DATA_PROCESSED) / f"features_{group}.parquet")
        assert len(frame) == len(master)
        assert not frame.isna().any().any()
        for app in config.APPLICATIONS:
            data_prep.assert_no_leakage(frame.columns, app)

    print("ALL INTEGRITY TESTS PASSED")


if __name__ == "__main__":
    main()
