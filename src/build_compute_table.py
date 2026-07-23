"""Build a transparent low-compute profile from per-seed primary runs."""
from __future__ import annotations

import os
import pandas as pd

import config


def main():
    path = os.path.join(config.RESULTS_TABLES, "results_per_seed.csv")
    df = pd.read_csv(path)
    subset = df[
        (df["split_type"] == "scaffold")
        & (df["test_subset"] == "test")
        & (df["feature_group"] == "precursor_descriptor")
        & (df["model"].isin(config.MODEL_NAMES))
    ]
    out = subset.groupby("model", as_index=False).agg(
        tuning_and_fit_time_sec_mean=("train_time_sec", "mean"),
        tuning_and_fit_time_sec_std=("train_time_sec", "std"),
        inference_us_per_sample_mean=("inference_time_sec_per_sample", lambda s: s.mean() * 1e6),
        peak_process_rss_delta_mb_mean=("peak_rss_delta_mb", "mean"),
        serialized_model_size_kb_mean=("model_size_kb", "mean"),
        n_runs=("seed", "size"),
    )
    out["gpu_used"] = "no"
    out.to_csv(os.path.join(config.RESULTS_TABLES, "compute_comparison.csv"), index=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
