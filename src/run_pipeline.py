"""Run the corrected V3 pipeline end-to-end on CPU."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time

import config
import data_prep
import featurize
import splits
import models
import build_compute_table
import confidence_screen
import error_analysis
import run_shap_trend_report
import make_figures


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_manifest(elapsed_seconds):
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "rdkit", "pyarrow", "psutil"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    output_hashes = {}
    for root, _, files in os.walk(config.RESULTS):
        for filename in files:
            path = os.path.join(root, filename)
            output_hashes[os.path.relpath(path, config.ROOT)] = _file_hash(path)
    manifest = {
        "pipeline_version": config.PIPELINE_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": elapsed_seconds,
        "seeds": config.SEEDS,
        "packages": packages,
        "output_sha256": output_hashes,
    }
    with open(os.path.join(config.RESULTS_AUDITS, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main():
    for directory in [
        config.DATA_PROCESSED, config.RESULTS_TABLES, config.RESULTS_FIGURES,
        config.RESULTS_MODELS, config.RESULTS_AUDITS,
    ]:
        os.makedirs(directory, exist_ok=True)
    start = time.perf_counter()

    section("1/9 DATA PREPARATION AND PARSING AUDIT")
    data_prep.main()
    section("2/9 FEATURE CONSTRUCTION")
    featurize.main()
    section("3/9 SPLIT GENERATION AND LEAKAGE AUDIT")
    splits.main()
    section("4/9 MODEL ABLATION, BASELINES, AND DEPLOYMENT SELECTION")
    models.main()
    section("5/9 LOW-COMPUTE PROFILE")
    build_compute_table.main()
    section("6/9 CALIBRATION AND CONFIDENCE SCREENING")
    confidence_screen.main()
    section("7/9 POST-HOC SHAP INTERPRETATION")
    run_shap_trend_report.main()
    section("8/9 ERROR ANALYSIS")
    error_analysis.main()
    section("9/9 FIGURES AND REPRODUCIBILITY MANIFEST")
    make_figures.main()

    elapsed = time.perf_counter() - start
    write_run_manifest(elapsed)
    section(f"DONE IN {elapsed / 60:.1f} MINUTES")
    print(f"Tables:  {config.RESULTS_TABLES}")
    print(f"Figures: {config.RESULTS_FIGURES}")
    print(f"Audits:  {config.RESULTS_AUDITS}")


if __name__ == "__main__":
    main()
