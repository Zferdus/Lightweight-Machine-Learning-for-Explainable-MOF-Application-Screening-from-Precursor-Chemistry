"""Post-hoc SHAP trend report for the saved deployment models."""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import config
import models
import shap_analysis


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))
    features = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet"))

    for app in config.APPLICATIONS:
        path = os.path.join(config.RESULTS_MODELS, f"deployment_{app}.joblib")
        bundle = joblib.load(path) if os.path.exists(path) else models.fit_deployment_model(app, master, features)
        model, meta = bundle["model"], bundle["metadata"]
        split = {key: np.asarray(value, dtype=int) for key, value in meta["split_indices"].items()}
        X_test = features.loc[split["test"], meta["feature_columns"]]
        sample_linkers = master.loc[split["train"], "linker_smiles"].dropna().sample(
            min(500, len(split["train"])), random_state=meta["seed"]
        ).tolist()
        table = shap_analysis.build_trend_table_on_test(model, X_test, sample_linkers)
        table.insert(0, "deployment_model", meta["model_name"])
        table.insert(0, "application", app)
        table.to_csv(os.path.join(config.RESULTS_TABLES, f"shap_trend_{app}.csv"), index=False)
        print(app, meta["model_name"])
        print(table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
