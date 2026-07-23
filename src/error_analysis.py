"""Error analysis for the validation-selected deployment model."""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import config
import metal_properties as mp
import models


def classify_error_cause(row, mol_wt_low, mol_wt_high, top_metals: set[str]) -> str:
    if row.get("primary_metal") not in top_metals:
        return "unusual_metal"
    if row.get("MolWt") < mol_wt_low or row.get("MolWt") > mol_wt_high:
        return "out_of_domain_linker_size"
    if row.get("n_linker_fragments", 1) > 1:
        return "multiple_linker_fragments"
    return "unexplained"


def build_error_table(app_name, master, feature_df, descriptor_df):
    bundle_path = os.path.join(config.RESULTS_MODELS, f"deployment_{app_name}.joblib")
    bundle = joblib.load(bundle_path) if os.path.exists(bundle_path) else models.fit_deployment_model(app_name, master, feature_df)
    model, meta = bundle["model"], bundle["metadata"]
    split = {k: np.asarray(v, dtype=int) for k, v in meta["split_indices"].items()}
    columns = meta["feature_columns"]

    prop = config.APPLICATIONS[app_name]["source_property"]
    cutoff = meta["label_cutoff_value"]
    labels = (master[prop] >= cutoff).astype(int)
    X_test = feature_df.loc[split["test"], columns]
    y_test = labels.loc[split["test"]].to_numpy()
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= meta["selected_probability_threshold"]).astype(int)

    mol_wt_low = descriptor_df.loc[split["train"], "MolWt"].quantile(0.05)
    mol_wt_high = descriptor_df.loc[split["train"], "MolWt"].quantile(0.95)
    train_metals = master.loc[split["train"], "metal_frag"].apply(
        lambda x: mp.metal_fragment_features(x)["primary_metal"]
    )
    top_metals = set(train_metals.value_counts().head(15).index)

    rows = []
    for pos, idx in enumerate(split["test"]):
        if predictions[pos] == y_test[pos]:
            continue
        primary_metal = mp.metal_fragment_features(master.loc[idx, "metal_frag"])["primary_metal"]
        audit_row = {
            "MolWt": descriptor_df.loc[idx, "MolWt"],
            "primary_metal": primary_metal,
            "n_linker_fragments": master.loc[idx, "n_linker_fragments"],
        }
        rows.append({
            "application": app_name,
            "mof_id": int(master.loc[idx, "mof_id"]),
            "precursor": master.loc[idx, "precursor"],
            "true_label": int(y_test[pos]),
            "predicted_label": int(predictions[pos]),
            "probability": float(probabilities[pos]),
            "error_type": "false_positive" if predictions[pos] == 1 else "false_negative",
            "likely_cause": classify_error_cause(audit_row, mol_wt_low, mol_wt_high, top_metals),
            "deployment_model": meta["model_name"],
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))
    full_features = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet"))
    descriptors = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_descriptor_only.parquet"))
    for app in config.APPLICATIONS:
        table = build_error_table(app, master, full_features, descriptors)
        table.to_csv(os.path.join(config.RESULTS_TABLES, f"error_analysis_{app}.csv"), index=False)
        print(app, table.groupby(["error_type", "likely_cause"]).size() if len(table) else "no errors")


if __name__ == "__main__":
    main()
