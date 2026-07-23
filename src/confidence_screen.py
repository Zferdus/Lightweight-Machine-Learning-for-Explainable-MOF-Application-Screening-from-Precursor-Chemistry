"""Confidence-calibrated screening using the validation-selected deployment model."""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False
from sklearn.metrics import brier_score_loss
import shap

import config
import models


def confidence_bucket(probability: float) -> str:
    if probability >= config.CONFIDENCE_HIGH:
        return "prioritize"
    if probability >= config.CONFIDENCE_LOW:
        return "needs further testing"
    return "deprioritize"


def _expected_calibration_error(y_true, probabilities, n_bins=10):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(probabilities, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(ece)


def _normalize_binary_shap(values, n_samples, n_features):
    """Return a 2-D (samples, features) matrix for the positive class."""
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # New SHAP: (samples, features, outputs/classes)
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            return arr[:, :, 1 if arr.shape[2] > 1 else 0]
        # Older/list-like layout: (outputs/classes, samples, features)
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            return arr[1 if arr.shape[0] > 1 else 0, :, :]
    raise ValueError(f"Unsupported SHAP shape {arr.shape}; expected a binary-class explanation matrix.")


def _shap_matrix(model, X):
    if hasattr(model, "named_steps"):
        scaler = model.named_steps["standardscaler"]
        estimator = model.named_steps["logisticregression"]
        X_scaled = scaler.transform(X)
        values = shap.LinearExplainer(estimator, X_scaled).shap_values(X_scaled)
    else:
        values = shap.TreeExplainer(model)(X).values
    return _normalize_binary_shap(values, len(X), X.shape[1])


def _top_reason(values, columns, k=2):
    order = np.argsort(np.abs(values))[::-1][:k]
    return " / ".join(("+" if values[i] >= 0 else "-") + str(columns[i]) for i in order)


def build_screening_table(app_name: str, master: pd.DataFrame, feature_df: pd.DataFrame):
    path = os.path.join(config.RESULTS_MODELS, f"deployment_{app_name}.joblib")
    bundle = joblib.load(path) if os.path.exists(path) else models.fit_deployment_model(app_name, master, feature_df)
    raw_model, meta = bundle["model"], bundle["metadata"]
    split = {key: np.asarray(value, dtype=int) for key, value in meta["split_indices"].items()}

    columns = meta["feature_columns"]
    X_valid = feature_df.loc[split["valid"], columns]
    X_test = feature_df.loc[split["test"], columns]
    prop = config.APPLICATIONS[app_name]["source_property"]
    cutoff = float(meta["label_cutoff_value"])
    higher = config.APPLICATIONS[app_name]["higher_is_better"]
    labels = (master[prop] >= cutoff).astype(int) if higher else (master[prop] <= cutoff).astype(int)
    y_valid = labels.loc[split["valid"]].to_numpy()
    y_test = labels.loc[split["test"]].to_numpy()

    raw_test = raw_model.predict_proba(X_test)[:, 1]
    if HAS_FROZEN:
        calibrated = CalibratedClassifierCV(FrozenEstimator(raw_model), method=config.CALIBRATION_METHOD)
    else:
        calibrated = CalibratedClassifierCV(raw_model, method=config.CALIBRATION_METHOD, cv="prefit")
    calibrated.fit(X_valid, y_valid)
    calibrated_test = calibrated.predict_proba(X_test)[:, 1]

    explain_X = X_test.iloc[: min(len(X_test), config.SHAP_EXPLAIN_MAX)]
    explain_values = _shap_matrix(raw_model, explain_X)
    reason_map = {
        int(idx): _top_reason(explain_values[pos], columns)
        for pos, idx in enumerate(explain_X.index)
    }

    threshold = float(meta["selected_probability_threshold"])
    rows = []
    for pos, idx in enumerate(split["test"]):
        probability = float(calibrated_test[pos])
        rows.append({
            "mof_id": int(master.loc[idx, "mof_id"]),
            "precursor": master.loc[idx, "precursor"],
            "true_label": int(y_test[pos]),
            "prediction": "promising" if probability >= threshold else "not promising",
            "decision_threshold": threshold,
            "calibrated_probability": probability,
            "raw_probability": float(raw_test[pos]),
            "shap_reason": reason_map.get(int(idx), "not computed"),
            "action": confidence_bucket(probability),
            "deployment_model": meta["model_name"],
        })
    table = pd.DataFrame(rows).sort_values("calibrated_probability", ascending=False)

    calibration = {
        "application": app_name,
        "deployment_model": meta["model_name"],
        "brier_before": brier_score_loss(y_test, raw_test),
        "brier_after": brier_score_loss(y_test, calibrated_test),
        "ece_before": _expected_calibration_error(y_test, raw_test),
        "ece_after": _expected_calibration_error(y_test, calibrated_test),
        "n_test": len(y_test),
    }

    fraction_pos, mean_pred = calibration_curve(y_test, calibrated_test, n_bins=10, strategy="quantile")
    curve = pd.DataFrame({
        "application": app_name,
        "mean_predicted_probability": mean_pred,
        "observed_positive_fraction": fraction_pos,
    })

    tier = table.groupby("action").agg(
        n=("true_label", "size"),
        observed_positive_rate=("true_label", "mean"),
        mean_calibrated_probability=("calibrated_probability", "mean"),
    ).reset_index()
    tier.insert(0, "application", app_name)
    return table, calibration, curve, tier, calibrated


def main():
    os.makedirs(config.RESULTS_TABLES, exist_ok=True)
    os.makedirs(config.RESULTS_MODELS, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))
    features = pd.read_parquet(os.path.join(config.DATA_PROCESSED, "features_precursor_descriptor.parquet"))

    summaries, curves, tiers = [], [], []
    for app in config.APPLICATIONS:
        table, summary, curve, tier, calibrated = build_screening_table(app, master, features)
        table.to_csv(os.path.join(config.RESULTS_TABLES, f"confidence_screen_{app}.csv"), index=False)
        joblib.dump(calibrated, os.path.join(config.RESULTS_MODELS, f"calibrated_{app}.joblib"))
        summaries.append(summary)
        curves.append(curve)
        tiers.append(tier)
        print(summary)

    pd.DataFrame(summaries).to_csv(os.path.join(config.RESULTS_TABLES, "calibration_summary.csv"), index=False)
    pd.concat(curves, ignore_index=True).to_csv(os.path.join(config.RESULTS_TABLES, "calibration_curve.csv"), index=False)
    pd.concat(tiers, ignore_index=True).to_csv(os.path.join(config.RESULTS_TABLES, "confidence_tier_audit.csv"), index=False)


if __name__ == "__main__":
    main()
