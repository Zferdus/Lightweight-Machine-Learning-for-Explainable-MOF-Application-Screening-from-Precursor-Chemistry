"""Model tuning, leakage-safe evaluation, multi-seed aggregation, and deployment selection."""
from __future__ import annotations

import gc
import json
import os
import pickle
import threading
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import baseline as baseline_mod
import config
import data_prep
import shap_analysis
import splits as splits_mod


METRIC_KEYS = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "enrichment_top10", "enrichment_top25", "train_time_sec",
    "inference_time_sec_per_sample", "peak_rss_delta_mb", "model_size_kb",
]


def _build_model(model_name: str, params: dict, seed: int):
    if model_name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, solver="liblinear", random_state=seed, **params
            ),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=config.N_JOBS, **params)
    if model_name == "xgboost":
        return xgb.XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            n_jobs=config.N_JOBS,
            tree_method="hist",
            **params,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _safe_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan


def _enrichment_factor(y_true, y_prob, top_fraction):
    y_true = np.asarray(y_true)
    n_top = max(1, int(np.ceil(len(y_true) * top_fraction)))
    top = np.argsort(y_prob)[::-1][:n_top]
    base = y_true.mean()
    return float(y_true[top].mean() / base) if base > 0 else np.nan


def select_probability_threshold(y_valid, valid_prob):
    best_threshold, best_score = 0.5, -1.0
    for threshold in config.THRESHOLD_GRID:
        score = f1_score(y_valid, valid_prob >= threshold, zero_division=0)
        if score > best_score + 1e-12:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def _measure_call(func):
    """Measure wall time and process RSS peak delta, including native allocations."""
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error:
                pass
            stop.wait(0.02)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = func()
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join(timeout=1)
    peak_delta_mb = max(0.0, (peak[0] - baseline) / 1024**2)
    return result, elapsed, peak_delta_mb


def tune_and_fit(model_name, X_train, y_train, X_valid, y_valid, seed: int):
    candidates = []
    for params in config.HYPERPARAM_GRIDS[model_name]:
        model = _build_model(model_name, params, seed)
        model.fit(X_train, y_train)
        valid_prob = model.predict_proba(X_valid)[:, 1]
        valid_auc = _safe_auc(y_valid, valid_prob)
        candidates.append((np.nan_to_num(valid_auc, nan=-1.0), params, model, valid_prob))
    best_auc, best_params, best_model, best_valid_prob = max(candidates, key=lambda x: x[0])
    threshold, valid_f1 = select_probability_threshold(y_valid, best_valid_prob)
    return best_model, best_params, float(best_auc), threshold, valid_f1


def _evaluate_predictions(y_true, y_prob, threshold):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": _safe_auc(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "enrichment_top10": _enrichment_factor(y_true, y_prob, 0.10),
        "enrichment_top25": _enrichment_factor(y_true, y_prob, 0.25),
        "y_pred": y_pred,
    }


def train_and_eval(
    model_name: str,
    feature_df: pd.DataFrame,
    master: pd.DataFrame,
    app_name: str,
    split_idx: dict,
    seed: int,
    test_key: str = "test",
):
    data_prep.assert_no_leakage(feature_df.columns, app_name)
    labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)

    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx[test_key]
    if len(test_idx) == 0:
        return None, None, None

    X_train, y_train = feature_df.loc[train_idx], labels.loc[train_idx].to_numpy()
    X_valid, y_valid = feature_df.loc[valid_idx], labels.loc[valid_idx].to_numpy()
    X_test, y_test = feature_df.loc[test_idx], labels.loc[test_idx].to_numpy()

    def fit_call():
        return tune_and_fit(model_name, X_train, y_train, X_valid, y_valid, seed)

    (model, best_params, valid_auc, threshold, valid_f1), train_time, peak_rss = _measure_call(fit_call)

    # Warm up before timing the measured inference pass.
    if len(X_test):
        model.predict_proba(X_test.iloc[: min(8, len(X_test))])
    start = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - start

    evaluated = _evaluate_predictions(y_test, y_prob, threshold)
    y_pred = evaluated.pop("y_pred")
    metrics = {
        **evaluated,
        "train_time_sec": train_time,
        "inference_time_sec_total": inference_time,
        "inference_time_sec_per_sample": inference_time / max(1, len(y_test)),
        "peak_rss_delta_mb": peak_rss,
        "model_size_kb": len(pickle.dumps(model)) / 1024,
        "best_params": json.dumps(best_params, sort_keys=True),
        "validation_auc": valid_auc,
        "validation_f1_at_selected_threshold": valid_f1,
        "selected_probability_threshold": threshold,
        "label_cutoff_value": cutoff,
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "n_test": len(test_idx),
        "train_positive_rate": float(labels.loc[train_idx].mean()),
        "valid_positive_rate": float(labels.loc[valid_idx].mean()),
        "test_positive_rate": float(labels.loc[test_idx].mean()),
    }

    predictions = pd.DataFrame({
        "mof_id": master.loc[test_idx, "mof_id"].astype(int).to_numpy(),
        "row_index": np.asarray(test_idx, dtype=int),
        "y_true": y_test,
        "y_probability": y_prob,
        "y_pred": y_pred,
    })
    return model, metrics, predictions


def _result_row(metadata: dict, metrics: dict):
    return {**metadata, **metrics}


def _aggregate_per_seed(per_seed: pd.DataFrame):
    group_cols = ["application", "feature_group", "model", "split_type", "test_subset"]
    rows = []
    for keys, group in per_seed.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = group["seed"].nunique()
        for metric in METRIC_KEYS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if values.notna().sum() > 1 else 0.0
        for metric in [
            "validation_auc", "validation_f1_at_selected_threshold",
            "selected_probability_threshold", "label_cutoff_value",
            "n_train", "n_valid", "n_test",
            "train_positive_rate", "valid_positive_rate", "test_positive_rate",
        ]:
            row[f"{metric}_mean"] = pd.to_numeric(group[metric], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _run_feature_group_evaluation(
    master,
    feature_groups,
    split_fn,
    split_name,
    seeds,
    test_keys=("test",),
):
    result_rows, prediction_frames, conflict_rows = [], [], []
    for seed in seeds:
        split_idx = split_fn(master, seed=seed)
        splits_mod.save_split_indices(split_idx, split_name, seed)
        for app_name in config.APPLICATIONS:
            labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)
            conflict_rows.extend(
                splits_mod.label_conflict_audit(master, labels, split_idx, app_name, cutoff, split_name, seed)
            )
            for group_name, feature_df in feature_groups.items():
                for model_name in config.MODEL_NAMES:
                    for test_key in test_keys:
                        if test_key not in split_idx:
                            continue
                        model, metrics, predictions = train_and_eval(
                            model_name, feature_df, master, app_name, split_idx, seed, test_key=test_key
                        )
                        if metrics is None:
                            continue
                        metadata = {
                            "application": app_name,
                            "feature_group": group_name,
                            "model": model_name,
                            "split_type": split_name,
                            "test_subset": test_key,
                            "seed": seed,
                        }
                        result_rows.append(_result_row(metadata, metrics))
                        predictions = predictions.assign(**metadata)
                        prediction_frames.append(predictions)
                        print(
                            f"[{split_name}:{test_key}][seed={seed}][{app_name}] "
                            f"{group_name}/{model_name}: AUC={metrics['roc_auc']:.3f}, "
                            f"F1={metrics['f1']:.3f}, threshold={metrics['selected_probability_threshold']:.3f}"
                        )
                        del model
                        gc.collect()
    return pd.DataFrame(result_rows), prediction_frames, conflict_rows


def _run_shap_selected(master, full_features):
    rows, predictions_all, feature_audit, conflict_rows = [], [], [], []
    for seed in config.SEEDS:
        split_idx = splits_mod.scaffold_split(master, seed=seed)
        for app_name in config.APPLICATIONS:
            labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)
            selected = shap_analysis.select_shap_features(
                app_name, master, full_features, split_idx, seed=seed
            )
            feature_audit.extend({
                "application": app_name,
                "seed": seed,
                "rank": rank + 1,
                "feature": feature,
            } for rank, feature in enumerate(selected))
            selected_df = full_features[selected]
            for model_name in config.MODEL_NAMES:
                _, metrics, predictions = train_and_eval(
                    model_name, selected_df, master, app_name, split_idx, seed
                )
                metadata = {
                    "application": app_name,
                    "feature_group": "shap_selected",
                    "model": model_name,
                    "split_type": "scaffold",
                    "test_subset": "test",
                    "seed": seed,
                }
                rows.append(_result_row(metadata, metrics))
                predictions_all.append(predictions.assign(**metadata))
                gc.collect()
    return pd.DataFrame(rows), predictions_all, pd.DataFrame(feature_audit)


def _run_baselines(master, precursor_features):
    rows = []
    for seed in config.SEEDS:
        split_idx = splits_mod.scaffold_split(master, seed=seed)
        for app_name in config.APPLICATIONS:
            labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)
            baseline_df = baseline_mod.evaluate_baselines(precursor_features, master, labels, split_idx)
            for _, result in baseline_df.iterrows():
                rows.append({
                    "application": app_name,
                    "feature_group": "baseline",
                    "model": result["baseline"],
                    "split_type": "scaffold",
                    "test_subset": "test",
                    "seed": seed,
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": result["f1"],
                    "roc_auc": result["roc_auc"],
                    "pr_auc": result["pr_auc"],
                    "enrichment_top10": np.nan,
                    "enrichment_top25": np.nan,
                    "train_time_sec": np.nan,
                    "inference_time_sec_per_sample": np.nan,
                    "peak_rss_delta_mb": np.nan,
                    "model_size_kb": np.nan,
                    "validation_auc": result["valid_auc"],
                    "validation_f1_at_selected_threshold": np.nan,
                    "selected_probability_threshold": result["threshold"],
                    "label_cutoff_value": cutoff,
                    "n_train": len(split_idx["train"]),
                    "n_valid": len(split_idx["valid"]),
                    "n_test": len(split_idx["test"]),
                    "train_positive_rate": labels.loc[split_idx["train"]].mean(),
                    "valid_positive_rate": labels.loc[split_idx["valid"]].mean(),
                    "test_positive_rate": labels.loc[split_idx["test"]].mean(),
                    "baseline_k": result["k"],
                    "baseline_metal_weight": result["metal_weight"],
                    "best_params": "{}",
                })
    return pd.DataFrame(rows)


def fit_deployment_model(app_name, master, feature_df, seed=config.SEED, save=True):
    """Select the deployment model on scaffold validation AUC and save it."""
    split_idx = splits_mod.scaffold_split(master, seed=seed)
    labels, cutoff = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)
    X_train, y_train = feature_df.loc[split_idx["train"]], labels.loc[split_idx["train"]].to_numpy()
    X_valid, y_valid = feature_df.loc[split_idx["valid"]], labels.loc[split_idx["valid"]].to_numpy()

    candidates = []
    for model_name in config.MODEL_NAMES:
        model, params, valid_auc, threshold, valid_f1 = tune_and_fit(
            model_name, X_train, y_train, X_valid, y_valid, seed
        )
        candidates.append((valid_auc, valid_f1, model_name, model, params, threshold))
    valid_auc, valid_f1, model_name, model, params, threshold = max(candidates, key=lambda x: (x[0], x[1]))

    metadata = {
        "application": app_name,
        "seed": seed,
        "feature_group": "precursor_descriptor",
        "model_name": model_name,
        "best_params": params,
        "validation_auc": valid_auc,
        "validation_f1": valid_f1,
        "selected_probability_threshold": threshold,
        "label_cutoff_value": cutoff,
        "feature_columns": feature_df.columns.tolist(),
        "split_indices": {
            key: [int(v) for v in np.asarray(split_idx[key], dtype=int).tolist()]
            for key in ("train", "valid", "test")
        },
    }
    bundle = {"model": model, "metadata": metadata}
    if save:
        os.makedirs(config.RESULTS_MODELS, exist_ok=True)
        joblib.dump(bundle, os.path.join(config.RESULTS_MODELS, f"deployment_{app_name}.joblib"))
        with open(os.path.join(config.RESULTS_MODELS, f"deployment_{app_name}.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    return bundle


def main():
    for directory in [config.RESULTS_TABLES, config.RESULTS_MODELS, config.RESULTS_AUDITS]:
        os.makedirs(directory, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))
    feature_groups = {
        name: pd.read_parquet(os.path.join(config.DATA_PROCESSED, f"features_{name}.parquet"))
        for name in config.BASE_FEATURE_GROUPS
    }

    scaffold_df, predictions, conflict_rows = _run_feature_group_evaluation(
        master, feature_groups, splits_mod.scaffold_split, "scaffold", config.SEEDS
    )
    shap_df, shap_predictions, shap_audit = _run_shap_selected(
        master, feature_groups["precursor_descriptor"]
    )
    predictions.extend(shap_predictions)
    baseline_df = _run_baselines(master, feature_groups["precursor_only"])

    random_df, random_predictions, random_conflicts = _run_feature_group_evaluation(
        master, feature_groups, splits_mod.random_split, "random_group_aware", [config.SEED]
    )
    predictions.extend(random_predictions)
    conflict_rows.extend(random_conflicts)

    metal_df, metal_predictions, metal_conflicts = _run_feature_group_evaluation(
        master,
        feature_groups,
        splits_mod.metal_element_holdout_split,
        "metal_element_holdout",
        [config.SEED],
        test_keys=("test", "test_partial_unseen", "test_fully_unseen"),
    )
    predictions.extend(metal_predictions)
    conflict_rows.extend(metal_conflicts)

    per_seed = pd.concat([scaffold_df, shap_df, baseline_df, random_df, metal_df], ignore_index=True, sort=False)
    per_seed.to_csv(os.path.join(config.RESULTS_TABLES, "results_per_seed.csv"), index=False)
    aggregate = _aggregate_per_seed(per_seed)
    aggregate.to_csv(os.path.join(config.RESULTS_TABLES, "ablation_results_full.csv"), index=False)

    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            os.path.join(config.RESULTS_TABLES, "predictions_per_run.csv"), index=False
        )
    shap_audit.to_csv(os.path.join(config.RESULTS_AUDITS, "shap_selected_features_per_seed.csv"), index=False)
    pd.DataFrame(conflict_rows).drop_duplicates().to_csv(
        os.path.join(config.RESULTS_AUDITS, "label_conflict_ceiling_audit.csv"), index=False
    )

    deployment_rows = []
    for app_name in config.APPLICATIONS:
        bundle = fit_deployment_model(
            app_name, master, feature_groups["precursor_descriptor"], save=True
        )
        meta = bundle["metadata"]
        deployment_rows.append({
            key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
            for key, value in meta.items()
            if key not in {"split_indices", "feature_columns"}
        })
    pd.DataFrame(deployment_rows).to_csv(
        os.path.join(config.RESULTS_TABLES, "deployment_model_summary.csv"), index=False
    )

    print(f"Saved corrected results to {config.RESULTS_TABLES}")
    return aggregate


if __name__ == "__main__":
    main()
