"""Validation-tuned class-prior and chemical-similarity baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import DataStructs
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import config
import metal_properties as mp


def _to_bitvect(row: np.ndarray):
    bv = DataStructs.ExplicitBitVect(len(row))
    bv.SetBitsFromList(np.flatnonzero(row).astype(int).tolist())
    return bv


def _linker_similarity_matrix(train_fp: pd.DataFrame, query_fp: pd.DataFrame) -> np.ndarray:
    """Compute query-by-train Tanimoto similarity once per evaluation fold."""
    train_bv = [_to_bitvect(row) for row in train_fp.to_numpy()]
    out = np.empty((len(query_fp), len(train_fp)), dtype=np.float32)
    for i, row in enumerate(query_fp.to_numpy()):
        out[i] = DataStructs.BulkTanimotoSimilarity(_to_bitvect(row), train_bv)
    return out


def _metal_binary_matrix(fragments: pd.Series) -> np.ndarray:
    symbols = sorted(mp.KNOWN_METALS)
    col = {symbol: i for i, symbol in enumerate(symbols)}
    matrix = np.zeros((len(fragments), len(symbols)), dtype=np.uint8)
    for r, fragment in enumerate(fragments):
        for symbol in set(mp.get_metal_atoms(fragment)):
            if symbol in col:
                matrix[r, col[symbol]] = 1
    return matrix


def _metal_jaccard_matrix(train_fragments: pd.Series, query_fragments: pd.Series) -> np.ndarray:
    train = _metal_binary_matrix(train_fragments)
    query = _metal_binary_matrix(query_fragments)
    intersection = query.astype(np.float32) @ train.astype(np.float32).T
    union = query.sum(axis=1, keepdims=True) + train.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _knn_probabilities(similarity: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), similarity.shape[1])
    indices = np.argpartition(similarity, -k, axis=1)[:, -k:]
    weights = np.take_along_axis(similarity, indices, axis=1)
    labels = y_train[indices]
    denominator = weights.sum(axis=1)
    numerator = (weights * labels).sum(axis=1)
    prior = float(y_train.mean())
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, prior, dtype=float),
        where=denominator > 0,
    )


def _safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def _optimal_threshold(y_true, probs):
    best_threshold, best_score = 0.5, -1.0
    for threshold in config.THRESHOLD_GRID:
        score = f1_score(y_true, probs >= threshold, zero_division=0)
        if score > best_score + 1e-12:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def evaluate_baselines(fp_df: pd.DataFrame, master: pd.DataFrame, labels: pd.Series, split_idx: dict):
    fp_cols = [column for column in fp_df.columns if column.startswith("fp_")]
    train, valid, test = split_idx["train"], split_idx["valid"], split_idx["test"]
    X_train = fp_df.loc[train, fp_cols]
    X_valid = fp_df.loc[valid, fp_cols]
    X_test = fp_df.loc[test, fp_cols]
    y_train = labels.loc[train].to_numpy(dtype=float)
    y_valid = labels.loc[valid].to_numpy(dtype=int)
    y_test = labels.loc[test].to_numpy(dtype=int)

    rows = []
    prior = float(y_train.mean())
    valid_prior = np.full(len(y_valid), prior)
    test_prior = np.full(len(y_test), prior)
    prior_threshold = _optimal_threshold(y_valid, valid_prior)
    rows.append({
        "baseline": "class_prior",
        "k": np.nan,
        "metal_weight": np.nan,
        "valid_auc": _safe_auc(y_valid, valid_prior),
        "threshold": prior_threshold,
        "roc_auc": _safe_auc(y_test, test_prior),
        "pr_auc": average_precision_score(y_test, test_prior),
        "f1": f1_score(y_test, test_prior >= prior_threshold, zero_division=0),
    })

    valid_linker = _linker_similarity_matrix(X_train, X_valid)
    valid_metal = _metal_jaccard_matrix(
        master.loc[train, "metal_frag"], master.loc[valid, "metal_frag"]
    )

    best_by_name = {}
    for baseline_name, weights in [
        ("linker_tanimoto_knn", [0.0]),
        ("metal_aware_precursor_knn", config.METAL_SIMILARITY_WEIGHT_GRID),
    ]:
        best = None
        for metal_weight in weights:
            combined = (
                valid_linker
                if metal_weight == 0
                else (valid_linker + metal_weight * valid_metal) / (1.0 + metal_weight)
            )
            for k in config.KNN_K_GRID:
                valid_prob = _knn_probabilities(combined, y_train, k)
                valid_auc = _safe_auc(y_valid, valid_prob)
                candidate = (np.nan_to_num(valid_auc, nan=-1.0), -k, -metal_weight)
                if best is None or candidate > best[0]:
                    best = (candidate, k, metal_weight, valid_prob, valid_auc)
        best_by_name[baseline_name] = best

    test_linker = _linker_similarity_matrix(X_train, X_test)
    test_metal = _metal_jaccard_matrix(
        master.loc[train, "metal_frag"], master.loc[test, "metal_frag"]
    )
    for baseline_name, best in best_by_name.items():
        _, k, metal_weight, valid_prob, valid_auc = best
        valid_threshold = _optimal_threshold(y_valid, valid_prob)
        test_combined = (
            test_linker
            if metal_weight == 0
            else (test_linker + metal_weight * test_metal) / (1.0 + metal_weight)
        )
        test_prob = _knn_probabilities(test_combined, y_train, k)
        rows.append({
            "baseline": baseline_name,
            "k": int(k),
            "metal_weight": float(metal_weight),
            "valid_auc": valid_auc,
            "threshold": valid_threshold,
            "roc_auc": _safe_auc(y_test, test_prob),
            "pr_auc": average_precision_score(y_test, test_prob),
            "f1": f1_score(y_test, test_prob >= valid_threshold, zero_division=0),
        })
    return pd.DataFrame(rows)
