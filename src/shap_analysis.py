"""Leakage-safe SHAP feature selection and post-hoc interpretation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

import config
import data_prep
import splits as splits_mod

FEATURE_NOTES = {
    "MolWt": "associated with linker/framework size",
    "TPSA": "polar surface area; associated with framework polarity",
    "NumHDonors": "H-bond donor count",
    "NumHAcceptors": "H-bond acceptor count",
    "NumRotatableBonds": "linker flexibility indicator",
    "NumAromaticRings": "aromatic-ring count; associated with rigidity",
    "RingCount": "overall linker ring content",
    "FractionCSP3": "sp3 fraction; flexibility/rigidity indicator",
    "MolLogP": "hydrophobicity indicator",
    "NumHeteroatoms": "heteroatom content",
    "HeavyAtomCount": "linker size proxy",
    "NumCarboxylGroups": "carboxylate coordination-group count",
    "NumPyridylN": "pyridyl-type nitrogen count",
    "metal_electronegativity_avg": "average metal electronegativity",
    "metal_covalent_radius_avg": "average metal covalent radius",
    "metal_atomic_weight_avg": "average metal atomic weight",
    "metal_atomic_number_avg": "average metal atomic number",
    "num_metal_atoms": "metal-node nuclearity",
    "num_distinct_metals": "mixed-metal node indicator",
    "n_linker_fragments": "number of parsed organic precursor fragments",
}


def decode_fingerprint_bit(bit_idx: int, sample_smiles_list, radius=None, nbits=None):
    radius = config.MORGAN_RADIUS if radius is None else radius
    nbits = config.MORGAN_NBITS if nbits is None else nbits
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    for smiles in sample_smiles_list:
        mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
        if mol is None:
            continue
        additional = AllChem.AdditionalOutput()
        additional.AllocateBitInfoMap()
        generator.GetFingerprint(mol, additionalOutput=additional)
        info = additional.GetBitInfoMap()
        if bit_idx not in info:
            continue
        atom_idx, rad = info[bit_idx][0]
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
        try:
            submol = Chem.PathToSubmol(mol, env)
            frag = Chem.MolToSmiles(submol)
            if frag:
                return f"substructure: {frag}"
        except Exception:
            pass
    return "fingerprint bit; no decoded example found"


def select_shap_features(
    app_name: str,
    master: pd.DataFrame,
    feat_df: pd.DataFrame,
    split_idx: dict,
    seed: int,
    top_k: int | None = None,
):
    """Select features separately for each seed using that seed's train fold."""
    top_k = config.SHAP_TOP_K if top_k is None else top_k
    labels, _ = splits_mod.make_labels_from_train_threshold(master, split_idx["train"], app_name)
    data_prep.assert_no_leakage(feat_df.columns, app_name)

    X_train = feat_df.loc[split_idx["train"]]
    y_train = labels.loc[split_idx["train"]]
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.10,
        random_state=seed,
        eval_metric="logloss",
        n_jobs=config.N_JOBS,
    )
    model.fit(X_train, y_train)

    if len(X_train) > config.SHAP_EXPLAIN_MAX:
        X_explain = X_train.sample(config.SHAP_EXPLAIN_MAX, random_state=seed)
    else:
        X_explain = X_train
    # XGBoost's native pred_contribs are SHAP values and are much faster
    # than constructing a generic TreeExplainer for this feature-selection pass.
    dmatrix = xgb.DMatrix(X_explain, feature_names=X_explain.columns.tolist())
    values = model.get_booster().predict(
        dmatrix, pred_contribs=True, approx_contribs=True
    )[:, :-1]  # last column is the bias term
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    return X_train.columns[order].tolist()


def _normalize_binary_shap(values, n_samples, n_features):
    """Return a 2-D (samples, features) matrix for the positive class."""
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            return arr[:, :, 1 if arr.shape[2] > 1 else 0]
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            return arr[1 if arr.shape[0] > 1 else 0, :, :]
    raise ValueError(f"Unsupported SHAP shape {arr.shape}; expected a binary-class explanation matrix.")


def build_trend_table_on_test(model, X_test: pd.DataFrame, sample_linkers):
    X_explain = X_test.sample(
        min(len(X_test), config.SHAP_EXPLAIN_MAX), random_state=config.SEED
    )
    if hasattr(model, "named_steps"):
        scaler = model.named_steps["standardscaler"]
        estimator = model.named_steps["logisticregression"]
        scaled = scaler.transform(X_explain)
        raw_values = shap.LinearExplainer(estimator, scaled).shap_values(scaled)
    else:
        raw_values = shap.TreeExplainer(model)(X_explain).values
    values = _normalize_binary_shap(raw_values, len(X_explain), X_explain.shape[1])
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:config.SHAP_TOP_K]

    rows = []
    for i in order:
        feature = X_explain.columns[i]
        xvals = X_explain.iloc[:, i].to_numpy()
        svals = values[:, i]
        if np.std(xvals) == 0 or np.std(svals) == 0:
            trend = "flat"
        else:
            corr = np.corrcoef(xvals, svals)[0, 1]
            trend = "nonlinear" if abs(corr) < 0.15 else ("positive" if corr > 0 else "negative")
        if feature.startswith("fp_"):
            note = decode_fingerprint_bit(int(feature.split("_")[1]), sample_linkers)
        elif feature.startswith("metal_") and feature not in FEATURE_NOTES:
            note = "metal identity indicator"
        else:
            note = FEATURE_NOTES.get(feature, "n/a")
        rows.append({
            "feature": feature,
            "mean_abs_shap": float(mean_abs[i]),
            "shap_trend": trend,
            "materials_meaning": note,
        })
    return pd.DataFrame(rows)
