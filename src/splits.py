"""Leakage-audited data splits and split-level diagnostics."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

RDLogger.DisableLog("rdApp.*")

import config
import metal_properties as mp


def make_labels_from_train_threshold(master: pd.DataFrame, train_idx, application_name: str):
    spec = config.APPLICATIONS[application_name]
    prop = spec["source_property"]
    train_values = master.loc[train_idx, prop].dropna().to_numpy()
    if len(train_values) == 0:
        raise ValueError(f"No training values available for {application_name}")
    cutoff = float(np.percentile(train_values, spec["percentile_cutoff"]))
    labels = (master[prop] >= cutoff).astype(int) if spec["higher_is_better"] else (master[prop] <= cutoff).astype(int)
    return labels, cutoff


@lru_cache(maxsize=None)
def _murcko_group_key(smiles: str) -> str:
    """Return a non-empty scaffold key.

    Acyclic molecules have an empty Bemis-Murcko scaffold. Treating all such
    molecules as one group creates a giant unstable split. V3 therefore falls
    back to the canonical full linker for acyclic molecules.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return f"INVALID::{smiles}"
        canonical = Chem.MolToSmiles(mol, canonical=True)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return f"ACYCLIC::{canonical}"
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
        return f"SCAFFOLD::{scaffold_smiles}" if scaffold_smiles else f"ACYCLIC::{canonical}"
    except Exception:
        return f"INVALID::{smiles}"


def scaffold_group_keys(master: pd.DataFrame) -> pd.Series:
    return master["canonical_linker"].fillna(master["linker_smiles"]).apply(_murcko_group_key)


def _balanced_group_split(indices, groups, seed: int, ratios=(0.70, 0.15, 0.15)):
    """Greedily assign whole groups to approximately balanced partitions."""
    indices = np.asarray(indices)
    groups = pd.Series(np.asarray(groups), index=indices)
    grouped = groups.groupby(groups).apply(lambda s: s.index.to_numpy())

    rng = np.random.RandomState(seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_names = ["train", "valid", "test"]
    n_total = len(indices)
    targets = dict(zip(split_names, np.asarray(ratios) * n_total))
    assigned = {name: [] for name in split_names}
    counts = {name: 0 for name in split_names}

    for _, member_idx in group_items:
        size = len(member_idx)
        # Prefer the split with the largest relative remaining capacity.
        def score(name):
            remaining = targets[name] - counts[name]
            overflow = max(0.0, counts[name] + size - targets[name])
            return (remaining / max(targets[name], 1.0)) - 4.0 * (overflow / max(targets[name], 1.0))

        chosen = max(split_names, key=score)
        assigned[chosen].extend(member_idx.tolist())
        counts[chosen] += size

    # Shuffle rows within each split without changing group assignment.
    out = {}
    for i, name in enumerate(split_names):
        arr = np.asarray(assigned[name], dtype=int)
        np.random.RandomState(seed + 1009 * (i + 1)).shuffle(arr)
        out[name] = arr

    if any(len(out[name]) == 0 for name in split_names):
        raise ValueError(f"Empty partition produced: { {k: len(v) for k,v in out.items()} }")
    return out


def random_split(master: pd.DataFrame, seed: int | None = None):
    seed = config.SEED if seed is None else seed
    idx = master.index.to_numpy()
    groups = master["canonical_precursor"].astype(str).to_numpy()
    return _balanced_group_split(idx, groups, seed)


def scaffold_split(master: pd.DataFrame, seed: int | None = None):
    seed = config.SEED if seed is None else seed
    idx = master.index.to_numpy()
    groups = scaffold_group_keys(master).to_numpy()
    split = _balanced_group_split(idx, groups, seed)
    split["scaffold_group_key"] = groups
    return split


def metal_element_holdout_split(
    master: pd.DataFrame,
    seed: int | None = None,
    holdout_frac: float | None = None,
):
    seed = config.SEED if seed is None else seed
    holdout_frac = config.METAL_ELEMENT_HOLDOUT_FRAC if holdout_frac is None else holdout_frac
    rng = np.random.RandomState(seed)

    element_sets = master["metal_frag"].apply(lambda frag: set(mp.get_metal_atoms(frag)))
    all_elements = sorted(set().union(*element_sets)) if len(element_sets) else []
    if len(all_elements) < 2:
        raise ValueError("Too few recognized metal elements for a holdout split.")

    n_holdout = max(1, int(round(len(all_elements) * holdout_frac)))
    n_holdout = min(n_holdout, len(all_elements) - 1)
    holdout_elements = set(rng.choice(all_elements, size=n_holdout, replace=False).tolist())

    has_any = element_sets.apply(lambda s: bool(s & holdout_elements))
    fully_unseen = element_sets.apply(lambda s: bool(s) and s.issubset(holdout_elements))
    partial_unseen = has_any & ~fully_unseen

    train_pool = master.index[~has_any].to_numpy()
    test_all = master.index[has_any].to_numpy()
    test_partial = master.index[partial_unseen].to_numpy()
    test_fully = master.index[fully_unseen].to_numpy()

    # Group-aware train/validation split prevents duplicate-precursor tuning leakage.
    pool_groups = master.loc[train_pool, "canonical_precursor"].astype(str).to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_rel, valid_rel = next(gss.split(train_pool, groups=pool_groups))
    train_idx, valid_idx = train_pool[train_rel], train_pool[valid_rel]

    train_elements = set().union(*element_sets.loc[train_idx]) if len(train_idx) else set()
    violation = train_elements & holdout_elements
    if violation:
        raise ValueError(f"Metal holdout leakage: {sorted(violation)}")

    train_prec = set(master.loc[train_idx, "canonical_precursor"])
    valid_prec = set(master.loc[valid_idx, "canonical_precursor"])
    if train_prec & valid_prec:
        raise ValueError("Exact precursor leakage between metal-holdout train and validation sets.")

    return {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_all,
        "test_partial_unseen": test_partial,
        "test_fully_unseen": test_fully,
        "holdout_elements": sorted(holdout_elements),
    }


def _pairwise_overlap(master, a_idx, b_idx, column):
    return len(set(master.loc[a_idx, column].astype(str)) & set(master.loc[b_idx, column].astype(str)))


def split_audit(master: pd.DataFrame, split_idx: dict, split_type: str, seed: int) -> dict:
    row = {
        "split_type": split_type,
        "seed": seed,
        "n_train": len(split_idx["train"]),
        "n_valid": len(split_idx["valid"]),
        "n_test": len(split_idx["test"]),
        "train_fraction": len(split_idx["train"]) / len(master),
        "valid_fraction": len(split_idx["valid"]) / len(master),
        "test_fraction": len(split_idx["test"]) / len(master),
        "precursor_overlap_train_valid": _pairwise_overlap(master, split_idx["train"], split_idx["valid"], "canonical_precursor"),
        "precursor_overlap_train_test": _pairwise_overlap(master, split_idx["train"], split_idx["test"], "canonical_precursor"),
        "linker_overlap_train_test": _pairwise_overlap(master, split_idx["train"], split_idx["test"], "canonical_linker"),
    }
    if split_type == "scaffold":
        keys = scaffold_group_keys(master)
        row["scaffold_overlap_train_test"] = len(
            set(keys.loc[split_idx["train"]]) & set(keys.loc[split_idx["test"]])
        )
    if split_type == "metal_element_holdout":
        row["n_test_partial_unseen"] = len(split_idx.get("test_partial_unseen", []))
        row["n_test_fully_unseen"] = len(split_idx.get("test_fully_unseen", []))
        row["holdout_elements"] = ";".join(split_idx.get("holdout_elements", []))
    return row


def label_conflict_audit(master: pd.DataFrame, labels: pd.Series, split_idx: dict, application: str, cutoff: float, split_type: str, seed: int):
    rows = []
    for subset in ["train", "valid", "test"]:
        idx = split_idx[subset]
        frame = pd.DataFrame({
            "group": master.loc[idx, "canonical_precursor"].astype(str).to_numpy(),
            "label": labels.loc[idx].to_numpy(),
        })
        grouped = frame.groupby("group")["label"]
        sizes = grouped.size()
        conflicted = grouped.nunique() > 1
        majority_correct = grouped.value_counts().groupby(level=0).max().sum()
        rows.append({
            "application": application,
            "split_type": split_type,
            "seed": seed,
            "subset": subset,
            "cutoff": cutoff,
            "n_rows": len(frame),
            "positive_rate": frame["label"].mean(),
            "n_duplicate_groups": int((sizes > 1).sum()),
            "n_conflicting_label_groups": int(conflicted.sum()),
            "n_rows_in_conflicting_groups": int(sizes.loc[conflicted].sum()) if conflicted.any() else 0,
            "majority_label_accuracy_ceiling": float(majority_correct / len(frame)) if len(frame) else np.nan,
        })
    return rows


def save_split_indices(split_idx: dict, split_type: str, seed: int):
    os.makedirs(config.RESULTS_AUDITS, exist_ok=True)
    serializable = {}
    for key, value in split_idx.items():
        if key == "scaffold_group_key":
            continue
        if isinstance(value, np.ndarray):
            serializable[key] = [int(x) for x in value]
        else:
            serializable[key] = value
    path = os.path.join(config.RESULTS_AUDITS, f"split_indices_{split_type}_seed{seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def main():
    os.makedirs(config.RESULTS_AUDITS, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))
    audits = []
    for seed in config.SEEDS:
        for name, fn in [
            ("random_group_aware", random_split),
            ("scaffold", scaffold_split),
            ("metal_element_holdout", metal_element_holdout_split),
        ]:
            split = fn(master, seed=seed)
            audit = split_audit(master, split, name, seed)
            audits.append(audit)
            save_split_indices(split, name, seed)
            print(audit)

    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(os.path.join(config.RESULTS_AUDITS, "split_audit.csv"), index=False)

    scaffold_rows = audit_df[audit_df["split_type"] == "scaffold"]
    if (scaffold_rows["precursor_overlap_train_test"] > 0).any():
        raise AssertionError("Scaffold split has precursor leakage.")
    if (scaffold_rows["linker_overlap_train_test"] > 0).any():
        raise AssertionError("Scaffold split has linker leakage.")
    if (scaffold_rows["scaffold_overlap_train_test"] > 0).any():
        raise AssertionError("Scaffold split has scaffold leakage.")
    return audit_df


if __name__ == "__main__":
    main()
