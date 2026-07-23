"""Deterministic precursor-derived feature construction.

V3 removes the data-fitted top-15 metal vocabulary. Every recognized metal
has a fixed indicator column, so the feature schema never depends on test-set
frequency information.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

import config
import data_prep
import metal_properties as mp

RDKIT_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "MolLogP": Descriptors.MolLogP,
    "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumCarboxylGroups": lambda m: len(
        m.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1]"))
    ),
    "NumPyridylN": lambda m: len(m.GetSubstructMatches(Chem.MolFromSmarts("n"))),
}


def _safe_mol(smiles: str):
    if not smiles or pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def linker_descriptor_row(smiles: str) -> dict[str, float]:
    mol = _safe_mol(smiles)
    if mol is None:
        raise ValueError(f"Unparseable linker reached featurization: {smiles!r}")
    row: dict[str, float] = {}
    for name, func in RDKIT_DESCRIPTOR_FUNCS.items():
        try:
            row[name] = float(func(mol))
        except Exception as exc:
            raise RuntimeError(f"RDKit descriptor {name} failed for {smiles!r}") from exc
    return row


def linker_fingerprint(smiles: str) -> np.ndarray:
    mol = _safe_mol(smiles)
    if mol is None:
        raise ValueError(f"Unparseable linker reached fingerprinting: {smiles!r}")
    generator = AllChem.GetMorganGenerator(radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_NBITS)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((config.MORGAN_NBITS,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _metal_indicator_row(fragment: str) -> dict[str, int]:
    present = set(mp.get_metal_atoms(fragment))
    return {f"metal_{symbol}": int(symbol in present) for symbol in sorted(mp.KNOWN_METALS)}


def build_features(master: pd.DataFrame):
    if master["linker_parse_failed"].any():
        raise ValueError("Modeling table still contains unparseable linkers.")

    fp_matrix = np.stack([linker_fingerprint(s) for s in master["linker_smiles"]])
    fp_df = pd.DataFrame(
        fp_matrix,
        columns=[f"fp_{i}" for i in range(config.MORGAN_NBITS)],
        index=master.index,
    )

    metal_numeric = pd.DataFrame(
        [mp.metal_fragment_features(m) for m in master["metal_frag"]], index=master.index
    )
    metal_indicators = pd.DataFrame(
        [_metal_indicator_row(m) for m in master["metal_frag"]], index=master.index
    )

    precursor_only = pd.concat(
        [
            fp_df,
            metal_indicators,
            metal_numeric[["num_metal_atoms", "num_distinct_metals"]],
            master[["n_linker_fragments"]].astype(float),
        ],
        axis=1,
    )

    descriptor_df = pd.DataFrame(
        [linker_descriptor_row(s) for s in master["linker_smiles"]], index=master.index
    )
    descriptor_only = pd.concat(
        [
            descriptor_df,
            metal_numeric[[
                "metal_electronegativity_avg",
                "metal_covalent_radius_avg",
                "metal_atomic_weight_avg",
                "metal_atomic_number_avg",
            ]],
        ],
        axis=1,
    )

    precursor_descriptor = pd.concat([precursor_only, descriptor_only], axis=1)
    for frame in [precursor_only, descriptor_only, precursor_descriptor]:
        if frame.isna().any().any():
            raise ValueError("NaN detected in generated features.")
        if not frame.index.equals(master.index):
            raise ValueError("Feature/master row alignment failed.")

    return precursor_only, descriptor_only, precursor_descriptor


def main():
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)
    os.makedirs(config.RESULTS_AUDITS, exist_ok=True)
    master = pd.read_csv(os.path.join(config.DATA_PROCESSED, "master_table.csv"))
    feature_groups = dict(zip(
        config.BASE_FEATURE_GROUPS,
        build_features(master),
    ))

    audit_rows = []
    for name, frame in feature_groups.items():
        for app in config.APPLICATIONS:
            data_prep.assert_no_leakage(frame.columns, app)
        path = os.path.join(config.DATA_PROCESSED, f"features_{name}.parquet")
        frame.to_parquet(path)
        audit_rows.append({
            "feature_group": name,
            "n_rows": len(frame),
            "n_features": frame.shape[1],
            "n_missing": int(frame.isna().sum().sum()),
            "all_numeric": bool(all(np.issubdtype(dtype, np.number) for dtype in frame.dtypes)),
        })
        print(f"{name}: {frame.shape}")

    pd.DataFrame(audit_rows).to_csv(
        os.path.join(config.RESULTS_AUDITS, "feature_schema_audit.csv"), index=False
    )
    with open(os.path.join(config.RESULTS_AUDITS, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump({name: list(frame.columns) for name, frame in feature_groups.items()}, f, indent=2)
    return feature_groups


if __name__ == "__main__":
    main()
