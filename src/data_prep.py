"""Data preparation, precursor parsing, integrity checks, and audit outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import config
import metal_properties as mp


def _sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_single(prop_key: str) -> pd.DataFrame:
    path = os.path.join(config.DATA_RAW, config.RAW_FILES[prop_key])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required raw file not found: {path}")
    return pd.read_csv(path, header=None, names=["precursor", prop_key])


def load_master_table() -> pd.DataFrame:
    """Merge the six row-aligned XRayPro property files by row position."""
    dfs = [_load_single(k) for k in config.RAW_FILES]
    lengths = {len(df) for df in dfs}
    if len(lengths) != 1:
        raise ValueError(f"Raw files have unequal row counts: {sorted(lengths)}")

    base = dfs[0]["precursor"].astype(str).to_numpy()
    for key, df in zip(config.RAW_FILES, dfs):
        cur = df["precursor"].astype(str).to_numpy()
        if not np.array_equal(cur, base):
            bad = np.flatnonzero(cur != base)[:10].tolist()
            raise ValueError(
                f"Row alignment failed for {key}; first mismatching rows: {bad}. "
                "Do not continue with a positional merge."
            )

    master = pd.concat(
        [dfs[0][["precursor"]]]
        + [df[[key]] for key, df in zip(config.RAW_FILES, dfs)],
        axis=1,
    )
    master.insert(0, "source_row_id", np.arange(len(master), dtype=int))
    master["mof_id"] = master["source_row_id"]
    return master


def _parse_fragment(fragment: str):
    try:
        return Chem.MolFromSmiles(fragment, sanitize=False)
    except Exception:
        return None


def _canonical_fragment(fragment: str) -> str:
    mol = _parse_fragment(fragment)
    if mol is None:
        return str(fragment)
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return str(fragment)


def split_precursor(precursor: str) -> tuple[str, str, str, int, int]:
    """Split a dot-separated precursor into metal, organic-linker, and auxiliary parts.

    V3 rule: any fragment containing a recognized MOF metal is classified as a
    metal-node fragment even when it also contains carbon. This prevents
    organometallic node fragments from being silently treated as linkers.
    Carbon-containing, metal-free fragments are linker candidates. Remaining
    fragments are retained in ``auxiliary_frag`` rather than discarded.
    """
    fragments = [f.strip() for f in str(precursor).split(".") if f.strip()]
    metal_parts: list[str] = []
    linker_parts: list[str] = []
    auxiliary_parts: list[str] = []
    ambiguous = 0

    for frag in fragments:
        mol = _parse_fragment(frag)
        if mol is None:
            auxiliary_parts.append(frag)
            ambiguous += 1
            continue
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        contains_metal = any(symbol in mp.KNOWN_METALS for symbol in symbols)
        contains_carbon = "C" in symbols
        if contains_metal:
            metal_parts.append(frag)
        elif contains_carbon:
            linker_parts.append(frag)
        else:
            auxiliary_parts.append(frag)

    metal = ".".join(metal_parts)
    linker = ".".join(linker_parts)
    auxiliary = ".".join(auxiliary_parts)
    return metal, linker, auxiliary, len(linker_parts), ambiguous


def add_precursor_parsing(master: pd.DataFrame) -> pd.DataFrame:
    parsed = master["precursor"].apply(split_precursor)
    out = master.copy()
    out[[
        "metal_frag",
        "linker_smiles",
        "auxiliary_frag",
        "n_linker_fragments",
        "n_ambiguous_fragments",
    ]] = pd.DataFrame(parsed.tolist(), index=out.index)

    out["canonical_linker"] = out["linker_smiles"].apply(_canonical_fragment)
    out["canonical_precursor"] = out["precursor"].apply(
        lambda x: ".".join(sorted(_canonical_fragment(f) for f in str(x).split(".") if f))
    )
    return out


def flag_unparseable_linkers(master: pd.DataFrame) -> pd.DataFrame:
    def _valid(smiles: str) -> bool:
        if pd.isna(smiles) or str(smiles).strip() == "":
            return False
        try:
            return Chem.MolFromSmiles(str(smiles)) is not None
        except Exception:
            return False

    out = master.copy()
    out["linker_parse_failed"] = ~out["linker_smiles"].apply(_valid)
    return out


def precursor_degeneracy_property_report(master: pd.DataFrame) -> pd.DataFrame:
    """Report raw property variation without calling it a classification ceiling."""
    rows = []
    for precursor, group in master.groupby("canonical_precursor", dropna=False):
        if len(group) < 2:
            continue
        rows.append({
            "canonical_precursor": precursor,
            "n_rows": len(group),
            "co2_min": group["co2_uptake_lp"].min(),
            "co2_max": group["co2_uptake_lp"].max(),
            "co2_range": group["co2_uptake_lp"].max() - group["co2_uptake_lp"].min(),
            "ch4_min": group["ch4_uptake_hp"].min(),
            "ch4_max": group["ch4_uptake_hp"].max(),
            "ch4_range": group["ch4_uptake_hp"].max() - group["ch4_uptake_hp"].min(),
        })
    return pd.DataFrame(rows)


def build_parsing_audit(master: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "n_rows_before_exclusion": int(len(master)),
        "n_unique_raw_precursors": int(master["precursor"].nunique(dropna=False)),
        "n_unique_canonical_precursors": int(master["canonical_precursor"].nunique(dropna=False)),
        "n_linker_parse_failed": int(master["linker_parse_failed"].sum()),
        "n_rows_with_multiple_linker_fragments": int((master["n_linker_fragments"] > 1).sum()),
        "n_rows_with_auxiliary_fragments": int(master["auxiliary_frag"].fillna("").ne("").sum()),
        "n_rows_with_ambiguous_fragments": int((master["n_ambiguous_fragments"] > 0).sum()),
        "n_rows_without_recognized_metal": int(master["metal_frag"].fillna("").eq("").sum()),
    }
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])


def assert_no_leakage(feature_columns: Iterable[str], application_name: str) -> None:
    forbidden = set(config.RAW_FILES) | {
        f"label_{app}" for app in config.APPLICATIONS
    } | {
        "precursor", "canonical_precursor", "metal_frag", "linker_smiles",
        "canonical_linker", "auxiliary_frag", "mof_id", "source_row_id",
        "linker_parse_failed",
    }
    leaked = forbidden.intersection(set(feature_columns))
    if leaked:
        raise ValueError(
            f"Data leakage detected for {application_name}: forbidden columns {sorted(leaked)}"
        )


def write_raw_manifest() -> None:
    rows = []
    for key, fname in config.RAW_FILES.items():
        path = os.path.join(config.DATA_RAW, fname)
        rows.append({
            "property_key": key,
            "filename": fname,
            "bytes": os.path.getsize(path),
            "sha256": _sha256(path),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(config.RESULTS_AUDITS, "raw_data_manifest.csv"), index=False
    )


def main() -> pd.DataFrame:
    for directory in [config.DATA_PROCESSED, config.RESULTS_AUDITS]:
        os.makedirs(directory, exist_ok=True)

    raw = load_master_table()
    parsed = flag_unparseable_linkers(add_precursor_parsing(raw))

    build_parsing_audit(parsed).to_csv(
        os.path.join(config.RESULTS_AUDITS, "precursor_parsing_audit.csv"), index=False
    )
    precursor_degeneracy_property_report(parsed).to_csv(
        os.path.join(config.DATA_PROCESSED, "precursor_degeneracy_property_report.csv"),
        index=False,
    )

    if config.EXCLUDE_UNPARSEABLE_LINKERS:
        master = parsed.loc[~parsed["linker_parse_failed"]].copy()
    else:
        master = parsed.copy()
    master = master.reset_index(drop=True)
    master["mof_id"] = np.arange(len(master), dtype=int)

    out = os.path.join(config.DATA_PROCESSED, "master_table.csv")
    master.to_csv(out, index=False)
    write_raw_manifest()

    summary = {
        "pipeline_version": config.PIPELINE_VERSION,
        "raw_rows": int(len(raw)),
        "modeling_rows": int(len(master)),
        "excluded_unparseable": int(parsed["linker_parse_failed"].sum()) if config.EXCLUDE_UNPARSEABLE_LINKERS else 0,
        "unique_canonical_precursors": int(master["canonical_precursor"].nunique()),
        "master_table_sha256": _sha256(out),
    }
    with open(os.path.join(config.RESULTS_AUDITS, "data_prep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return master


if __name__ == "__main__":
    main()
