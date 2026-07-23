"""
metal_properties.py
--------------------
A small physicochemical property table for the metal elements
commonly found in MOF nodes. Pauling electronegativity values are
standard reference constants (not available directly from RDKit's
periodic table, so they're hardcoded here); covalent radius, atomic
weight, and atomic number are pulled programmatically from RDKit so
we don't have to hand-type those.

This is intentionally scoped to metals that actually show up in MOF
synthesis (transition metals, lanthanides, a handful of main-group /
post-transition metals used as MOF nodes) rather than the full
periodic table.
"""
from functools import lru_cache
from rdkit import Chem

# Pauling electronegativity, standard reference values.
PAULING_ELECTRONEGATIVITY = {
    "Li": 0.98, "Na": 0.93, "K": 0.82, "Rb": 0.82, "Cs": 0.79,
    "Be": 1.57, "Mg": 1.31, "Ca": 1.00, "Sr": 0.95, "Ba": 0.89,
    "Sc": 1.36, "Y": 1.22, "La": 1.10, "Ce": 1.12, "Pr": 1.13,
    "Nd": 1.14, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Tb": 1.10,
    "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10,
    "Lu": 1.27, "Th": 1.30, "U": 1.38,
    "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83,
    "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Ru": 2.20, "Rh": 2.28,
    "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "Hf": 1.30, "Ta": 1.50,
    "W": 2.36, "Re": 1.90, "Ir": 2.20, "Pt": 2.28, "Au": 2.54,
    "Hg": 2.00, "Al": 1.61, "Ga": 1.81, "In": 1.78, "Sn": 1.96,
    "Sb": 2.05, "Bi": 2.02, "Pb": 2.33, "Ge": 2.01,
}

KNOWN_METALS = set(PAULING_ELECTRONEGATIVITY.keys())

_pt = Chem.GetPeriodicTable()


@lru_cache(maxsize=None)
def get_metal_atoms(metal_frag_smiles: str):
    """
    Parses a metal-fragment SMILES (e.g. "Cl[Mn][Mn]Cl", "[Zn][O]([Zn])([Zn])[Zn]")
    and returns the list of atom symbols that are recognized MOF metals
    (bridging ligand atoms like O/Cl/OH are excluded).
    """
    if not metal_frag_smiles:
        return []
    mol = Chem.MolFromSmiles(metal_frag_smiles, sanitize=False)
    if mol is None:
        return []
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return [s for s in symbols if s in KNOWN_METALS]


def metal_fragment_features(metal_frag_smiles: str) -> dict:
    """
    Returns a fixed-size dict of numeric features describing the metal
    node, averaged across all recognized metal atoms present (MOF
    clusters can be multi-metal, e.g. Zn4O nodes).
    """
    metals = get_metal_atoms(metal_frag_smiles)

    if not metals:
        return {
            "primary_metal": "UNK",
            "num_metal_atoms": 0,
            "num_distinct_metals": 0,
            "metal_electronegativity_avg": 0.0,
            "metal_covalent_radius_avg": 0.0,
            "metal_atomic_weight_avg": 0.0,
            "metal_atomic_number_avg": 0.0,
        }

    en = [PAULING_ELECTRONEGATIVITY[m] for m in metals]
    rcov = [_pt.GetRcovalent(m) for m in metals]
    weight = [_pt.GetAtomicWeight(m) for m in metals]
    num = [_pt.GetAtomicNumber(m) for m in metals]

    # most frequent metal element -> used for one-hot encoding elsewhere
    primary_metal = max(set(metals), key=metals.count)

    return {
        "primary_metal": primary_metal,
        "num_metal_atoms": len(metals),
        "num_distinct_metals": len(set(metals)),
        "metal_electronegativity_avg": sum(en) / len(en),
        "metal_covalent_radius_avg": sum(rcov) / len(rcov),
        "metal_atomic_weight_avg": sum(weight) / len(weight),
        "metal_atomic_number_avg": sum(num) / len(num),
    }
