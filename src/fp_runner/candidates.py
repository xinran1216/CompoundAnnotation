from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from .utils import ppm_window
import ast

def compute_morgan_fp_bits(smiles: str, radius=2, n_bits=2048):
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def find_db5_columns(df: pd.DataFrame):
    mass_col = None
    for c in ["monoisotopic_mass", "MonoisotopicMass", "MONOISOTOPIC_MASS", "ExactMass", "EXACT_MASS", "AverageMass"]:
        if c in df.columns:
            mass_col = c
            break
    fingerprint_col = None
    for c in ["fingerprint", "Fingerprint", "fp_bits", "fp"]:
        if c in df.columns:
            fingerprint_col = c
            break
    smiles_col = None
    for c in ["CANONICAL_SMILES", "canonical_smiles", "SMILES"]:
        if c in df.columns:
            smiles_col = c
            break
    inchikey_col = None
    for c in ["InChIKey", "InChIkey", "inchikey"]:
        if c in df.columns:
            inchikey_col = c
            break
    name_col = None
    for c in ["CompoundName", "COMPOUND_NAME", "name", "Name"]:
        if c in df.columns:
            name_col = c
            break
    formula_col = None
    for c in ["formula", "Formula", "molecularFormula", "MOLECULAR_FORMULA"]:
        if c in df.columns:
            formula_col = c
            break
    return mass_col, fingerprint_col, smiles_col, inchikey_col, name_col, formula_col

def _decode_fingerprint(value) -> Optional[List[int]]:
    """Decode a stored fingerprint (sparse list of 1 positions) into dense bits."""
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            indices = [int(v) for v in value]
        else:
            parsed = ast.literal_eval(str(value))
            if not isinstance(parsed, (list, tuple)):
                return None
            indices = [int(v) for v in parsed]
    except Exception:
        return None

    max_idx = max(indices) if indices else -1
    dense = [0] * (max_idx + 1 if max_idx >= 0 else 0)
    for idx in indices:
        if idx >= 0:
            dense[idx] = 1
    return dense

def retrieve_candidate_dict(
    db5: pd.DataFrame,
    masses: List[float],
    ppm: float,
    *,
    selected_fp_idx: Optional[List[int]] = None,
    fp_filter_idx: Optional[List[int]] = None,
) -> Dict[Tuple[str, str, str], List[int]]:
    """Return map: (CompoundName, InChIKey, Formula) -> fingerprint bits list."""
    mass_col, fingerprint_col, smiles_col, inchikey_col, name_col, formula_col = find_db5_columns(db5)
    if mass_col is None or (fingerprint_col is None and smiles_col is None):
        return {}
    mask = np.zeros(len(db5), dtype=bool)
    for m in masses:
        lo, hi = ppm_window(m, ppm)
        mask |= db5[mass_col].astype(float).between(lo, hi)
    sub = db5.loc[mask].copy()
    out = {}
    for _, row in sub.iterrows():
        fp = None
        if fingerprint_col:
            fp = _decode_fingerprint(row.get(fingerprint_col))
        if fp is None and smiles_col:
            fp = compute_morgan_fp_bits(row.get(smiles_col, None))
        if fp is None:
            continue

        if fp_filter_idx:
            if max(fp_filter_idx) >= len(fp):
                continue
            fp = [fp[i] for i in fp_filter_idx]

        if selected_fp_idx:
            if max(selected_fp_idx) >= len(fp):
                continue
            fp = [fp[i] for i in selected_fp_idx]

        fp = [int(v) for v in fp]
        name = str(row.get(name_col, "")) if name_col else ""
        inchikey = str(row.get(inchikey_col, "")) if inchikey_col else ""
        formula = str(row.get(formula_col, "")) if formula_col else ""
        out[(name, inchikey, formula)] = fp
    return out
