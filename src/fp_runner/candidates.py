from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from .utils import ppm_window
import ast


def _parse_masses(value) -> List[float]:
    """Return a list of numeric masses from a DB5 mass cell.

    Handles scalars, lists/tuples, and stringified lists. Invalid entries return
    an empty list.
    """

    def _coerce(v):
        try:
            return float(v)
        except Exception:
            return None

    if value is None:
        return []
    if isinstance(value, (int, float)):
        mass = _coerce(value)
        return [mass] if mass is not None else []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [m for m in (_coerce(v) for v in value) if m is not None]
    if isinstance(value, str):
        stripped = value.strip()
        # Strings like "[107.04, 131.05]" should be treated as a list of masses
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [m for m in (_coerce(v) for v in parsed) if m is not None]
            except Exception:
                return []
        mass = _coerce(stripped)
        return [mass] if mass is not None else []
    return []

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

def _decode_fingerprint(value, *, target_len: Optional[int] = None) -> Optional[List[int]]:
    """Decode a stored fingerprint (sparse list of 1 positions) into dense bits.

    If ``target_len`` is provided, the returned list is padded with zeros to that
    length so downstream index selection (fp filter + selected fp indices) can
    mirror the notebook logic without discarding candidates whose stored
    fingerprint is shorter than the requested indices.
    """

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
    length = max(max_idx + 1, target_len or 0)
    dense = [0] * max(length, 0)
    for idx in indices:
        if idx >= 0 and idx < len(dense):
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
    mass_lists = db5[mass_col].apply(_parse_masses)
    mask = np.zeros(len(db5), dtype=bool)
    for m in masses:
        lo, hi = ppm_window(m, ppm)
        mask |= mass_lists.apply(lambda vals: any(lo <= v <= hi for v in vals)).to_numpy()
    sub = db5.loc[mask].copy()
    needed_len = -1
    if fp_filter_idx:
        needed_len = max(needed_len, max(fp_filter_idx))
    if selected_fp_idx:
        needed_len = max(needed_len, max(selected_fp_idx))
    needed_len = (needed_len + 1) if needed_len >= 0 else None

    def _pick_indices(fp_bits: List[int], indices: List[int]) -> List[int]:
        if not indices:
            return fp_bits
        # Preserve ascending order of requested indices like the notebook loop
        ordered = sorted(indices)
        return [fp_bits[i] if i < len(fp_bits) else 0 for i in ordered]

    out = {}
    for _, row in sub.iterrows():
        fp = None
        if fingerprint_col:
            fp = _decode_fingerprint(row.get(fingerprint_col), target_len=needed_len)
        if fp is None and smiles_col:
            fp = compute_morgan_fp_bits(
                row.get(smiles_col, None), n_bits=needed_len or 2048
            )
        if fp is None:
            continue

        if fp_filter_idx:
            fp = _pick_indices(fp, fp_filter_idx)

        if selected_fp_idx:
            fp = _pick_indices(fp, selected_fp_idx)

        fp = [int(v) for v in fp]
        name = str(row.get(name_col, "")) if name_col else ""
        inchikey = str(row.get(inchikey_col, "")) if inchikey_col else ""
        formula = str(row.get(formula_col, "")) if formula_col else ""
        out[(name, inchikey, formula)] = fp
    return out
