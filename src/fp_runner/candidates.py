from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from .utils import ppm_window

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
    for c in ["MonoisotopicMass", "MONOISOTOPIC_MASS", "ExactMass", "EXACT_MASS", "AverageMass"]:
        if c in df.columns:
            mass_col = c
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
    for c in ["Formula", "molecularFormula", "MOLECULAR_FORMULA"]:
        if c in df.columns:
            formula_col = c
            break
    return mass_col, smiles_col, inchikey_col, name_col, formula_col

def retrieve_candidate_dict(
    db5: pd.DataFrame,
    masses: List[float],
    ppm: float
) -> Dict[Tuple[str, str, str], List[int]]:
    """Return map: (CompoundName, InChIKey, Formula) -> fingerprint bits list."""
    mass_col, smiles_col, inchikey_col, name_col, formula_col = find_db5_columns(db5)
    if mass_col is None or smiles_col is None:
        return {}
    mask = np.zeros(len(db5), dtype=bool)
    for m in masses:
        lo, hi = ppm_window(m, ppm)
        mask |= db5[mass_col].astype(float).between(lo, hi)
    sub = db5.loc[mask].copy()
    out = {}
    for _, row in sub.iterrows():
        smiles = row.get(smiles_col, None)
        fp = compute_morgan_fp_bits(smiles)
        if fp is None:
            continue
        name = str(row.get(name_col, "")) if name_col else ""
        inchikey = str(row.get(inchikey_col, "")) if inchikey_col else ""
        formula = str(row.get(formula_col, "")) if formula_col else ""
        out[(name, inchikey, formula)] = fp
    return out
