from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from .utils import minmax_grouped

def tanimoto_from_lists(real_fp: List[int], pred_fp: List[int]) -> float:
    a = np.array(real_fp, dtype=int)
    b = np.array(pred_fp, dtype=int)
    if a.shape != b.shape:
        # zero Tanimoto if different lengths (shouldn’t happen after selection)
        return 0.0
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return (inter / union) if union else 0.0

def blend_score(tanimoto: float, sirius_score: Optional[float]) -> float:
    if sirius_score is None:
        return tanimoto
    return 0.8 * tanimoto + 0.2 * float(sirius_score)

def build_sirius_lookup(
    tsv: pd.DataFrame,
    id_col_guess=("id", "spec", "spectrumId", "spectrum_id"),
    formula_col="molecularFormula",
    score_col_guess=("SiriusScore", "score", "scores"),
) -> pd.DataFrame:
    ic = None
    for c in id_col_guess:
        if c in tsv.columns:
            ic = c
            break
    sc = None
    for c in score_col_guess:
        if c in tsv.columns:
            sc = c
            break
    if ic is None or sc is None or formula_col not in tsv.columns:
        return pd.DataFrame(columns=["challenge_id", "molecularFormula", "SiriusScore"])

    df = tsv.copy()

    def _cid(x):
        s = str(x)
        for tok in s.replace("_", "-").split("-")[::-1]:
            if tok.isdigit():
                return int(tok)
        try:
            return int(s)
        except Exception:
            return None

    df["challenge_id"] = df[ic].map(_cid)
    df = df.dropna(subset=["challenge_id"])
    df = df.rename(columns={sc: "SiriusScore"})
    return df[["challenge_id", "molecularFormula", "SiriusScore"]]

def normalize_sirius_within_challenge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SiriusScore"] = df.groupby("challenge_id")["SiriusScore"].transform(minmax_grouped)
    return df
