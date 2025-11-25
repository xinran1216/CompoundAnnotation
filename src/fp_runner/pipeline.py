from typing import Dict, Tuple, List, Optional
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
import pickle

from .features import select_bins, select_fp_indices
from .model import load_keras_model
from .candidates import retrieve_candidate_dict
from .scoring import (
    tanimoto_from_lists,
    blend_score,
    normalize_sirius_within_challenge,
)

def calculate_adduct(pm: float, mode: str) -> List[float]:
    # adduct-adjusted neutral masses to search around (simple offsets)
    pos = {'M+H': pm - 1.007276, 'M+NH4': pm - 18.033823, 'M+Na': pm - 22.989218}
    neg = {'M-H': pm + 1.007276, 'M+Cl': pm - 34.969402, 'M+FA-H': pm - 44.998201}
    if mode == "positive":
        return list(pos.values())
    elif mode == "negative":
        return list(neg.values())
    raise ValueError("ion mode must be 'positive' or 'negative'")

def predict_fingerprint(binned_vector: List[float], model, selected_fp_idx: List[int]) -> List[int]:
    arr = np.array(binned_vector, dtype=float).reshape(1, -1)
    pred = model.predict(arr, verbose=0)[0]
    pred = pred[selected_fp_idx]
    return np.round(pred).astype(int).tolist()

def bin_and_predict(df: pd.DataFrame, bins: List[int], model, selected_fp_idx: List[int]) -> pd.DataFrame:
    df = df.copy()
    df["Spectrum Vector"] = df["Spectrum Vector"].apply(lambda x: [x[i] for i in bins])
    # If ground truth fp exists, keep; otherwise just proceed
    if "fp_decode" in df.columns:
        df = df.dropna(subset=["fp_decode"])
    df["fp_predicted"] = df["Spectrum Vector"].apply(lambda x: predict_fingerprint(x, model, selected_fp_idx))
    return df

def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _parse_challenge_id(value) -> Optional[int]:
    s = str(value)
    # robust: look for trailing numeric token split by '-' or '_'
    tokens = s.replace("_", "-").split("-")
    for tok in reversed(tokens):
        if tok.isdigit():
            return int(tok)
    try:
        return int(s)
    except Exception:
        return None

def run_year(
    df: pd.DataFrame,
    db5: pd.DataFrame,
    sirius_df: Optional[pd.DataFrame],
    bins_perf_pkl: str,
    f1_per_fp_pkl: str,
    model_path: str,
    ion_mode: str,
    ppm: float,
    out_dir: str,
    fp_filter_pkl: Optional[str] = None,
    top_bins: int = 500,
    challenge_id_col_guess: Tuple[str, ...] = ("ChallengeName", "Challenge", "id", "ID"),
    precursor_col_guess: Tuple[str, ...] = ("PRECURSOR_MZ", "precursor_mz", "precursorMz"),
    formula_col_guess: Tuple[str, ...] = ("Actual Molecular Formula", "Formula", "molecularFormula"),
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # Select spectral bins and fp indices
    bins = select_bins(bins_perf_pkl, top_n=top_bins)
    fp_idx = select_fp_indices(f1_per_fp_pkl, thresh=0.85)
    model = load_keras_model(model_path)

    fp_filter_idx = None
    if fp_filter_pkl:
        with open(fp_filter_pkl, "rb") as f:
            fp_filter_idx = list(pickle.load(f))

    # Normalize Sirius scores per challenge if present
    sirius_norm = None
    if sirius_df is not None and len(sirius_df):
        sirius_norm = normalize_sirius_within_challenge(sirius_df)

    # Column resolution
    cid_col = _find_first_col(df, list(challenge_id_col_guess))
    prec_col = _find_first_col(df, list(precursor_col_guess))
    if cid_col is None or prec_col is None or "Spectrum Vector" not in df.columns:
        raise ValueError("Input CASMI dataframe must include challenge id, PRECURSOR_MZ, and Spectrum Vector columns")

    # Predict fingerprints on binned spectra
    df_pred = bin_and_predict(df, bins, model, fp_idx)

    results = []
    for _, row in df_pred.iterrows():
        name = row[cid_col]
        challenge_id = _parse_challenge_id(name)
        if challenge_id is None:
            # skip if we cannot determine challenge id
            continue

        # Build candidate set around adduct-adjusted masses
        masses = calculate_adduct(float(row[prec_col]), ion_mode)
        cand = retrieve_candidate_dict(
            db5,
            masses,
            ppm,
            selected_fp_idx=fp_idx,
            fp_filter_idx=fp_filter_idx,
        )

        # per-challenge sirius map
        sirius_map = {}
        if sirius_norm is not None and len(sirius_norm):
            sub = sirius_norm[sirius_norm["challenge_id"] == challenge_id]
            sirius_map = dict(zip(sub["molecularFormula"], sub["SiriusScore"]))

        compound_scores = {}
        tanis = []
        for key, fp_bits in cand.items():
            t = tanimoto_from_lists(fp_bits, row["fp_predicted"])
            tanis.append(t)
            form = key[2] if isinstance(key, tuple) and len(key) >= 3 else ""
            sscore = sirius_map.get(form)
            compound_scores[key] = blend_score(t, sscore)

        # Write per-challenge file
        rows = []
        for (nm, inchikey, formula), score in sorted(compound_scores.items(), key=lambda kv: kv[1], reverse=True):
            rows.append([str(nm), str(inchikey), str(formula), f"{score:.6f}"])
        out_txt = os.path.join(out_dir, f"{name}_prediction.txt")
        with open(out_txt, "w") as f:
            f.write(str(name) + "\n")
            if rows:
                f.write(tabulate(rows, headers=["Compound Name", "InChIKey", "Formula", "Score"]) + "\n")
            else:
                f.write("(no candidates found)\n")

        # Accuracy vs. actual formula if present
        actual_formula = None
        for c in formula_col_guess:
            if c in df_pred.columns:
                actual_formula = str(row[c])
                break
        top_formula = rows[0][2] if rows else ""
        correct = (top_formula == actual_formula) if actual_formula else None
        results.append({
            "challenge": str(name),
            "top_formula": top_formula,
            "actual_formula": actual_formula,
            "correct": correct,
            "mean_tanimoto": float(np.mean(tanis)) if tanis else None,
        })

    out_csv = os.path.join(out_dir, "summary.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    return out_csv
