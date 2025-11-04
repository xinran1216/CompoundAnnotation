
from typing import Dict, Tuple, List
import os, json
import numpy as np
import pandas as pd
from tabulate import tabulate
from .features import select_bins, select_fp_indices
from .model import load_keras_model
from .candidates import retrieve_candidate_dict
from .scoring import tanimoto_from_lists, blend_score, build_sirius_lookup, normalize_sirius_within_challenge

def calculate_adduct(pm: float, mode: str):
    pos = {'M+H': pm - 1.007276, 'M+NH4': pm - 18.033823, 'M+Na': pm - 22.989218}
    neg = {'M-H': pm + 1.007276, 'M+Cl': pm - 34.969402, 'M+FA-H': pm - 44.998201}
    if mode == "positive":
        return list(pos.values())
    elif mode == "negative":
        return list(neg.values())
    raise ValueError("ion mode must be 'positive' or 'negative'")

def predict_fingerprint(binned_vector: List[float], model, selected_fp_idx: List[int]):
    import numpy as np
    arr = np.array(binned_vector, dtype=float).reshape(1, -1)
    pred = model.predict(arr, verbose=0)[0]
    pred = pred[selected_fp_idx]
    return np.round(pred).astype(int).tolist()

def bin_and_predict(df: pd.DataFrame, bins: List[int], model, selected_fp_idx: List[int]):
    df = df.copy()
    df["Spectrum Vector"] = df["Spectrum Vector"].apply(lambda x: [x[i] for i in bins])
    df = df.dropna(subset=["fp_decode"]) if "fp_decode" in df.columns else df
    df["fp_predicted"] = df["Spectrum Vector"].apply(lambda x: predict_fingerprint(x, model, selected_fp_idx))
    return df

def run_year(
    df: pd.DataFrame,
    db5: pd.DataFrame,
    sirius_df: pd.DataFrame | None,
    bins_perf_pkl: str,
    f1_per_fp_pkl: str,
    model_path: str,
    ion_mode: str,
    ppm: float,
    out_dir: str,
    challenge_id_col_guess=(\"ChallengeName\",\"Challenge\",\"id\",\"ID\"),
    precursor_col_guess=(\"PRECURSOR_MZ\",\"precursor_mz\",\"precursorMz\"),
    formula_col_guess=(\"Actual Molecular Formula\",\"Formula\",\"molecularFormula\"),
):
    os.makedirs(out_dir, exist_ok=True)

    bins = select_bins(bins_perf_pkl, top_n=500)
    fp_idx = select_fp_indices(f1_per_fp_pkl, thresh=0.85)
    model = load_keras_model(model_path)

    sirius_norm = None
    if sirius_df is not None and len(sirius_df):
        sirius_norm = normalize_sirius_within_challenge(sirius_df)

    cid_col = next((c for c in challenge_id_col_guess if c in df.columns), None)
    prec_col = next((c for c in precursor_col_guess if c in df.columns), None)
    if cid_col is None or prec_col is None or "Spectrum Vector" not in df.columns:
        raise ValueError("CASMI df must include challenge id, PRECURSOR_MZ, and Spectrum Vector")

    df_pred = bin_and_predict(df, bins, model, fp_idx)

    results = []
    for i, row in df_pred.iterrows():
        name = str(row[cid_col])
        challenge_id = None
        for tok in str(name).replace("_","-").split("-")[::-1]:
            if tok.isdigit():
                challenge_id = int(tok); break
        if challenge_id is None:
            try:
                challenge_id = int(str(name))
            except:
                continue

        masses = calculate_adduct(float(row[prec_col]), ion_mode)
        cand = retrieve_candidate_dict(db5, masses, ppm)

        sirius_map = {}
        if sirius_norm is not None and len(sirius_norm):
            sub = sirius_norm[(sirius_norm["challenge_id"] == challenge_id)]
            sirius_map = dict(zip(sub["molecularFormula"], sub["SiriusScore"]))

        compound_scores = {}
        tanis = []
        for key, fp_bits in cand.items():
            t = tanimoto_from_lists(fp_bits, row["fp_predicted"])
            tanis.append(t)
            form = key[2] if isinstance(key, tuple) and len(key) >= 3 else ""
            sscore = sirius_map.get(form, None)
            compound_scores[key] = blend_score(t, sscore)

        # Write per-challenge file
        lines = []
        for (nm, inchikey, formula), score in sorted(compound_scores.items(), key=lambda kv: kv[1], reverse=True):
            lines.append([str(nm), str(inchikey), str(formula), f"{score:.6f}"])
        out_txt = os.path.join(out_dir, f"{name}_prediction.txt")
        from tabulate import tabulate
        with open(out_txt, "w") as f:
            f.write(str(name) + "\\n")
            if lines:
                f.write(tabulate(lines, headers=["Compound Name","InChIKey","Formula","Score"]) + "\\n")
            else:
                f.write("(no candidates found)\\n")

        # Accuracy vs. actual formula if present
        actual_formula = None
        for c in formula_col_guess:
            if c in df_pred.columns:
                actual_formula = str(row[c]); break
        top_formula = lines[0][2] if lines else ""
        correct = (top_formula == actual_formula) if actual_formula else None
        results.append({
            "challenge": str(name),
            "top_formula": top_formula,
            "actual_formula": actual_formula,
            "correct": correct,
            "mean_tanimoto": float(np.mean(tanis)) if len(tanis) else None,
        })

    out_csv = os.path.join(out_dir, "summary.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    return out_csv
