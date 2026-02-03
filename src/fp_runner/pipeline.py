import os, sys, pickle, ast
from typing import List, Optional, Dict, Tuple, Callable, Iterable, Iterator, Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from .model import load_keras_model

try:
    from tqdm import tqdm  
except Exception:  
    tqdm = None


def _call_progress(progress: Optional[Callable[[str], None]], message: str):
    if progress is None:
        print(message, file=sys.stderr, flush=True)
    else:
        try:
            progress(message)
        except Exception:
            print(message, file=sys.stderr, flush=True)


def _iter_with_progress(
    iterable: Iterable[Any],
    *,
    total: Optional[int],
    desc: str,
    progress: Optional[Callable[[str], None]],
    unit: str = "item",
) -> Iterator[Any]:
    if tqdm is not None and total is not None:
        yield from tqdm(iterable, total=total, desc=desc, unit=unit)
        return

    if total is None:
        for x in iterable:
            yield x
        return

    total = int(total)
    if total <= 0:
        for x in iterable:
            yield x
        return

    every = max(1, total // 50)
    i = 0
    for x in iterable:
        i += 1
        if i == 1 or i == total or (i % every == 0):
            _call_progress(progress, f"{desc}: {i}/{total} ({int(round(100*i/total))}%)")
        yield x


def _install_numpy_aliases_and_shims():
    import numpy as _np
    aliases = [
        "numpy._core","numpy._core.multiarray","numpy.core","numpy.core.multiarray",
        "numpy.core.numerictypes","numpy.core.umath","numpy.core._multiarray_umath","numpy.multiarray",
    ]
    for name in aliases:
        if name not in sys.modules:
            sys.modules[name] = _np
    def _np_reconstruct(subtype, shape, dtype):
        try:
            return _np.empty(shape, dtype=dtype).view(subtype)
        except Exception:
            arr = _np.empty(shape, dtype=dtype)
            try: arr.__class__ = subtype
            except Exception: pass
            return arr
    for modname in ["numpy","numpy.core","numpy.core.multiarray","numpy._core","numpy._core.multiarray","numpy.multiarray"]:
        mod = sys.modules.get(modname)
        if mod is not None and not hasattr(mod, "_reconstruct"):
            try: setattr(mod, "_reconstruct", _np_reconstruct)
            except Exception: pass

class _NumpyCompatUnpickler(pickle.Unpickler):
    _MAP = {
        "numpy._core":"numpy","numpy._core.multiarray":"numpy","numpy.core":"numpy",
        "numpy.core.multiarray":"numpy","numpy.core.numerictypes":"numpy","numpy.core.umath":"numpy",
        "numpy.core._multiarray_umath":"numpy","numpy.multiarray":"numpy",
    }
    def find_class(self, module, name):
        for k, v in self._MAP.items():
            if module == k or module.startswith(k + "."):
                module = module.replace(k, v, 1)
                break
        if module == "numpy" and name == "_reconstruct":
            _install_numpy_aliases_and_shims()
            import numpy as _np
            return getattr(_np, "_reconstruct")
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name)

def safe_pd_read_pickle(path):
    _install_numpy_aliases_and_shims()
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return _NumpyCompatUnpickler(f).load()

def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def select_bins(perf_pkl: str, top_n: int) -> List[int]:
    """
    Accepts performance files that may use any of these metrics:
      f1_score_val, f1_micro_val, f1, f1_val
    and a bin index column:
      bin_index (required)
    Sorts descending by the chosen metric and returns top-N bin indices.
    """
    df = safe_pd_read_pickle(perf_pkl)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    metric_col = _first_existing_column(
        df,
        ["f1_score_val", "f1_micro_val", "f1_val", "f1"]
    )
    if metric_col is None:
        raise ValueError("Performance PKL must contain one of: f1_score_val, f1_micro_val, f1_micro_val, f1_val, f1")

    if "bin_index" not in df.columns:
        raise ValueError("Performance PKL must contain column: bin_index")

    df = df.sort_values(metric_col, ascending=False).head(top_n)
    return df["bin_index"].astype(int).tolist()

def select_fp_indices(f1_pkl: str, thresh: float) -> List[int]:
    _install_numpy_aliases_and_shims()
    with open(f1_pkl, "rb") as f:
        arr = _NumpyCompatUnpickler(f).load()
    arr = np.asarray(arr, float)
    return np.where(arr >= float(thresh))[0].astype(int).tolist()


def load_fp_filter(fp_filter_pkl: Optional[str]) -> Optional[List[int]]:
    if fp_filter_pkl is None:
        return None
    _install_numpy_aliases_and_shims()

    with open(fp_filter_pkl, "rb") as f:
        head = f.read(128)

    is_csv = fp_filter_pkl.lower().endswith((".csv", ".tsv")) or head.startswith(b"Standard_InChIKey,")

    if not is_csv:
        with open(fp_filter_pkl, "rb") as f:
            idx = _NumpyCompatUnpickler(f).load()
        return list(map(int, idx))

    cols = head.splitlines()[0].decode("utf-8", errors="ignore").split(",")
    pick = next((c for c in ("idx", "index", "cid") if c in cols), None)

    out: List[int] = []
    offset = 0
    for chunk in pd.read_csv(fp_filter_pkl, usecols=[pick] if pick else None, chunksize=200_000):
        if pick:
            s = pd.to_numeric(chunk[pick], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            out.extend(s.astype("int64").tolist())
        else:
            n = len(chunk)
            out.extend(range(offset, offset + n))
            offset += n

    return out



def get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def decode_fp_indices(indices, length: int = 7293) -> List[int]:
    fp = [0] * length
    for x in indices:
        try:
            i = int(x)
        except Exception:
            continue
        if 0 <= i < length:
            fp[i] = 1
    return fp

def parse_fp_cell(val) -> Optional[List[int]]:
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return [int(v) for v in val]
    s = str(val).strip()
    if not s:
        return None
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    if not s:
        return []
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out

def tanimoto(a: List[int], b: List[int]) -> float:
    x = np.asarray(a, int)
    y = np.asarray(b, int)
    if x.shape != y.shape:
        return 0.0
    inter = int(((x == 1) & (y == 1)).sum())
    union = int(((x == 1) | (y == 1)).sum())
    return inter / union if union else 0.0

def _coerce_vector(v) -> np.ndarray:
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, float)
    try:
        val = ast.literal_eval(str(v))
        if isinstance(val, (list, tuple, np.ndarray)):
            return np.asarray(val, float)
    except Exception:
        pass
    raise ValueError("Spectrum Vector row is not a list/array and cannot be parsed.")

def run_pipeline(
    casmi_df: pd.DataFrame,
    db5_df: pd.DataFrame,
    model_path: str,
    bins_perf_pkl: str,
    f1_fp_pkl: str,
    fp_f1_thresh: float,
    fp_filter_pkl: Optional[str],
    ion_mode: str,
    ppm: float,
    top_bins: int,
    out_dir: str,
    progress: Optional[Callable[[str], None]] = None,   # <-- new
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # 1) feature selection + model
    _call_progress(progress, "[1/6] Selecting top bins")
    bins = select_bins(bins_perf_pkl, top_bins)

    _call_progress(progress, "[2/6] Selecting fingerprint indices (F1 threshold)")
    sel_fp_idx = select_fp_indices(f1_pkl=f1_fp_pkl, thresh=fp_f1_thresh)
    n_sel = len(sel_fp_idx)

    _call_progress(progress, "[3/6] Loading fingerprint filter (if provided)")
    fp_filter = load_fp_filter(fp_filter_pkl)

    _call_progress(progress, "[4/6] Loading Keras model")
    model = load_keras_model(model_path)

    # 2) bin spectra
    spec_col = "Spectrum Vector"
    if spec_col not in casmi_df.columns:
        raise ValueError(f"Missing '{spec_col}' column in CASMI data.")

    def _subspec(v):
        a = _coerce_vector(v)
        return [float(a[i]) for i in bins]

    _call_progress(progress, "[5/6] Subsetting spectra to selected bins")
    casmi = casmi_df.copy()
    n_rows = len(casmi)
    casmi[spec_col] = [
        _subspec(v)
        for v in _iter_with_progress(casmi[spec_col].tolist(), total=n_rows, desc="Binning spectra", progress=progress, unit="row")
    ]

    # 3) predict fingerprints
    def _pred_fp(v):
        a = np.asarray(v, float).reshape(1, -1)
        p = model.predict(a, verbose=0)[0]
        if n_sel > 0:
            if p.shape[0] == n_sel:
                fp = p
            else:
                fp = p[sel_fp_idx]
        else:
            fp = p
        return np.round(fp).astype(int).tolist()

    _call_progress(progress, "[6/6] Predicting fingerprints")
    casmi["fp_pred"] = [
        _pred_fp(v)
        for v in _iter_with_progress(casmi[spec_col].tolist(), total=n_rows, desc="Predicting FPs", progress=progress, unit="row")
    ]

    # 4) key columns
    chall_col   = get_col(casmi, [
        "ChallengeName","Challenge","Compound Number","challenge","id","input_row_idx"
    ])
    prec_col    = get_col(casmi, [
        "Precursor m/z (Da)","PRECURSOR_MZ","precursor_mz","PRECURSOR_MZ_Da"
    ])
    formula_col = get_col(casmi, [
        "Actual Molecular Formula","Formula","Molecular Formula","molecularFormula","formula"
    ])

    if chall_col is None:
        chall_col = "_challenge_auto_"
        casmi[chall_col] = [f"challenge_{i:06d}" for i in range(len(casmi))]

    if prec_col is None:
        raise ValueError("Could not find precursor m/z column (e.g., 'precursor_mz').")

    # DB5 columns
    mass_col = get_col(db5_df, ["monoisotopic_mass","exact_mass","MonoisotopicMass","ExactMass"])
    name_col = get_col(db5_df, ["name","Name","MassBank_Name"])
    inch_col = get_col(db5_df, ["InChIKey","InChIkey2D","inchikey","InChI Key"])
    form_col = get_col(db5_df, ["formula","molecularFormula","Formula","Molecular Formula"])
    if mass_col is None:
        raise ValueError("DB5 mass column not found.")
    mass_numeric = pd.to_numeric(db5_df[mass_col], errors="coerce")

    # 5) loop over challenges
    results = []
    mode = str(ion_mode).lower()
    row_iter = _iter_with_progress(
        casmi.iterrows(),
        total=len(casmi),
        desc="Candidate retrieval",
        progress=progress,
        unit="challenge",
    )

    for _, row in row_iter:
        cname = str(row[chall_col])
        try:
            pmz = float(row[prec_col])
        except Exception:
            continue

        # adducts
        if mode in ("positive","pos","p"):
            masses = [pmz - 1.007276, pmz - 18.033823, pmz - 22.989218]
        elif mode in ("negative","neg","n"):
            masses = [pmz + 1.007276, pmz - 34.969402, pmz - 44.998201]
        else:
            raise ValueError("ion_mode must be positive/negative.")

        fp_pred = row["fp_pred"]
        pred_len = len(fp_pred)

        # candidate retrieval
        cand_map: Dict[Tuple[str, str, str], List[int]] = {}
        for mass in masses:
            if mass is None or not np.isfinite(mass):
                continue
            min_w = mass * 1_000_000.0 / (1_000_000.0 + ppm)
            max_w = mass * 1_000_000.0 / (1_000_000.0 - ppm)
            msk = (mass_numeric >= min_w) & (mass_numeric <= max_w)
            sub = db5_df.loc[msk]
            if sub.empty:
                continue
            for _, r2 in sub.iterrows():
                idx_list = parse_fp_cell(r2.get("fingerprint", None))
                if idx_list is None:
                    continue

                full_fp = decode_fp_indices(idx_list, length=7293)
                print("len(full_fp)=", len(full_fp), "max_filter=", max(fp_filter), "min_filter=", min(fp_filter))
                base_fp = [full_fp[i] for i in fp_filter] if fp_filter is not None else full_fp

                if n_sel > 0:
                    if len(base_fp) == pred_len:
                        cand_fp = base_fp
                    else:
                        cand_fp = [base_fp[i] for i in sel_fp_idx if i < len(base_fp)]
                else:
                    cand_fp = base_fp

                cand_fp = [int(v) for v in cand_fp]

                name = str(r2[name_col]) if name_col else ""
                inch = str(r2[inch_col]) if inch_col else ""
                form = str(r2[form_col]) if form_col else ""
                cand_map[(name, inch, form)] = cand_fp

        rows, tanis = [], []
        for (name, inch, form), cand_fp in cand_map.items():
            t = tanimoto(cand_fp, fp_pred)
            tanis.append(t)
            rows.append([name, inch, form, f"{t:.6f}"])
        rows_sorted = sorted(rows, key=lambda x: float(x[3]), reverse=True)

        out_path = os.path.join(out_dir, f"{cname}_prediction.txt")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(cname + "\n")
            if rows_sorted:
                f.write(tabulate(rows_sorted, headers=["Compound Name","InChIKey","Formula","Tanimoto"]) + "\n")
            else:
                f.write("(no candidates found)\n")

        act_form = str(row[formula_col]) if (formula_col and formula_col in casmi.columns) else None
        top_form = rows_sorted[0][2] if rows_sorted else ""
        correct = (top_form == act_form) if act_form is not None else None

        results.append({
            "challenge": cname,
            "top_formula": top_form,
            "actual_formula": act_form,
            "correct": correct,
            "mean_tanimoto": float(np.mean(tanis)) if tanis else None,
        })

    summary_path = os.path.join(out_dir, "summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False, encoding="utf-8", lineterminator="\n")
    _call_progress(progress, f"Wrote summary: {summary_path}")
    return summary_path
