import os, sys, argparse, pickle, json, ast
import numpy as np
import pandas as pd

def _install_numpy_aliases_and_shims():
    aliases = [
        "numpy._core","numpy._core.multiarray","numpy.core","numpy.core.multiarray",
        "numpy.core.numerictypes","numpy.core.umath","numpy.core._multiarray_umath","numpy.multiarray",
    ]
    for name in aliases:
        if name not in sys.modules:
            sys.modules[name] = np

    def _np_reconstruct(subtype, shape, dtype):
        try:
            return np.empty(shape, dtype=dtype).view(subtype)
        except Exception:
            arr = np.empty(shape, dtype=dtype)
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
        for k,v in self._MAP.items():
            if module == k or module.startswith(k + "."):
                module = module.replace(k, v, 1)
                break
        if module == "numpy" and name == "_reconstruct":
            _install_numpy_aliases_and_shims()
            return getattr(np, "_reconstruct")
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name)

def _read_pickle_tolerant(path):
    _install_numpy_aliases_and_shims()
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return _NumpyCompatUnpickler(f).load()

def _coerce_1d(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _to_dataframe(obj):
    if isinstance(obj, pd.DataFrame): return obj
    if isinstance(obj, dict) and "data" in obj and "columns" in obj:
        return pd.DataFrame(obj["data"], columns=obj["columns"])
    if isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(obj[1], (list, tuple)):
        data, cols = obj
        try: return pd.DataFrame(data, columns=list(cols))
        except Exception: pass
    if isinstance(obj, (list, tuple)) and (not obj or isinstance(obj[0], dict)):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        norm = {k: _coerce_1d(v) for k,v in obj.items()}
        try: return pd.DataFrame(norm)
        except Exception:
            norm2 = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in norm.items()}
            return pd.DataFrame(norm2)
    if isinstance(obj, np.ndarray):
        try: return pd.DataFrame.from_records(obj)
        except Exception: return pd.DataFrame({"value":[obj]})
    return pd.DataFrame(obj)

def read_any_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".json", ".jsonl", ".ndjson"):
        try:    return pd.read_json(path, lines=True)
        except ValueError:
                return pd.read_json(path)
    try:
        obj = _read_pickle_tolerant(path)
        return _to_dataframe(obj)
    except Exception:
        try:    return pd.read_csv(path, low_memory=False)
        except Exception as e:
            raise ValueError(f"Could not read file '{path}' as pickle/csv/parquet/json") from e

def parse_listlike(cell):
    if isinstance(cell, (list, tuple, np.ndarray)):
        return list(cell)
    if pd.isna(cell): return None
    s = str(cell).strip()
    if not s: return None
    try:
        val = json.loads(s)
        if isinstance(val, (list, tuple, np.ndarray)):
            return [float(x) for x in list(val)]
    except Exception:
        pass
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple, np.ndarray)):
            return [float(x) for x in list(val)]
    except Exception:
        pass
    try:
        return [float(x) for x in s.split(",")]
    except Exception:
        return None

from .pipeline import run_pipeline

def parse_args():
    p = argparse.ArgumentParser(description="CASMI formula prediction & candidate retrieval.")
    p.add_argument("--model", required=True)
    p.add_argument("--bins-performance-pkl", required=True)
    p.add_argument("--f1-per-fp-pkl", required=True)
    p.add_argument("--fp-f1-threshold", type=float, default=0.85)
    p.add_argument("--fp-filter-pkl", default=None)
    p.add_argument("--db5-csv", required=True)
    p.add_argument("--test-pkl", required=True)   # may be .csv/.parquet/.json/.pkl
    p.add_argument("--ion-mode", choices=["positive", "negative", "pos", "neg", "p", "n"], required=True)
    p.add_argument("--ppm", type=float, required=True)
    p.add_argument("--top-bins", type=int, default=500)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()

def main():
    args = parse_args()

    df = read_any_table(args.test_pkl)
    df = _to_dataframe(df)

    if "Spectrum Vector" in df.columns and len(df):
        if not isinstance(df["Spectrum Vector"].iloc[0], (list, tuple, np.ndarray)):
            df["Spectrum Vector"] = df["Spectrum Vector"].apply(parse_listlike)

    if "Spectrum Vector" not in df.columns:
        raise ValueError("Input must contain 'Spectrum Vector' (list/array of floats).")
    if df["Spectrum Vector"].isna().any():
        bad = int(df["Spectrum Vector"].isna().sum())
        raise ValueError(f"{bad} rows in 'Spectrum Vector' are empty/unparseable after reading your file.")

    db5_df = pd.read_csv(args.db5_csv, low_memory=False)
    os.makedirs(args.out_dir, exist_ok=True)

    summary_csv = run_pipeline(
        casmi_df=df,
        db5_df=db5_df,
        model_path=args.model,
        bins_perf_pkl=args.bins_performance_pkl,
        f1_fp_pkl=args.f1_per_fp_pkl,
        fp_f1_thresh=args.fp_f1_threshold,
        fp_filter_pkl=args.fp_filter_pkl,
        ion_mode=args.ion_mode,
        ppm=args.ppm,
        top_bins=args.top_bins,
        out_dir=args.out_dir,
    )
    print("Done. Summary:", summary_csv)

if __name__ == "__main__":
    main()
