
import pickle
import pandas as pd

def select_bins(perf_pkl_path: str, top_n: int = 500):
    df = pd.read_pickle(perf_pkl_path)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if "f1_score_val" not in df.columns or "bin_index" not in df.columns:
        raise ValueError("Performance PKL must contain columns: f1_score_val, bin_index")
    top = df.sort_values("f1_score_val", ascending=False).head(top_n)
    return top["bin_index"].tolist()

def select_fp_indices(f1_per_fp_pkl: str, thresh: float = 0.85):
    with open(f1_per_fp_pkl, "rb") as f:
        arr = pickle.load(f)
    import numpy as np
    arr = list(arr)
    idx = [i for i,v in enumerate(arr) if v >= thresh]
    return idx
