
import numpy as np
import pandas as pd

def coalesce_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def minmax_grouped(series: pd.Series):
    v = series.values.reshape(-1,1).astype(np.float64)
    mn = float(np.min(v))
    mx = float(np.max(v))
    if mx-mn < 1e-12:
        return np.zeros_like(series, dtype=float)
    return ((v - mn)/(mx-mn)).ravel()

def ppm_window(target_mass: float, ppm: float):
    delta = target_mass * ppm * 1e-6
    return target_mass - delta, target_mass + delta
