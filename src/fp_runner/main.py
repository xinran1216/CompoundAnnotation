import os
import argparse
import pandas as pd

from .pipeline import run_year
from .scoring import build_sirius_lookup

def parse_args():
    p = argparse.ArgumentParser(description="Single CASMI test-set runner (any year).")
    p.add_argument("--model", required=True, help="Path to Keras model (.h5/.keras).")
    p.add_argument("--bins-performance-pkl", required=True, help="PKL with f1_score_val + bin_index")
    p.add_argument("--f1-per-fp-pkl", required=True, help="PKL with per-fingerprint F1 (keep >=0.85 by default)")
    p.add_argument("--fp-filter-pkl", default=None, help="Optional PKL listing fingerprint indices to keep (e.g., fp_filtered_4606.pkl)")
    p.add_argument("--db5-csv", required=True, help="DB5 CSV file")
    p.add_argument("--test-pkl", required=True, help="Single CASMI dataset pickle to evaluate (any year)")
    p.add_argument("--sirius-tsv", action="append", default=None,
                   help="Optional SIRIUS TSV(s). Repeat flag to pass multiple files.")
    p.add_argument("--ion-mode", choices=["positive", "negative"], default="negative")
    p.add_argument("--ppm", type=float, default=5.0)
    p.add_argument("--top-bins", type=int, default=500)
    p.add_argument("--out-dir", required=True, help="Output directory")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load DB5
    db5 = pd.read_csv(args.db5_csv)

    # Load test dataset
    test_df = pd.read_pickle(args.test_pkl)

    # Load 0..N SIRIUS TSVs and combine
    sirius_df = None
    if args.sirius_tsv:
        frames = [pd.read_csv(p, sep="\t") for p in args.sirius_tsv]
        if len(frames) == 1:
            sirius_df = build_sirius_lookup(frames[0])
        else:
            tsv = pd.concat(frames, ignore_index=True)
            sirius_df = build_sirius_lookup(tsv)

    # Out folder named after dataset stem
    ds_name = os.path.splitext(os.path.basename(args.test_pkl))[0]
    ds_out = os.path.join(args.out_dir, ds_name)
    os.makedirs(ds_out, exist_ok=True)

    # Run pipeline (now honors --top-bins)
    run_year(
        test_df, db5, sirius_df,
        args.bins_performance_pkl, args.f1_per_fp_pkl, args.model,
        args.ion_mode, args.ppm, ds_out,
        fp_filter_pkl=args.fp_filter_pkl,
        top_bins=args.top_bins,
    )

    print(f"Done. Outputs in: {ds_out}")

if __name__ == "__main__":
    main()
