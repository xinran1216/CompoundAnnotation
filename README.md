# Candidate Selection and Ranking

Run **one testing set at a time**

## Usage
```bash
pip install -e .  # installs deps, but you can also just `pip install -r requirements.txt`

# from repo root
python -m fp_runner.main       --model /path/to/model.h5       --bins-performance-pkl saved_bins_performance_positive.p       --f1-per-fp-pkl f1_scores_per_fingerprint_4606_positive.p       --db5-csv DB5_v1.2_new.csv       --test-pkl /path/to/casmi_remove_01_2010_4606_pos.p       --sirius-tsv /path/to/sirius.tsv       --ion-mode positive       --ppm 5       --top-bins 500       --out-dir runs/out
```

> If no sirius-tsv is provided, scoring uses Tanimoto only.

### What it does
- select top-N bins (by `f1_score_val`) from the saved performance pkl
- load the deep learning model with custom metrics (Tanimoto, Sokal–Sneath 3)
- select FP bit indices from per-fingerprint F1 pkl
- predict binary fingerprint bits for each spectrum (binned vector)
- retrieve DB5 candidates within ppm windows (adduct-adjusted precursor masses)
- combine scores (0.8×Tanimoto + 0.2×normalized SIRIUS per challenge)
- write per-challenge candidates tables
