# Candidate Selection and Ranking

Run **one testing set at a time**

## Install

```bash
pip install -e .  # installs deps, but you can also just `pip install -r requirements.txt`
```

## Usage
```
# example usage
PYTHONPATH=. python -m src.fp_runner.main \
  --model ../best_model_2015_4606_negative_remove_01_lstm.h5 \
  --bins-performance-pkl ../saved_bins_performance.p \
  --f1-per-fp-pkl ../f1_scores_per_fingerprint_4606.p \
  --fp-f1-threshold 0.9 \
  --fp-filter-pkl ../fp_filtered_4606.p \
  --db5-csv ../DB5_v1.2_new.csv \
  --test-pkl ../processed_test.csv \
  --ion-mode negative \
  --ppm 5 \
  --top-bins 500 \
  --out-dir runs/out
```


### What it does
- select top-N bins (by `f1_score_val`) from the saved performance pkl
- load the deep learning model with custom metrics (Tanimoto, Sokal–Sneath 3)
- select FP bit indices from per-fingerprint F1 pkl
- predict binary fingerprint bits for each spectrum (binned vector)
- retrieve DB5 candidates within ppm windows (adduct-adjusted precursor masses)
- write per-challenge candidates tables
