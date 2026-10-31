# Behavioral Linkability of EV Charging Sessions — Analysis Code

Code accompanying the paper:

> "Behavioral Linkability of Electric Vehicle Charging Sessions
> in ISO 15118-Enabled Ecosystems: A Privacy–Utility Evaluation"
> (under double-anonymized review; author details withheld).

This repository contains the complete, leakage-controlled evaluation pipeline:
data acquisition, preprocessing, per-user chronological splitting, pair
construction with hard negatives, pairwise verification models, candidate-set
retrieval, feature ablation, metadata-generalization (mitigation) evaluation
with operational-utility metrics, and figure generation. An automated test
suite enforces the leakage guarantees (temporal split, train/test pair
disjointness, identifier exclusion from the feature matrix).

## What is (deliberately) NOT included

No charging records are redistributed. In line with the ACN-Data distribution
terms and to reduce privacy risk, this package excludes `data/` (raw and
processed records), the ground-truth tables, the HMAC salt, and all
prediction outputs. The pipeline re-creates everything from a fresh download.

## Requirements

- Python ≥ 3.8 (the paper's results were produced with Python 3.8.8)
- numpy 1.24.4, scikit-learn 1.3.2, matplotlib, pytz, pytest
- acnportal 0.3.3 (only for the download step)

```
pip install -r requirements.txt
```

## Reproducing the study

1. Request a free ACN-Data API token at https://ev.caltech.edu/dataset
2. `export ACNDATA_API_TOKEN="..."`
3. `./run_all.sh` — runs the test suite, downloads the Caltech and JPL sites
   if absent, and executes the full pipeline (experiments E1–E4, tables,
   figures). `RUN_SWEEP=1 ./run_all.sh` additionally runs the
   minimum-session-threshold sensitivity sweep.

Every stage is governed by the fixed seed 20260712; a timestamped run manifest
is written to `outputs/logs/run_manifest.txt`. Tables land in
`outputs/tables/`, figures in `outputs/figures/`.

## Layout

```
run_all.sh              one-command orchestration of the full study
src/                    pipeline stages (see run_all.sh for the exact order)
tests/                  leakage and correctness tests (pytest)
```

## License

MIT — see `LICENSE`.
