#!/usr/bin/env bash
#
# Reproducible end-to-end pipeline for the EV charging-session linkability study.
#
# The fixed seed (20260712) is baked into every stage; this script only sequences
# them and records a manifest. Run from the repository root:
#
#     ./run_all.sh                 # full main analysis (v1 + v2, E1-E4, figures)
#     RUN_SWEEP=1 ./run_all.sh     # also run the 10/15/30 min-session sweep (E1)
#     N_BOOT=200 ./run_all.sh      # faster bootstrap for a quick pass
#
# Requires the raw data in data/raw/<site>.json, or ACNDATA_API_TOKEN set so the
# download step can fetch it.
#
set -euo pipefail

# ---- configuration (override via environment) ----
SEED=20260712
MIN_SESSIONS="${MIN_SESSIONS:-20}"
GROUP_BY="${GROUP_BY:-user_site}"
TAG="min${MIN_SESSIONS}_${GROUP_BY}"
OBSERVERS=(v1 v2)
N_BOOT="${N_BOOT:-1000}"
ABLATION_NBOOT="${ABLATION_NBOOT:-300}"
RETRIEVAL_SEEDS="${RETRIEVAL_SEEDS:-100}"
SITES=(caltech jpl)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export RAW_DIR=data/raw
export INTERIM_DIR=data/interim
export PROCESSED_DIR=data/processed
mkdir -p "$RAW_DIR" "$INTERIM_DIR" "$PROCESSED_DIR" outputs/tables outputs/figures outputs/logs

LOG=outputs/logs/run_manifest.txt
stamp() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

stamp "=== run_all start (seed=$SEED tag=$TAG n_boot=$N_BOOT) ==="
python3 -c "import sys,sklearn,numpy,matplotlib;print('versions: py',sys.version.split()[0],'sklearn',sklearn.__version__,'numpy',numpy.__version__)" | tee -a "$LOG" || true

# ---- 0. code must be green before producing any numbers ----
stamp "test suite"
python3 -m pytest tests/ -q

# ---- 1. raw data (download only if missing) ----
need_dl=0
for s in "${SITES[@]}"; do [ -f "$RAW_DIR/$s.json" ] || need_dl=1; done
if [ "$need_dl" -eq 1 ]; then
  if [ -n "${ACNDATA_API_TOKEN:-}" ]; then
    stamp "download raw sessions"
    python3 src/download_acndata.py --sites "${SITES[@]}" --out-dir "$RAW_DIR"
  else
    stamp "ERROR: raw data missing and ACNDATA_API_TOKEN unset."
    exit 1
  fi
else
  stamp "raw data present; skipping download"
fi

# ---- 2. audit + 3. preprocess ----
stamp "audit";      python3 src/audit_data.py       | tee -a "$LOG"
stamp "preprocess"; python3 src/preprocess.py       | tee -a "$LOG"

# ---- 4. leakage-safe split (primary cohort) ----
stamp "split $TAG"
python3 src/split_sessions.py --min-sessions "$MIN_SESSIONS" --group-by "$GROUP_BY" | tee -a "$LOG"
SPLITS="$PROCESSED_DIR/splits_${TAG}.csv"

# ---- 5. pairs (both observers) ----
for obs in "${OBSERVERS[@]}"; do
  stamp "build_pairs $obs"
  python3 src/build_pairs.py --splits "$SPLITS" --observer "$obs" | tee -a "$LOG"
done

# ---- 6. E1 pairwise ----
stamp "train_pairwise"
python3 src/train_pairwise.py --tag "$TAG" --observer "${OBSERVERS[@]}" | tee -a "$LOG"
stamp "evaluate_pairwise"
python3 src/evaluate_pairwise.py --n-boot "$N_BOOT" | tee -a "$LOG"

# ---- 7. E3 ablation ----
for obs in "${OBSERVERS[@]}"; do
  stamp "ablation $obs"
  python3 src/ablation.py --tag "$TAG" --observer "$obs" --n-boot "$ABLATION_NBOOT" | tee -a "$LOG"
done

# ---- 8. E2 retrieval ----
stamp "build_retrieval_sets"
python3 src/build_retrieval_sets.py --splits "$SPLITS" | tee -a "$LOG"
IDX="$PROCESSED_DIR/retrieval_index_${TAG}.json"
for obs in "${OBSERVERS[@]}"; do
  stamp "evaluate_retrieval $obs"
  python3 src/evaluate_retrieval.py --index "$IDX" --tag "$TAG" --observer "$obs" --seeds "$RETRIEVAL_SEEDS" | tee -a "$LOG"
done

# ---- 9. E4 mitigation + utility ----
for obs in "${OBSERVERS[@]}"; do
  stamp "evaluate_mitigation $obs"
  python3 src/evaluate_mitigation.py --tag "$TAG" --observer "$obs" --n-boot "$ABLATION_NBOOT" | tee -a "$LOG"
done

# ---- 10. figures ----
stamp "make_figures"
python3 src/make_figures.py | tee -a "$LOG"

# ---- 11. optional selection-bias sweep (E1 across thresholds) ----
if [ "${RUN_SWEEP:-0}" = "1" ]; then
  for ms in 10 15 30; do
    swtag="min${ms}_${GROUP_BY}"
    stamp "SWEEP split ms=$ms"
    python3 src/split_sessions.py --min-sessions "$ms" --group-by "$GROUP_BY" | tee -a "$LOG"
    for obs in "${OBSERVERS[@]}"; do
      python3 src/build_pairs.py --splits "$PROCESSED_DIR/splits_${swtag}.csv" --observer "$obs" | tee -a "$LOG"
    done
    python3 src/train_pairwise.py --tag "$swtag" --observer "${OBSERVERS[@]}" --out-dir "outputs/preds_${swtag}" | tee -a "$LOG"
    python3 src/evaluate_pairwise.py --preds-dir "outputs/preds_${swtag}" \
        --out "outputs/tables/table3_pairwise_${swtag}.csv" --n-boot "$ABLATION_NBOOT" | tee -a "$LOG"
  done
fi

stamp "=== run_all complete: tables in outputs/tables/, figures in outputs/figures/ ==="
