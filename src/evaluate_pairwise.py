#!/usr/bin/env python3
"""Stage 4b: evaluate pairwise predictions into the Table 3 skeleton.

Reads outputs/preds/preds_{observer}_{model}.csv, computes ROC-AUC, PR-AUC, F1,
precision, recall and balanced accuracy, and attaches user-level bootstrap 95%
CIs to ROC-AUC and PR-AUC. Writes outputs/tables/table3_pairwise.csv.

Run (after train_pairwise):
    python src/evaluate_pairwise.py --n-boot 1000
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from metrics import point_metrics, user_level_bootstrap

_MODEL_ORDER = {"random": 0, "distance": 1, "logreg": 2, "rf": 3}
_FNAME = re.compile(r"preds_(?P<observer>v\d)_(?P<model>[a-z]+)\.csv")


def _load(path: Path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    scores = np.array([float(r["score"]) for r in rows])
    y = np.array([int(r["y"]) for r in rows], int)
    groups = np.array([r["user_hash_a"] for r in rows])
    threshold = float(rows[0]["threshold"])
    return scores, y, groups, threshold


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds-dir", default="outputs/preds")
    parser.add_argument("--out", default="outputs/tables/table3_pairwise.csv")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args(argv)

    preds_dir = Path(args.preds_dir)
    files = sorted(preds_dir.glob("preds_*.csv"))
    if not files:
        print(f"No prediction files in {preds_dir}/")
        return 1

    rows: List[Dict] = []
    for path in files:
        m = _FNAME.match(path.name)
        if not m:
            continue
        observer, model = m["observer"], m["model"]
        scores, y, groups, threshold = _load(path)
        pm = point_metrics(scores, y, threshold)
        roc_lo, roc_hi = user_level_bootstrap(scores, y, groups, "roc_auc", args.n_boot, args.seed)
        pr_lo, pr_hi = user_level_bootstrap(scores, y, groups, "pr_auc", args.n_boot, args.seed)
        rows.append({
            "observer": observer, "model": model,
            "n_test_pairs": len(y), "prevalence": round(float(y.mean()), 4),
            "roc_auc": round(pm["roc_auc"], 4),
            "roc_auc_lo": round(roc_lo, 4), "roc_auc_hi": round(roc_hi, 4),
            "pr_auc": round(pm["pr_auc"], 4),
            "pr_auc_lo": round(pr_lo, 4), "pr_auc_hi": round(pr_hi, 4),
            "f1": round(pm["f1"], 4), "precision": round(pm["precision"], 4),
            "recall": round(pm["recall"], 4), "balanced_acc": round(pm["balanced_acc"], 4),
        })

    rows.sort(key=lambda r: (r["observer"], _MODEL_ORDER.get(r["model"], 9)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nTable 3 (pairwise) -> {out}\n")
    print(f"{'obs':4} {'model':9} {'ROC-AUC [95% CI]':24} {'PR-AUC [95% CI]':24} {'F1':6}")
    for r in rows:
        roc = f"{r['roc_auc']:.3f} [{r['roc_auc_lo']:.3f},{r['roc_auc_hi']:.3f}]"
        pr = f"{r['pr_auc']:.3f} [{r['pr_auc_lo']:.3f},{r['pr_auc_hi']:.3f}]"
        print(f"{r['observer']:4} {r['model']:9} {roc:24} {pr:24} {r['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
