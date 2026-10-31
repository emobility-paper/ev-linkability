#!/usr/bin/env python3
"""Stage E3: feature-group ablation with per-site evaluation (section 18).

Two questions, two analyses:
  * single-group (A1-A4) and combinations (A5-A10): which behavior links ALONE?
  * leave-one-group-out (all_minus_*): which group, removed from the full model,
    costs the most? (RQ4)

Hyperparameters are held FIXED across ablations -- the whole point is to vary
only the feature set, not also re-tune. Each ablation is evaluated pooled and
per-site. The per-site cut keeps only pairs whose BOTH sessions are at that site,
which also makes same_site vacuous -- so the pooled-vs-per-site gap for the full
model quantifies how much of v2's edge is the trivial cross-site same_site
signal versus genuine behavior.

Run (after build_pairs; v2 is the interesting observer):
    python src/ablation.py --tag min20_user_site --observer v2 --n-boot 300
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import build_pairs as bp
from metrics import point_metrics, select_f1_threshold, user_level_bootstrap
from train_pairwise import _f, apply_impute, fit_impute, fit_scale

SEED = 20260712

# Behavioral grouping of the pair features (section 11 taxonomy).
GROUPS: Dict[str, List[str]] = {
    "time": ["circ_arrival_diff", "circ_departure_diff", "same_weekday", "same_weekend"],
    "energy": ["abs_energy_diff", "abs_avg_power_diff"],
    "duration": ["abs_duration_diff", "abs_active_duration_diff", "abs_idle_duration_diff"],
    "location": ["same_site"],
}

ABLATIONS: Dict[str, List[str]] = {
    "A1_time": ["time"],
    "A2_energy": ["energy"],
    "A3_duration": ["duration"],
    "A4_location": ["location"],
    "A5_time_energy": ["time", "energy"],
    "A6_time_duration": ["time", "duration"],
    "A7_time_location": ["time", "location"],
    "A8_energy_location": ["energy", "location"],
    "A9_time_energy_duration": ["time", "energy", "duration"],
    "A10_all": ["time", "energy", "duration", "location"],
    "LOGO_minus_time": ["energy", "duration", "location"],
    "LOGO_minus_energy": ["time", "duration", "location"],
    "LOGO_minus_duration": ["time", "energy", "location"],
    "LOGO_minus_location": ["time", "energy", "duration"],
}


def resolve_features(observer: str, group_names: List[str]) -> List[str]:
    """Feature columns for these groups, intersected with the observer's set."""
    avail = set(bp.FEATURE_COLUMNS[observer])
    cols: List[str] = []
    for g in group_names:
        for c in GROUPS[g]:
            if c in avail and c not in cols:
                cols.append(c)
    return cols


def load_full(path: Path, feat_names: List[str]):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    X = np.array([[_f(r[c]) for c in feat_names] for r in rows], float)
    y = np.array([int(r["y"]) for r in rows], int)
    groups = np.array([r["user_hash_a"] for r in rows])
    site_a = np.array([r["site_a"] for r in rows])
    site_b = np.array([r["site_b"] for r in rows])
    return X, y, groups, site_a, site_b


def _make_model(model: str, seed: int):
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="min20_user_site")
    parser.add_argument("--observer", default="v2", choices=["v1", "v2"])
    parser.add_argument("--model", default="rf", choices=["rf", "logreg"])
    parser.add_argument("--pairs-dir", default=os.environ.get("PROCESSED_DIR", "data/processed"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    feats = bp.FEATURE_COLUMNS[args.observer]
    idx = {c: i for i, c in enumerate(feats)}
    pd = Path(args.pairs_dir)
    Xtr, ytr, _, _, _ = load_full(pd / f"pairs_train_{args.tag}_{args.observer}.csv", feats)
    Xva, yva, _, _, _ = load_full(pd / f"pairs_val_{args.tag}_{args.observer}.csv", feats)
    Xte, yte, gte, sa, sb = load_full(pd / f"pairs_test_{args.tag}_{args.observer}.csv", feats)

    med = fit_impute(Xtr)
    Xtr_i, Xva_i, Xte_i = (apply_impute(X, med) for X in (Xtr, Xva, Xte))

    sites = sorted(set(sa) | set(sb))
    cuts = {"pooled": np.ones(len(yte), bool)}
    for s in sites:
        cuts[s] = (sa == s) & (sb == s)  # both endpoints at site s -> same_site vacuous

    rows: List[Dict] = []
    for ab_id, group_names in ABLATIONS.items():
        cols = resolve_features(args.observer, group_names)
        if not cols:
            continue  # e.g. location-only under v1
        bp.pp.assert_no_feature_leakage(cols)  # guard every subset
        ci = [idx[c] for c in cols]

        needs_scale = args.model == "logreg"
        if needs_scale:
            mean, std = fit_scale(Xtr_i[:, ci])
            tf = lambda X: (X[:, ci] - mean) / std
        else:
            tf = lambda X: X[:, ci]

        clf = _make_model(args.model, args.seed)
        clf.fit(tf(Xtr_i), ytr)
        val_scores = clf.predict_proba(tf(Xva_i))[:, 1]
        threshold = select_f1_threshold(val_scores, yva)
        test_scores = clf.predict_proba(tf(Xte_i))[:, 1]

        for cut_name, mask in cuts.items():
            m = mask
            if m.sum() < 50 or np.unique(yte[m]).size < 2:
                continue
            pm = point_metrics(test_scores[m], yte[m], threshold)
            roc_lo, roc_hi = user_level_bootstrap(
                test_scores[m], yte[m], gte[m], "roc_auc", args.n_boot, args.seed)
            pr_lo, pr_hi = user_level_bootstrap(
                test_scores[m], yte[m], gte[m], "pr_auc", args.n_boot, args.seed)
            rows.append({
                "ablation": ab_id, "groups": "+".join(group_names),
                "observer": args.observer, "model": args.model, "site": cut_name,
                "n_features": len(cols), "n_pairs": int(m.sum()),
                "roc_auc": round(pm["roc_auc"], 4),
                "roc_auc_lo": round(roc_lo, 4), "roc_auc_hi": round(roc_hi, 4),
                "pr_auc": round(pm["pr_auc"], 4),
                "pr_auc_lo": round(pr_lo, 4), "pr_auc_hi": round(pr_hi, 4),
                "f1": round(pm["f1"], 4),
            })

    out = Path(args.out) if args.out else Path(f"outputs/tables/ablation_{args.observer}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nAblation ({args.observer}, {args.model}) -> {out}\n")
    print(f"{'ablation':26} {'pooled':16} {'caltech':16} {'jpl':16}")
    by_ab: Dict[str, Dict[str, str]] = {}
    for r in rows:
        by_ab.setdefault(r["ablation"], {})[r["site"]] = f"{r['roc_auc']:.3f}"
    for ab_id in ABLATIONS:
        if ab_id not in by_ab:
            continue
        c = by_ab[ab_id]
        print(f"{ab_id:26} {c.get('pooled','-'):16} {c.get('caltech','-'):16} {c.get('jpl','-'):16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
