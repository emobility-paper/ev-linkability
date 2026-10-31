#!/usr/bin/env python3
"""Stage E4b: privacy-utility evaluation across mitigation profiles (sections 20-21).

For each profile P0-P4: recompute the FIXED pair set's features from mitigated
sessions, retrain the pairwise model (attacker adapts), and measure test ROC-AUC
/ PR-AUC (privacy risk). In parallel, measure operational utility loss on the
session data: daily-energy error, arrival-hour distribution shift, occupancy-curve
error, and site-utilization preservation (section 20).

The output feeds the privacy-utility Pareto (Figure 4): lower attack success at
lower utility loss dominates.

Run (after build_pairs):
    python src/evaluate_mitigation.py --tag min20_user_site --observer v2
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import mitigations as mit
from metrics import point_metrics, select_f1_threshold, user_level_bootstrap
from train_pairwise import _f, apply_impute, fit_impute

SEED = 20260712
EPS = 1e-9

_SESSION_KEYS = ["arrival_minute", "departure_minute", "weekday", "is_weekend",
                 "duration_min", "active_duration_min", "idle_duration_min",
                 "kWhDelivered", "connection_epoch"]


# --- loading -----------------------------------------------------------------
def load_session_arrays(path: Path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    sids = [r["sessionID"] for r in rows]
    cols: Dict[str, list] = {k: [] for k in _SESSION_KEYS}
    site = []
    for r in rows:
        for k in _SESSION_KEYS:
            cols[k].append(_f(r[k]) if k not in ("weekday", "is_weekend") else int(r[k]))
        site.append(r["site"])
    arr = {k: np.array(v, float) for k, v in cols.items()}
    arr["site"] = np.array(site, dtype=object)
    # avg_power for completeness (recomputed under mitigation anyway)
    act = arr["active_duration_min"]
    with np.errstate(divide="ignore", invalid="ignore"):
        arr["avg_power"] = np.where(act > 0, arr["kWhDelivered"] / (act / 60.0), np.nan)
    pos = {s: i for i, s in enumerate(sids)}
    return arr, pos


def load_pair_provenance(path: Path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    a = np.array([r["session_id_a"] for r in rows])
    b = np.array([r["session_id_b"] for r in rows])
    y = np.array([int(r["y"]) for r in rows], int)
    g = np.array([r["user_hash_a"] for r in rows])
    return a, b, y, g


def _gather(arrays: Dict[str, np.ndarray], positions: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[positions] for k, v in arrays.items()}


# --- utility metrics (section 20) -------------------------------------------
def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)

    def _kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / (y[mask] + EPS) + EPS)))

    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _hourly_occupancy(arr: np.ndarray, dep: np.ndarray) -> np.ndarray:
    occ = np.zeros(24)
    for h in range(24):
        t = h * 60
        same_day = (arr <= dep)
        active = np.where(same_day, (arr <= t) & (t < dep), (t >= arr) | (t < dep))
        occ[h] = active.sum()
    return occ


def utility_metrics(orig: Dict[str, np.ndarray], mitd: Dict[str, np.ndarray],
                    profile: str) -> Dict[str, float]:
    # daily total energy MAPE
    day = (orig["connection_epoch"] // 86400).astype(np.int64)
    days = np.unique(day)
    eo = np.array([orig["kWhDelivered"][day == d].sum() for d in days])
    em = np.array([mitd["kWhDelivered"][day == d].sum() for d in days])
    daily_mape = float(np.mean(np.abs(em - eo) / (eo + EPS)))

    # arrival-hour distribution
    ho, _ = np.histogram(orig["arrival_minute"] // 60, bins=np.arange(25))
    hm, _ = np.histogram(mitd["arrival_minute"] // 60, bins=np.arange(25))
    ho_n, hm_n = ho / (ho.sum() + EPS), hm / (hm.sum() + EPS)
    arrival_js = _js_divergence(ho_n, hm_n)
    peak_hour_match = int(np.argmax(ho) == np.argmax(hm))

    # occupancy curve
    occ_o = _hourly_occupancy(orig["arrival_minute"], orig["departure_minute"])
    occ_m = _hourly_occupancy(mitd["arrival_minute"], mitd["departure_minute"])
    occ_mae = float(np.mean(np.abs(occ_o - occ_m)) / (occ_o.mean() + EPS))
    peak_occ_shift = int(abs(int(np.argmax(occ_o)) - int(np.argmax(occ_m))))

    # site utilization: exact when location kept; undefined when removed
    if mit.PROFILES[profile]["location"] == "none":
        site_util_err = float("nan")  # not supported (location withheld)
    else:
        site_util_err = 0.0  # site labels unchanged under coarsening we can apply

    return {
        "daily_energy_mape": round(daily_mape, 4),
        "arrival_js": round(arrival_js, 4),
        "peak_hour_match": peak_hour_match,
        "occupancy_mae": round(occ_mae, 4),
        "peak_occ_shift_hours": peak_occ_shift,
        "site_util_err": site_util_err,
    }


# --- privacy: retrain the attacker under each profile ------------------------
def _rf(seed):
    return RandomForestClassifier(
        n_estimators=300, max_depth=16, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=-1)


def attack_under(arrays_mit, pos, prov, observer, seed):
    (a_tr, b_tr, y_tr, _), (a_va, b_va, y_va, _), (a_te, b_te, y_te, g_te) = prov

    def matrix(a_ids, b_ids):
        ap = np.array([pos[s] for s in a_ids])
        bp_ = np.array([pos[s] for s in b_ids])
        return mit.build_pair_matrix(_gather(arrays_mit, ap), _gather(arrays_mit, bp_), observer)

    Xtr, Xva, Xte = matrix(a_tr, b_tr), matrix(a_va, b_va), matrix(a_te, b_te)
    med = fit_impute(Xtr)
    Xtr, Xva, Xte = apply_impute(Xtr, med), apply_impute(Xva, med), apply_impute(Xte, med)

    clf = _rf(seed)
    clf.fit(Xtr, y_tr)
    va_scores = clf.predict_proba(Xva)[:, 1]
    threshold = select_f1_threshold(va_scores, y_va)
    te_scores = clf.predict_proba(Xte)[:, 1]
    pm = point_metrics(te_scores, y_te, threshold)
    return te_scores, y_te, g_te, pm


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="min20_user_site")
    parser.add_argument("--observer", default="v2", choices=["v1", "v2"])
    parser.add_argument("--features", default=os.environ.get("INTERIM_DIR", "data/interim") + "/eligible_sessions.csv")
    parser.add_argument("--pairs-dir", default=os.environ.get("PROCESSED_DIR", "data/processed"))
    parser.add_argument("--profiles", nargs="+", default=list(mit.PROFILES))
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    orig, pos = load_session_arrays(Path(args.features))
    pd = Path(args.pairs_dir)
    prov = tuple(load_pair_provenance(pd / f"pairs_{sp}_{args.tag}_{args.observer}.csv")
                 for sp in ("train", "val", "test"))

    rows: List[Dict] = []
    for profile in args.profiles:
        mitd = mit.mitigate_arrays(orig, profile)
        scores, y_te, g_te, pm = attack_under(mitd, pos, prov, args.observer, SEED)
        roc_lo, roc_hi = user_level_bootstrap(scores, y_te, g_te, "roc_auc", args.n_boot, SEED)
        util = utility_metrics(orig, mitd, profile)
        rows.append({
            "profile": profile, "observer": args.observer,
            "roc_auc": round(pm["roc_auc"], 4),
            "roc_auc_lo": round(roc_lo, 4), "roc_auc_hi": round(roc_hi, 4),
            "pr_auc": round(pm["pr_auc"], 4), **util,
        })

    out = Path(args.out) if args.out else Path(f"outputs/tables/mitigation_{args.tag}_{args.observer}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nPrivacy-utility ({args.observer}) -> {out}\n")
    print(f"{'prof':5} {'ROC-AUC [95% CI]':24} {'dailyE':7} {'arrJS':7} {'occMAE':7} {'pkShift':7}")
    for r in rows:
        roc = f"{r['roc_auc']:.3f} [{r['roc_auc_lo']:.3f},{r['roc_auc_hi']:.3f}]"
        print(f"{r['profile']:5} {roc:24} {r['daily_energy_mape']:<7.3f} "
              f"{r['arrival_js']:<7.3f} {r['occupancy_mae']:<7.3f} {r['peak_occ_shift_hours']:<7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
