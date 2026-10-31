#!/usr/bin/env python3
"""Candidate-set retrieval with pooled/site-matched diagnostics.

This version supports three orthogonal choices:

* candidate scope: pooled or same-site distractors;
* feature profile: native observer features or V2 without ``same_site``;
* training scope: pooled train/validation pairs or same-site-only pairs.

These options implement the two diagnostics needed to explain the V1/V2
inversion under site matching:

1. V2-minus-location:
   ``--observer v2 --feature-profile v2_minus_site --training-scope pooled
   --candidate-scope same_site``
2. Site-adapted retrieval:
   ``--training-scope same_site --candidate-scope same_site`` for V1 and V2.

References and candidate draws use the same deterministic seeds across variants,
so the diagnostic runs differ only in the fitted scorer and requested scope.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

import build_pairs as bp
from train_pairwise import _f, apply_impute, fit_pairwise_model

SEED = 20260712
FEATURE_PROFILES = ("native", "v2_minus_site")


def circ_vec(a: float, b: np.ndarray, period: int = 1440) -> np.ndarray:
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def rank_of_positive(
    pos_score: float,
    distractor_scores: np.ndarray,
    rng: np.random.Generator,
    noise: float = 1e-9,
) -> int:
    """Return the 1-based rank of the positive, breaking exact ties randomly."""
    scores = np.concatenate([[pos_score], distractor_scores])
    scores = scores + rng.normal(0.0, noise, size=scores.size)
    order = np.argsort(-scores)
    return int(np.where(order == 0)[0][0]) + 1


def retrieval_metrics(ranks: np.ndarray) -> Dict[str, float]:
    ranks = np.asarray(ranks, float)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def _stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def resolve_feature_cols(observer: str, feature_profile: str) -> List[str]:
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(f"Unknown feature profile: {feature_profile}")
    if feature_profile == "native":
        return list(bp.FEATURE_COLUMNS[observer])
    if observer != "v2":
        raise ValueError("v2_minus_site is only valid with --observer v2")
    return [c for c in bp.FEATURE_COLUMNS["v2"] if c != "same_site"]


class RetrievalScorer:
    """Score reference sessions against a candidate pool with a fitted model."""

    def __init__(self, feature_cols, clf, med, mean, std):
        self.feats = list(feature_cols)
        self.clf, self.med, self.mean, self.std = clf, med, mean, std

    def _feature(self, name, r, pool):
        if name == "circ_arrival_diff":
            return circ_vec(r["arrival_minute"], pool["arrival_minute"])
        if name == "circ_departure_diff":
            return circ_vec(r["departure_minute"], pool["departure_minute"])
        if name == "abs_duration_diff":
            return np.abs(r["duration_min"] - pool["duration_min"])
        if name == "abs_energy_diff":
            return np.abs(r["kWhDelivered"] - pool["kWhDelivered"])
        if name == "same_weekday":
            return (pool["weekday"] == r["weekday"]).astype(float)
        if name == "same_weekend":
            return (pool["is_weekend"] == r["is_weekend"]).astype(float)
        if name == "same_site":
            return (pool["site"] == r["site"]).astype(float)
        if name == "abs_active_duration_diff":
            return np.abs(r["active_duration_min"] - pool["active_duration_min"])
        if name == "abs_idle_duration_diff":
            return np.abs(r["idle_duration_min"] - pool["idle_duration_min"])
        if name == "abs_avg_power_diff":
            return np.abs(r["avg_power"] - pool["avg_power"])
        raise KeyError(name)

    def profile_scores(self, refs: List[Dict], pool: Dict[str, np.ndarray]) -> np.ndarray:
        total = np.zeros(len(pool["arrival_minute"]))
        for r in refs:
            X = np.column_stack([self._feature(c, r, pool) for c in self.feats])
            X = apply_impute(X, self.med)
            if self.mean is not None:
                X = (X - self.mean) / self.std
            total += self.clf.predict_proba(X)[:, 1]
        return total / len(refs)


def load_session_features(path: Path) -> Dict[str, Dict]:
    feats: Dict[str, Dict] = {}
    for r in csv.DictReader(path.open(newline="", encoding="utf-8")):
        active = _f(r["active_duration_min"])
        energy = _f(r["kWhDelivered"])
        avg_power = (
            energy / (active / 60.0)
            if (active and active > 0 and not np.isnan(energy))
            else np.nan
        )
        feats[r["sessionID"]] = {
            "site": r["site"],
            "arrival_minute": int(r["arrival_minute"]),
            "departure_minute": int(r["departure_minute"]),
            "weekday": int(r["weekday"]),
            "is_weekend": int(r["is_weekend"]),
            "duration_min": _f(r["duration_min"]),
            "active_duration_min": active,
            "idle_duration_min": _f(r["idle_duration_min"]),
            "kWhDelivered": energy,
            "avg_power": avg_power,
        }
    return feats


def build_pool_arrays(test_pool, feats) -> Dict[str, np.ndarray]:
    keys = [
        "arrival_minute",
        "departure_minute",
        "weekday",
        "is_weekend",
        "duration_min",
        "active_duration_min",
        "idle_duration_min",
        "kWhDelivered",
        "avg_power",
    ]
    cols = {k: [] for k in keys}
    site, uh = [], []
    for sid, u in test_pool:
        f = feats[sid]
        for k in keys:
            cols[k].append(f[k])
        site.append(f["site"])
        uh.append(u)
    arr = {k: np.array(v, float) for k, v in cols.items()}
    arr["site"] = np.array(site, dtype=object)
    arr["user_hash"] = np.array(uh, dtype=object)
    return arr


def _bootstrap_targets(
    per_target: Dict[str, List[int]], metric: str, n_boot: int, seed: int
):
    rng = np.random.default_rng(seed)
    uhs = list(per_target)
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(uhs, size=len(uhs), replace=True)
        ranks = np.concatenate([per_target[u] for u in chosen])
        vals.append(retrieval_metrics(ranks)[metric])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_model_info(path: Path, fitted: Dict, feature_profile: str) -> None:
    info = {
        "observer": fitted["observer"],
        "feature_profile": feature_profile,
        "training_scope": fitted["training_scope"],
        "feature_cols": fitted["feature_cols"],
        "n_train_pairs": fitted["n_train_pairs"],
        "n_val_pairs": fitted["n_val_pairs"],
        "train_prevalence": fitted["train_prevalence"],
        "val_prevalence": fitted["val_prevalence"],
        "validation_roc_auc": fitted["val_auc"],
        "selected_config": fitted["selected_config"],
        "feature_importances": fitted["feature_importances"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(info), indent=2), encoding="utf-8")


def default_output_name(tag, observer, feature_profile, training_scope, candidate_scope):
    # Preserve the original filenames for the two established analyses.
    if feature_profile == "native" and training_scope == "pooled":
        suffix = "" if candidate_scope == "pooled" else "_same_site"
        return Path(f"outputs/tables/retrieval_{tag}_{observer}{suffix}.csv")
    return Path(
        "outputs/tables/"
        f"retrieval_{tag}_{observer}_{feature_profile}_"
        f"train-{training_scope}_cand-{candidate_scope}.csv"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--tag", default="min20_user_site")
    parser.add_argument("--observer", default="v2", choices=["v1", "v2"])
    parser.add_argument("--model", default="rf", choices=["rf", "logreg"])
    parser.add_argument(
        "--feature-profile",
        default="native",
        choices=FEATURE_PROFILES,
        help="native observer fields, or V2 with same_site removed",
    )
    parser.add_argument(
        "--training-scope",
        default="pooled",
        choices=["pooled", "same_site"],
        help="Pair distribution used for model fitting and validation selection",
    )
    parser.add_argument(
        "--candidate-scope",
        default="pooled",
        choices=["pooled", "same_site"],
        help="Distractor population used in candidate-set retrieval",
    )
    parser.add_argument(
        "--features",
        default=os.environ.get("INTERIM_DIR", "data/interim")
        + "/eligible_sessions.csv",
    )
    parser.add_argument(
        "--pairs-dir", default=os.environ.get("PROCESSED_DIR", "data/processed")
    )
    parser.add_argument("--K", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--m", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--model-info-out",
        default=None,
        help="Optional JSON file with validation AUC, pair counts, and RF importances",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit/inspect the pairwise scorer and stop before retrieval",
    )
    args = parser.parse_args(argv)

    feature_cols = resolve_feature_cols(args.observer, args.feature_profile)

    index = json.loads(Path(args.index).read_text())
    targets = index["targets"]
    test_pool = index["test_pool"]
    feats = load_session_features(Path(args.features))
    pool = build_pool_arrays(test_pool, feats)
    pool_uh = pool["user_hash"]
    pool_site = pool["site"]

    fitted = fit_pairwise_model(
        args.tag,
        args.observer,
        Path(args.pairs_dir),
        model=args.model,
        feature_cols=feature_cols,
        training_scope=args.training_scope,
    )
    scorer = RetrievalScorer(
        fitted["feature_cols"],
        fitted["clf"],
        fitted["med"],
        fitted["mean"],
        fitted["std"],
    )

    print(
        f"Model: observer={args.observer}, feature_profile={args.feature_profile}, "
        f"training_scope={args.training_scope}, val_AUC={fitted['val_auc']:.4f}, "
        f"train_pairs={fitted['n_train_pairs']}, val_pairs={fitted['n_val_pairs']}"
    )
    if fitted["feature_importances"]:
        ordered = sorted(
            fitted["feature_importances"].items(), key=lambda kv: kv[1], reverse=True
        )
        print("RF feature importances:")
        for name, value in ordered:
            print(f"  {name:30s} {value:.4f}")

    if args.model_info_out:
        write_model_info(Path(args.model_info_out), fitted, args.feature_profile)

    if args.fit_only:
        print("Fit-only mode: model inspection complete; retrieval was not run.")
        return 0

    rows = []
    for m in args.m:
        profiles: Dict[str, np.ndarray] = {}
        pos_positions: Dict[str, np.ndarray] = {}
        for uh, entry in targets.items():
            if len(entry["train"]) < m:
                continue
            rng_ref = np.random.default_rng(_stable_seed("ref", uh, m))
            ref_sids = rng_ref.choice(entry["train"], size=m, replace=False)
            refs = [feats[s] for s in ref_sids]
            profiles[uh] = scorer.profile_scores(refs, pool)
            pos_positions[uh] = np.where(pool_uh == uh)[0]

        for K in args.K:
            per_target: Dict[str, List[int]] = {}
            for uh, prof in profiles.items():
                positives = pos_positions[uh]
                if len(positives) == 0:
                    continue

                eligible_positives = []
                distractors_by_pos = {}
                for pos in positives:
                    mask = pool_uh != uh
                    if args.candidate_scope == "same_site":
                        mask = mask & (pool_site == pool_site[pos])
                    distractors = np.where(mask)[0]
                    if len(distractors) >= K - 1:
                        pos_i = int(pos)
                        eligible_positives.append(pos_i)
                        distractors_by_pos[pos_i] = distractors
                if not eligible_positives:
                    continue

                # Deliberately identical across model variants: enables a clean
                # comparison on the same reference and candidate draws.
                rng = np.random.default_rng(_stable_seed("trial", uh, m, K))
                ranks = []
                for _ in range(args.seeds):
                    pos = int(rng.choice(eligible_positives))
                    distractors = distractors_by_pos[pos]
                    dsel = rng.choice(distractors, size=K - 1, replace=False)
                    ranks.append(rank_of_positive(prof[pos], prof[dsel], rng))
                per_target[uh] = np.array(ranks)

            if not per_target:
                print(
                    f"[warn] m={m}, K={K}: no target has enough distractors; skipping"
                )
                continue

            all_ranks = np.concatenate(list(per_target.values()))
            pm = retrieval_metrics(all_ranks)
            t1_lo, t1_hi = _bootstrap_targets(per_target, "top1", args.n_boot, SEED)
            mrr_lo, mrr_hi = _bootstrap_targets(per_target, "mrr", args.n_boot, SEED)
            rows.append(
                {
                    "observer": args.observer,
                    "model": args.model,
                    "feature_profile": args.feature_profile,
                    "training_scope": args.training_scope,
                    "candidate_scope": args.candidate_scope,
                    "m": m,
                    "K": K,
                    "n_targets": len(per_target),
                    "n_trials": len(all_ranks),
                    "top1": round(pm["top1"], 4),
                    "top1_lo": round(t1_lo, 4),
                    "top1_hi": round(t1_hi, 4),
                    "top3": round(pm["top3"], 4),
                    "top5": round(pm["top5"], 4),
                    "mrr": round(pm["mrr"], 4),
                    "mrr_lo": round(mrr_lo, 4),
                    "mrr_hi": round(mrr_hi, 4),
                    "median_rank": pm["median_rank"],
                    "random_top1": round(1.0 / K, 4),
                    "random_top3": round(min(3.0 / K, 1.0), 4),
                    "random_top5": round(min(5.0 / K, 1.0), 4),
                    "random_mrr": round(
                        sum(1.0 / j for j in range(1, K + 1)) / K, 4
                    ),
                }
            )

    if not rows:
        raise RuntimeError("Retrieval produced no output rows")

    out = (
        Path(args.out)
        if args.out
        else default_output_name(
            args.tag,
            args.observer,
            args.feature_profile,
            args.training_scope,
            args.candidate_scope,
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(
        f"\nRetrieval ({args.observer}, {args.model}, {args.feature_profile}, "
        f"train={args.training_scope}, candidates={args.candidate_scope}) -> {out}\n"
    )
    print(
        f"{'m':>2} {'K':>4} {'Top-1 [95% CI]':22} {'Top-5':7} "
        f"{'MRR [95% CI]':22} {'rand Top-1'}"
    )
    for r in rows:
        t1 = f"{r['top1']:.3f} [{r['top1_lo']:.3f},{r['top1_hi']:.3f}]"
        mrr = f"{r['mrr']:.3f} [{r['mrr_lo']:.3f},{r['mrr_hi']:.3f}]"
        print(
            f"{r['m']:>2} {r['K']:>4} {t1:22} {r['top5']:.3f}   "
            f"{mrr:22} {r['random_top1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
