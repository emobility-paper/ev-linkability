#!/usr/bin/env python3
"""Train pairwise linkability models and expose fitted scorers for retrieval.

This version preserves the original pooled analysis and adds two diagnostic
capabilities used by the retrieval robustness checks:

1. ``feature_cols`` may be supplied to fit a strict subset of an observer view,
   such as V2 without ``same_site``.
2. ``training_scope='same_site'`` restricts both training and validation to
   pairs whose two endpoints are at the same site.

The test set is not used by ``fit_pairwise_model``. Hyperparameters are selected
on the requested validation distribution only, while all imputation/scaling
statistics are fitted on the requested training distribution only.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import build_pairs as bp
from metrics import select_f1_threshold

SEED = 20260712

RF_GRID_DEFAULT = {
    "n_estimators": [300],
    "max_depth": [16, None],
    "min_samples_leaf": [2, 5],
    "max_features": ["sqrt"],
}
RF_GRID_FULL = {
    "n_estimators": [200, 500],
    "max_depth": [8, 16, None],
    "min_samples_leaf": [2, 5, 10],
    "max_features": ["sqrt", 0.5],
}

PAIR_SCOPES = ("pooled", "same_site")


def _f(v):
    return float(v) if v not in ("", "None", None) else np.nan


def _filter_rows(rows: List[dict], pair_scope: str) -> List[dict]:
    if pair_scope not in PAIR_SCOPES:
        raise ValueError(f"Unknown pair scope: {pair_scope}")
    if pair_scope == "pooled":
        return rows
    if rows and ("site_a" not in rows[0] or "site_b" not in rows[0]):
        raise ValueError(
            "same_site training requires site_a and site_b provenance columns "
            "in the pair CSV files"
        )
    return [r for r in rows if r["site_a"] == r["site_b"]]


def load_pairs(path: Path, feature_cols: List[str], pair_scope: str = "pooled"):
    """Load one pair partition, optionally retaining same-site pairs only."""
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows = _filter_rows(rows, pair_scope)
    if not rows:
        raise ValueError(f"No rows remain in {path} under pair_scope={pair_scope}")

    missing = [c for c in feature_cols if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing feature columns in {path.name}: {missing}")

    X = np.array([[_f(r[c]) for c in feature_cols] for r in rows], float)
    y = np.array([int(r["y"]) for r in rows], int)
    groups = np.array([r["user_hash_a"] for r in rows])
    return X, y, groups


def fit_impute(X_tr: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X_tr, axis=0)
    return np.where(np.isnan(med), 0.0, med)


def apply_impute(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    X = X.copy()
    bad = np.isnan(X)
    if bad.any():
        X[bad] = np.take(med, np.where(bad)[1])
    return X


def fit_scale(X_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _to_dissimilarity(X: np.ndarray, feat_names: List[str]) -> np.ndarray:
    X = X.copy()
    for j, name in enumerate(feat_names):
        if name.startswith("same_"):
            X[:, j] = 1.0 - X[:, j]
    return X


def distance_scores(X_imp: np.ndarray, feat_names: List[str], std: np.ndarray) -> np.ndarray:
    Xd = _to_dissimilarity(X_imp, feat_names) / std
    return np.exp(-Xd.sum(axis=1))


def _validate_feature_cols(observer: str, feature_cols: Optional[List[str]]) -> List[str]:
    available = list(bp.FEATURE_COLUMNS[observer])
    cols = available if feature_cols is None else list(feature_cols)
    if not cols:
        raise ValueError("At least one model feature is required")
    invalid = [c for c in cols if c not in available]
    if invalid:
        raise ValueError(
            f"Features {invalid} are not available under observer {observer}; "
            f"available={available}"
        )
    if len(cols) != len(set(cols)):
        raise ValueError("feature_cols contains duplicates")
    bp.pp.assert_no_feature_leakage(cols)
    return cols


def _check_binary(y: np.ndarray, partition: str, pair_scope: str) -> None:
    if np.unique(y).size < 2:
        raise ValueError(
            f"{partition} partition has fewer than two classes under "
            f"training_scope={pair_scope}"
        )


def train_observer(tag: str, observer: str, pairs_dir: Path, rng_seed: int, rf_grid: Dict):
    """Original pooled full-view training used for the main pairwise table."""
    feats = list(bp.FEATURE_COLUMNS[observer])
    Xtr, ytr, _ = load_pairs(pairs_dir / f"pairs_train_{tag}_{observer}.csv", feats)
    Xva, yva, _ = load_pairs(pairs_dir / f"pairs_val_{tag}_{observer}.csv", feats)
    Xte, yte, gte = load_pairs(pairs_dir / f"pairs_test_{tag}_{observer}.csv", feats)

    med = fit_impute(Xtr)
    Xtr_i, Xva_i, Xte_i = (apply_impute(X, med) for X in (Xtr, Xva, Xte))
    mean, std_s = fit_scale(Xtr_i)
    scale = lambda X: (X - mean) / std_s

    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    rng = np.random.default_rng(rng_seed)
    preds["random"] = (rng.random(len(yva)), rng.random(len(yte)))

    std_d = _to_dissimilarity(Xtr_i, feats).std(axis=0)
    std_d[std_d == 0] = 1.0
    preds["distance"] = (
        distance_scores(Xva_i, feats, std_d),
        distance_scores(Xte_i, feats, std_d),
    )

    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0, random_state=rng_seed
    )
    lr.fit(scale(Xtr_i), ytr)
    preds["logreg"] = (
        lr.predict_proba(scale(Xva_i))[:, 1],
        lr.predict_proba(scale(Xte_i))[:, 1],
    )

    best, best_auc, best_cfg = None, -1.0, None
    keys = list(rf_grid)
    for combo in itertools.product(*(rf_grid[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        rf = RandomForestClassifier(
            class_weight="balanced", random_state=rng_seed, n_jobs=-1, **cfg
        )
        rf.fit(Xtr_i, ytr)
        auc = roc_auc_score(yva, rf.predict_proba(Xva_i)[:, 1])
        if auc > best_auc:
            best, best_auc, best_cfg = rf, auc, cfg
    assert best is not None
    preds["rf"] = (best.predict_proba(Xva_i)[:, 1], best.predict_proba(Xte_i)[:, 1])
    print(f"  [{observer}] RF selected {best_cfg} (val ROC-AUC={best_auc:.3f})")

    return preds, yva, yte, gte


def fit_pairwise_model(
    tag: str,
    observer: str,
    pairs_dir: Path,
    seed: int = SEED,
    model: str = "rf",
    rf_grid: Optional[Dict] = None,
    feature_cols: Optional[List[str]] = None,
    training_scope: str = "pooled",
):
    """Fit one scorer for retrieval under a specified training distribution.

    Parameters
    ----------
    feature_cols:
        Subset of the selected observer's feature columns. Passing V2 columns
        without ``same_site`` implements the V2-minus-location diagnostic.
    training_scope:
        ``pooled`` uses the original pair distribution. ``same_site`` retains
        only rows where ``site_a == site_b`` in both train and validation.
    """
    if model not in ("rf", "logreg"):
        raise ValueError(f"Unsupported model: {model}")
    if training_scope not in PAIR_SCOPES:
        raise ValueError(f"Unsupported training_scope: {training_scope}")

    rf_grid = rf_grid or RF_GRID_DEFAULT
    feats = _validate_feature_cols(observer, feature_cols)

    Xtr, ytr, _ = load_pairs(
        pairs_dir / f"pairs_train_{tag}_{observer}.csv", feats, training_scope
    )
    Xva, yva, _ = load_pairs(
        pairs_dir / f"pairs_val_{tag}_{observer}.csv", feats, training_scope
    )
    _check_binary(ytr, "training", training_scope)
    _check_binary(yva, "validation", training_scope)

    med = fit_impute(Xtr)
    Xtr_i, Xva_i = apply_impute(Xtr, med), apply_impute(Xva, med)

    if model == "logreg":
        mean, std = fit_scale(Xtr_i)
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=seed
        )
        clf.fit((Xtr_i - mean) / std, ytr)
        val_auc = roc_auc_score(yva, clf.predict_proba((Xva_i - mean) / std)[:, 1])
        return {
            "clf": clf,
            "med": med,
            "mean": mean,
            "std": std,
            "observer": observer,
            "feature_cols": feats,
            "training_scope": training_scope,
            "n_train_pairs": int(len(ytr)),
            "n_val_pairs": int(len(yva)),
            "train_prevalence": float(ytr.mean()),
            "val_prevalence": float(yva.mean()),
            "val_auc": float(val_auc),
            "selected_config": {"C": 1.0},
            "feature_importances": None,
        }

    best, best_auc, best_cfg = None, -1.0, None
    keys = list(rf_grid)
    for combo in itertools.product(*(rf_grid[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        rf = RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1, **cfg
        )
        rf.fit(Xtr_i, ytr)
        auc = roc_auc_score(yva, rf.predict_proba(Xva_i)[:, 1])
        if auc > best_auc:
            best, best_auc, best_cfg = rf, auc, cfg

    if best is None:
        raise RuntimeError("Random-forest model selection produced no fitted model")

    importances = {
        name: float(value)
        for name, value in zip(feats, best.feature_importances_)
    }
    return {
        "clf": best,
        "med": med,
        "mean": None,
        "std": None,
        "observer": observer,
        "feature_cols": feats,
        "training_scope": training_scope,
        "n_train_pairs": int(len(ytr)),
        "n_val_pairs": int(len(yva)),
        "train_prevalence": float(ytr.mean()),
        "val_prevalence": float(yva.mean()),
        "val_auc": float(best_auc),
        "selected_config": best_cfg,
        "feature_importances": importances,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="min20_user_site")
    parser.add_argument("--observer", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    parser.add_argument("--pairs-dir", default=os.environ.get("PROCESSED_DIR", "data/processed"))
    parser.add_argument("--out-dir", default="outputs/preds")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rf-full", action="store_true")
    args = parser.parse_args(argv)

    pairs_dir = Path(args.pairs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rf_grid = RF_GRID_FULL if args.rf_full else RF_GRID_DEFAULT

    for observer in args.observer:
        print(f"Training observer {observer} (tag={args.tag})")
        preds, yva, yte, gte = train_observer(
            args.tag, observer, pairs_dir, args.seed, rf_grid
        )
        for model, (val_scores, test_scores) in preds.items():
            threshold = select_f1_threshold(val_scores, yva)
            out_path = out_dir / f"preds_{observer}_{model}.csv"
            with out_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["score", "y", "user_hash_a", "threshold"])
                for sc, yy, g in zip(test_scores, yte, gte):
                    w.writerow([sc, yy, g, threshold])
    print(f"Predictions written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
