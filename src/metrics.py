#!/usr/bin/env python3
"""Metrics for pairwise linkability, with user-level (cluster) bootstrap.

The bootstrap unit is the IDENTITY, not the pair (section 22): a single user
contributes many correlated pairs, so resampling pairs would understate the
CI. We anchor every pair on user_hash_a during pair construction, so "the pairs
of identity u" = pairs with user_hash_a == u, and we cluster-resample those.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_METRIC_FNS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}


def select_f1_threshold(scores: np.ndarray, y: np.ndarray, n_grid: int = 201) -> float:
    """Threshold maximizing F1, chosen on VALIDATION scores only (section 16)."""
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_grid)))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def point_metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> Dict[str, float]:
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    pred = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
    }


def user_level_bootstrap(
    scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    metric: str = "roc_auc",
    n_boot: int = 1000,
    seed: int = 20260712,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Percentile CI for a metric under identity-level resampling with replacement."""
    fn = _METRIC_FNS[metric]
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    idx_by_group: Dict[object, np.ndarray] = {g: np.where(groups == g)[0] for g in uniq}

    stats: List[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in chosen])
        yb = y[idx]
        if np.unique(yb).size < 2:  # metric undefined on a one-class resample
            continue
        stats.append(float(fn(yb, scores[idx])))
    if len(stats) < 2:
        return float("nan"), float("nan")
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0
    lo, hi = np.percentile(stats, [lo_q, hi_q])
    return float(lo), float(hi)
