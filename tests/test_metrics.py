"""Metric-core tests: threshold selection, point metrics, cluster bootstrap."""
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import metrics as M  # noqa: E402


def test_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    pm = M.point_metrics(scores, y, threshold=0.5)
    assert pm["roc_auc"] == 1.0
    assert pm["pr_auc"] == 1.0


def test_f1_threshold_separates_clean_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = M.select_f1_threshold(scores, y)
    pred = (scores >= t).astype(int)
    # a threshold in the gap yields perfect F1
    assert np.array_equal(pred, y)


def test_random_scores_bracket_half():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    scores = rng.random(n)
    groups = rng.integers(0, 50, n)  # 50 identities
    lo, hi = M.user_level_bootstrap(scores, y, groups, "roc_auc", n_boot=300, seed=1)
    assert lo <= 0.5 <= hi
    assert lo < hi


def test_bootstrap_ci_orders_correctly():
    # strong signal -> CI sits well above 0.5
    rng = np.random.default_rng(2)
    n = 1500
    y = rng.integers(0, 2, n)
    scores = y + rng.normal(0, 0.3, n)  # signal-carrying
    groups = rng.integers(0, 40, n)
    lo, hi = M.user_level_bootstrap(scores, y, groups, "roc_auc", n_boot=300, seed=3)
    assert lo > 0.5
    assert lo <= hi


def test_pr_auc_metric_selectable():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.05, 0.95])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    lo, hi = M.user_level_bootstrap(scores, y, groups, "pr_auc", n_boot=200, seed=4)
    assert 0.0 <= lo <= hi <= 1.0
