"""Retrieval-core tests: ranking, circular diff parity, Top-k/MRR behavior."""
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import build_pairs as bp  # noqa: E402
import evaluate_retrieval as er  # noqa: E402


def test_circ_vec_matches_scalar_circular_diff():
    rng = np.random.default_rng(0)
    a = 137
    b = rng.integers(0, 1440, 50)
    vec = er.circ_vec(a, b)
    scalar = np.array([bp.circular_diff(a, int(x)) for x in b])
    assert np.allclose(vec, scalar)


def test_rank_positive_when_highest_is_one():
    rng = np.random.default_rng(1)
    r = er.rank_of_positive(0.99, np.array([0.1, 0.2, 0.3]), rng)
    assert r == 1


def test_rank_positive_when_lowest_is_k():
    rng = np.random.default_rng(2)
    distractors = np.array([0.9, 0.8, 0.7, 0.6])
    r = er.rank_of_positive(0.01, distractors, rng)
    assert r == len(distractors) + 1  # 5


def test_perfect_scores_give_top1_one():
    # positive always strictly highest -> Top-1 = 1.0
    rng = np.random.default_rng(3)
    ranks = [er.rank_of_positive(1.0, rng.random(9) * 0.5, rng) for _ in range(200)]
    m = er.retrieval_metrics(np.array(ranks))
    assert m["top1"] == 1.0
    assert m["mrr"] == 1.0


def test_random_scores_bracket_uniform_top1():
    # positive drawn from same distribution as K-1=9 distractors -> Top-1 ~ 1/10
    rng = np.random.default_rng(4)
    ranks = []
    for _ in range(20000):
        ranks.append(er.rank_of_positive(rng.random(), rng.random(9), rng))
    m = er.retrieval_metrics(np.array(ranks))
    assert 0.08 < m["top1"] < 0.12  # ~0.10


def test_metrics_monotone_in_k_semantics():
    ranks = np.array([1, 2, 3, 4, 5, 6])
    m = er.retrieval_metrics(ranks)
    assert m["top1"] <= m["top3"] <= m["top5"]
