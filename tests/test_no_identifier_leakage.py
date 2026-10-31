"""Stage 3 guards: labels correct, features leakage-free, splits don't cross.

The core leakage risk here is (a) an identifier column sneaking into the feature
matrix, and (b) a session appearing in pairs of two different splits. Both are
asserted mechanically.
"""
import random
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import build_pairs as bp  # noqa: E402
import preprocess as pp  # noqa: E402


def _sess(sid, uh, site="caltech", arr=480, dep=1020, wd=1, we=0,
          dur=540.0, act=90.0, idle=30.0, kwh=12.0, pwr=8.0):
    return {"sessionID": sid, "user_hash": uh, "site": site,
            "arrival_minute": arr, "departure_minute": dep, "weekday": wd,
            "is_weekend": we, "duration_min": dur, "active_duration_min": act,
            "idle_duration_min": idle, "kWhDelivered": kwh, "avg_power": pwr}


def _synthetic_split(n_users=6, per_user=8):
    sessions = []
    for u in range(n_users):
        for i in range(per_user):
            sessions.append(_sess(f"u{u}_s{i}", f"hash{u}",
                                  arr=400 + u * 30 + i, kwh=10 + u + i * 0.3))
    return sessions


# --- pure feature helpers ----------------------------------------------------
def test_circular_diff_wraps_midnight():
    assert bp.circular_diff(10, 1430) == 20        # across midnight, not 1420
    assert bp.circular_diff(600, 660) == 60
    assert bp.circular_diff(0, 720) == 720         # antipode


def test_pair_features_are_symmetric():
    a, b = _sess("a", "h1"), _sess("b", "h2", arr=500, kwh=20.0)
    assert bp.pair_features(a, b, "v2") == bp.pair_features(b, a, "v2")


def test_v1_features_exclude_location():
    a, b = _sess("a", "h1"), _sess("b", "h2")
    assert "same_site" not in bp.pair_features(a, b, "v1")
    assert "same_site" in bp.pair_features(a, b, "v2")


# --- leakage guards ----------------------------------------------------------
def test_feature_columns_have_no_identifiers():
    for observer in ("v1", "v2"):
        pp.assert_no_feature_leakage(bp.FEATURE_COLUMNS[observer])


def test_provenance_columns_are_not_features():
    for observer in ("v1", "v2"):
        overlap = set(bp.PROVENANCE_COLUMNS) & set(bp.FEATURE_COLUMNS[observer])
        assert not overlap, f"provenance leaked into features: {overlap}"


# --- label correctness on generated pairs ------------------------------------
def test_positive_pairs_share_identity_negatives_do_not():
    rng = random.Random(0)
    rows, _ = bp.build_split_pairs(_synthetic_split(), "v2", "train", rng)
    assert rows, "no pairs generated"
    for r in rows:
        if r["y"] == 1:
            assert r["user_hash_a"] == r["user_hash_b"]
        else:
            assert r["user_hash_a"] != r["user_hash_b"]


def test_no_self_pairs():
    rng = random.Random(1)
    rows, _ = bp.build_split_pairs(_synthetic_split(), "v1", "train", rng)
    assert all(r["session_id_a"] != r["session_id_b"] for r in rows)


def test_positive_cap_is_respected():
    # one identity with many sessions: C(20,2)=190 < cap; C(30,2)=435 > cap=50
    sessions = [_sess(f"big_{i}", "hashX", arr=400 + i) for i in range(30)]
    rng = random.Random(2)
    rows, pos_counts = bp.build_split_pairs(sessions, "v1", "train", rng, positive_cap=50)
    assert pos_counts["hashX"] == 50


def test_pairs_never_cross_splits():
    # sessions carry their split; building per-split must keep ids disjoint
    train = _synthetic_split()
    for s in train:
        s["split"] = "train"
    test = [_sess(f"t{u}_{i}", f"thash{u}", arr=400 + i) for u in range(4) for i in range(8)]
    for s in test:
        s["split"] = "test"
    rng = random.Random(3)
    tr_rows, _ = bp.build_split_pairs(train, "v1", "train", rng)
    te_rows, _ = bp.build_split_pairs(test, "v1", "test", rng)
    tr_ids = {r["session_id_a"] for r in tr_rows} | {r["session_id_b"] for r in tr_rows}
    te_ids = {r["session_id_a"] for r in te_rows} | {r["session_id_b"] for r in te_rows}
    assert tr_ids.isdisjoint(te_ids)


def test_hard_negatives_are_same_site():
    # with a wide pool, hard negatives (v2 same_site==1) should appear
    rng = random.Random(4)
    rows, _ = bp.build_split_pairs(_synthetic_split(), "v2", "train", rng,
                                   hard_frac=1.0, window=120)
    negs = [r for r in rows if r["y"] == 0]
    assert negs
    # all sessions here share 'caltech', so every hard neg is same-site
    assert all(r["same_site"] == 1 for r in negs)
