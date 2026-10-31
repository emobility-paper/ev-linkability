"""Temporal-split correctness: floor proportions, chronology, cohort filter.

These run offline on synthetic in-memory records -- no pipeline run required.
"""
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import split_sessions as sp  # noqa: E402


def _mk(uid, n, base=1_500_000_000, step=86_400):
    """n sessions for one user at strictly increasing connection epochs."""
    return [
        {"sessionID": f"{uid}_{i}", "userID": uid, "site": "caltech",
         "connection_epoch": base + i * step}
        for i in range(n)
    ]


def test_proportions_follow_floor_rule():
    recs = _mk("A", 20) + _mk("B", 25)
    out = sp.assign_splits(recs, min_sessions=20)
    for uid, n in (("A", 20), ("B", 25)):
        u = [r for r in out if r["userID"] == uid]
        counts = {s: sum(1 for r in u if r["split"] == s) for s in ("train", "val", "test")}
        assert counts["train"] == math.floor(0.60 * n)
        assert counts["val"] == math.floor(0.20 * n)
        assert counts["train"] + counts["val"] + counts["test"] == n


def test_chronology_is_preserved_per_user():
    recs = _mk("A", 23)
    out = sp.assign_splits(recs, min_sessions=20)
    times = {
        s: sorted(float(r["connection_epoch"]) for r in out if r["split"] == s)
        for s in ("train", "val", "test")
    }
    # every train session precedes every val session, and val precedes test
    assert max(times["train"]) <= min(times["val"])
    assert max(times["val"]) <= min(times["test"])


def test_users_below_threshold_are_dropped():
    recs = _mk("A", 20) + _mk("small", 19)
    out = sp.assign_splits(recs, min_sessions=20)
    assert {r["userID"] for r in out} == {"A"}


def test_every_session_assigned_exactly_once():
    recs = _mk("A", 21) + _mk("B", 34)
    out = sp.assign_splits(recs, min_sessions=20)
    ids = [r["sessionID"] for r in out]
    assert len(ids) == len(set(ids)) == 55  # 21 + 34, none dropped, none duplicated


def test_split_is_deterministic():
    recs = _mk("A", 27)
    a = sp.assign_splits(list(recs), min_sessions=20)
    b = sp.assign_splits(list(reversed(recs)), min_sessions=20)  # order shouldn't matter
    key = lambda out: sorted((r["sessionID"], r["split"]) for r in out)
    assert key(a) == key(b)
