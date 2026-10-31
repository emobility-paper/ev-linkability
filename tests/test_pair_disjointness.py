"""Leakage guards: no session crosses a split, no raw identifier is shared.

A session appearing in two partitions is the classic way pairwise AUC gets
silently inflated, so it is asserted mechanically here.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import preprocess as pp  # noqa: E402
import split_sessions as sp  # noqa: E402


def _mk(uid, n, base=1_500_000_000, step=3_600):
    return [
        {"sessionID": f"{uid}_{i}", "userID": uid, "site": "jpl",
         "connection_epoch": base + i * step}
        for i in range(n)
    ]


def test_splits_are_session_disjoint():
    out = sp.assign_splits(_mk("A", 22) + _mk("B", 30), min_sessions=20)
    by = {s: {r["sessionID"] for r in out if r["split"] == s}
          for s in ("train", "val", "test")}
    assert by["train"].isdisjoint(by["val"])
    assert by["train"].isdisjoint(by["test"])
    assert by["val"].isdisjoint(by["test"])


def test_split_map_carries_no_raw_identifiers():
    # the exact columns the shared split file is written with
    pp.assert_no_raw_identifiers(sp.PUBLIC_SPLIT_COLUMNS)


def test_feature_guard_rejects_grouping_keys_and_ids():
    # user_hash and sessionID are fine in the split map but must NOT be features
    with pytest.raises(AssertionError):
        pp.assert_no_feature_leakage(["arrival_minute", "user_hash"])
    with pytest.raises(AssertionError):
        pp.assert_no_feature_leakage(["arrival_minute", "sessionID"])
    with pytest.raises(AssertionError):
        pp.assert_no_feature_leakage(["userID", "kWhDelivered"])


def test_feature_guard_passes_clean_feature_list():
    pp.assert_no_feature_leakage(
        ["arrival_minute", "departure_minute", "duration_min", "kWhDelivered"]
    )
