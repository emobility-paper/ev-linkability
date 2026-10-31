"""Ablation group-mapping tests: groups partition the v2 feature set, and
resolution against an observer intersects correctly."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import ablation as ab  # noqa: E402
import build_pairs as bp  # noqa: E402


def test_groups_partition_v2_features():
    all_grouped = [c for cols in ab.GROUPS.values() for c in cols]
    # no feature appears in two groups
    assert len(all_grouped) == len(set(all_grouped))
    # groups cover exactly the v2 feature set
    assert set(all_grouped) == set(bp.FEATURE_COLUMNS["v2"])


def test_resolve_intersects_observer():
    # v1 has no location column, so a location-only ablation resolves empty
    assert ab.resolve_features("v1", ["location"]) == []
    # v2 all-groups resolves to the full v2 feature set
    assert set(ab.resolve_features("v2", ["time", "energy", "duration", "location"])) == \
        set(bp.FEATURE_COLUMNS["v2"])


def test_resolve_preserves_no_duplicates():
    cols = ab.resolve_features("v2", ["time", "time", "energy"])
    assert len(cols) == len(set(cols))


def test_a10_equals_full_feature_set():
    assert set(ab.resolve_features("v2", ab.ABLATIONS["A10_all"])) == \
        set(bp.FEATURE_COLUMNS["v2"])
