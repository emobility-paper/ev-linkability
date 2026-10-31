"""Mitigation-engine tests: rounding grids, day-wrap, location removal, monotonicity."""
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import build_pairs as bp  # noqa: E402
import mitigations as mit  # noqa: E402


def test_round_time_grid_and_wrap():
    m = np.array([137.0, 1439.0, 0.0, 725.0])
    r30 = mit.round_time(m, 30)
    assert r30[0] == 150  # 137 -> nearest 30 is 150 (13 away vs 17 to 120)
    assert r30[1] == 0    # 1439 -> 1440 -> wraps to 0
    r60 = mit.round_time(m, 60)
    assert r60[3] == 720  # 725 -> 720


def test_round_value_and_nan_passthrough():
    x = np.array([12.34, 0.4, np.nan])
    r1 = mit.round_value(x, 1.0)
    assert r1[0] == 12.0
    assert r1[1] == 0.0
    assert np.isnan(r1[2])


def test_p0_is_identity_on_rounded_fields():
    arrays = _toy_arrays()
    out = mit.mitigate_arrays(arrays, "P0")
    assert np.array_equal(out["arrival_minute"], arrays["arrival_minute"])
    assert np.array_equal(out["kWhDelivered"], arrays["kWhDelivered"])


def test_p4_removes_location():
    arrays = _toy_arrays()
    out = mit.mitigate_arrays(arrays, "P4")
    assert len(set(out["site"])) == 1  # all collapsed to one constant


def test_coarser_profile_reduces_distinct_arrival_values():
    arrays = _toy_arrays(n=500)
    d0 = len(np.unique(mit.mitigate_arrays(arrays, "P0")["arrival_minute"]))
    d2 = len(np.unique(mit.mitigate_arrays(arrays, "P2")["arrival_minute"]))
    d4 = len(np.unique(mit.mitigate_arrays(arrays, "P4")["arrival_minute"]))
    assert d0 >= d2 >= d4


def test_pair_matrix_shape_and_same_site_vacuous_under_p4():
    arrays = _toy_arrays(n=40)
    mitd = mit.mitigate_arrays(arrays, "P4")
    a = {k: v[:20] for k, v in mitd.items()}
    b = {k: v[20:] for k, v in mitd.items()}
    X = mit.build_pair_matrix(a, b, "v2")
    assert X.shape == (20, len(bp.FEATURE_COLUMNS["v2"]))
    j = bp.FEATURE_COLUMNS["v2"].index("same_site")
    assert np.all(X[:, j] == 1.0)  # location removed -> same_site constant


def _toy_arrays(n=8):
    rng = np.random.default_rng(0)
    return {
        "arrival_minute": rng.integers(0, 1440, n).astype(float),
        "departure_minute": rng.integers(0, 1440, n).astype(float),
        "weekday": rng.integers(0, 7, n),
        "is_weekend": rng.integers(0, 2, n),
        "duration_min": rng.uniform(30, 600, n),
        "active_duration_min": rng.uniform(20, 400, n),
        "idle_duration_min": rng.uniform(0, 200, n),
        "kWhDelivered": rng.uniform(1, 40, n),
        "connection_epoch": 1_500_000_000 + rng.integers(0, 10_000_000, n).astype(float),
        "site": np.array(rng.choice(["caltech", "jpl"], n), dtype=object),
        "avg_power": rng.uniform(2, 12, n),
    }
