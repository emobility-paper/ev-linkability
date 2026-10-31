#!/usr/bin/env python3
"""Stage E4a: metadata-generalization mitigation engine (section 17).

Applies time / energy / duration rounding and location coarsening to single
sessions BEFORE pair features are recomputed (section 28.4). Combined profiles
P0-P4 (section 17.6):

    profile  time    energy   duration  location
    P0       exact   exact    exact     site*
    P1       15 min  0.5 kWh  15 min    site*   (*cluster unavailable in ACN-Data)
    P2       30 min  1 kWh    30 min    site
    P3       60 min  2 kWh    60 min    site
    P4       3 h     5 kWh    3 h       none

NOTE ON LOCATION: ACN-Data exposes site but not station/cluster, so P0/P1's
station/cluster levels are approximated by site. Location therefore only truly
changes at P4 (removed). Under P4 every site is set to a single constant, making
same_site vacuous -- the faithful representation of "location withheld."
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import build_pairs as bp

# time/duration in minutes; energy in kWh; None = no rounding.
PROFILES: Dict[str, Dict[str, Optional[float]]] = {
    "P0": {"time": None, "energy": None, "duration": None, "location": "site"},
    "P1": {"time": 15, "energy": 0.5, "duration": 15, "location": "site"},
    "P2": {"time": 30, "energy": 1.0, "duration": 30, "location": "site"},
    "P3": {"time": 60, "energy": 2.0, "duration": 60, "location": "site"},
    "P4": {"time": 180, "energy": 5.0, "duration": 180, "location": "none"},
}

_LOCATION_REMOVED = "__REMOVED__"


def round_time(minutes: np.ndarray, delta: Optional[float]) -> np.ndarray:
    """Round a minute-of-day array to a grid, wrapping at the day boundary."""
    if delta is None:
        return np.asarray(minutes, float).copy()
    return (delta * np.round(np.asarray(minutes, float) / delta)) % 1440


def round_value(x: np.ndarray, delta: Optional[float]) -> np.ndarray:
    """Round energy/duration to a grid; NaN passes through unchanged."""
    if delta is None:
        return np.asarray(x, float).copy()
    return delta * np.round(np.asarray(x, float) / delta)


def mitigate_arrays(arrays: Dict[str, np.ndarray], profile: str) -> Dict[str, np.ndarray]:
    """Return a mitigated copy of the session feature arrays under a profile."""
    p = PROFILES[profile]
    out = dict(arrays)  # weekday, is_weekend, connection_epoch, kWh pass through unless rounded
    out["arrival_minute"] = round_time(arrays["arrival_minute"], p["time"])
    out["departure_minute"] = round_time(arrays["departure_minute"], p["time"])
    out["duration_min"] = round_value(arrays["duration_min"], p["duration"])
    out["active_duration_min"] = round_value(arrays["active_duration_min"], p["duration"])
    out["idle_duration_min"] = round_value(arrays["idle_duration_min"], p["duration"])
    out["kWhDelivered"] = round_value(arrays["kWhDelivered"], p["energy"])

    # avg_power recomputed from the observable (rounded) energy and duration
    act = out["active_duration_min"]
    with np.errstate(divide="ignore", invalid="ignore"):
        power = np.where((act > 0), out["kWhDelivered"] / (act / 60.0), np.nan)
    out["avg_power"] = power

    if p["location"] == "none":
        out["site"] = np.full(len(arrays["site"]), _LOCATION_REMOVED, dtype=object)
    else:
        out["site"] = np.asarray(arrays["site"], dtype=object).copy()
    return out


def _paircol(name: str, a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> np.ndarray:
    if name == "circ_arrival_diff":
        d = np.abs(a["arrival_minute"] - b["arrival_minute"]) % 1440
        return np.minimum(d, 1440 - d)
    if name == "circ_departure_diff":
        d = np.abs(a["departure_minute"] - b["departure_minute"]) % 1440
        return np.minimum(d, 1440 - d)
    if name == "abs_duration_diff":
        return np.abs(a["duration_min"] - b["duration_min"])
    if name == "abs_energy_diff":
        return np.abs(a["kWhDelivered"] - b["kWhDelivered"])
    if name == "same_weekday":
        return (a["weekday"] == b["weekday"]).astype(float)
    if name == "same_weekend":
        return (a["is_weekend"] == b["is_weekend"]).astype(float)
    if name == "same_site":
        return (a["site"] == b["site"]).astype(float)
    if name == "abs_active_duration_diff":
        return np.abs(a["active_duration_min"] - b["active_duration_min"])
    if name == "abs_idle_duration_diff":
        return np.abs(a["idle_duration_min"] - b["idle_duration_min"])
    if name == "abs_avg_power_diff":
        return np.abs(a["avg_power"] - b["avg_power"])
    raise KeyError(name)


def build_pair_matrix(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray],
                      observer: str) -> np.ndarray:
    """Vectorized pair-feature matrix for aligned a-side/b-side session arrays."""
    feats: List[str] = bp.FEATURE_COLUMNS[observer]
    return np.column_stack([_paircol(c, a, b) for c in feats])
