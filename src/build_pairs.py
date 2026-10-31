#!/usr/bin/env python3
"""Stage 3: build labeled session pairs for pairwise linkability (section 13).

Pairs are built WITHIN each split only, so no pair straddles the train/val/test
boundary. Positives are same-identity session pairs (capped per identity);
negatives are different-identity pairs at a 1:5 positive:negative ratio, of
which a configurable fraction are HARD negatives -- same site and a nearby
arrival time -- and the rest random.

Feature columns follow the observer view (section 12). Provenance columns
(session ids, user_hash) are written for auditing and user-level bootstrap but
are NEVER model features; test_no_identifier_leakage enforces that separation.

Run (after split_sessions):
    python src/build_pairs.py \
        --splits data/processed/splits_min20_user_site.csv \
        --observer v2
"""
from __future__ import annotations

import argparse
import bisect
import csv
import itertools
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import preprocess as pp  # leakage vocabulary lives here

SEED = 20260712

# --- feature / provenance vocabulary ----------------------------------------
_V1_FEATURES = [
    "circ_arrival_diff", "circ_departure_diff",
    "abs_duration_diff", "abs_energy_diff",
    "same_weekday", "same_weekend",
]
_V2_EXTRA = [
    "same_site", "abs_active_duration_diff",
    "abs_idle_duration_diff", "abs_avg_power_diff",
]
FEATURE_COLUMNS: Dict[str, List[str]] = {
    "v1": list(_V1_FEATURES),
    "v2": list(_V1_FEATURES) + list(_V2_EXTRA),
}
# Written for grouping/audit only -- must never be handed to a model.
PROVENANCE_COLUMNS = [
    "session_id_a", "session_id_b", "user_hash_a", "user_hash_b",
    "site_a", "site_b", "split",
]


# --- pure helpers (unit-tested directly) ------------------------------------
def circular_diff(a: float, b: float, period: int = 1440) -> float:
    """Smallest distance between two minutes-of-day on a 24h circle."""
    d = abs(a - b) % period
    return min(d, period - d)


def _absdiff(x: Optional[float], y: Optional[float]) -> Optional[float]:
    return abs(x - y) if x is not None and y is not None else None


def pair_features(sa: Dict[str, Any], sb: Dict[str, Any], observer: str) -> Dict[str, Any]:
    """Symmetric pairwise features for one session pair under an observer view."""
    f = {
        "circ_arrival_diff": circular_diff(sa["arrival_minute"], sb["arrival_minute"]),
        "circ_departure_diff": circular_diff(sa["departure_minute"], sb["departure_minute"]),
        "abs_duration_diff": abs(sa["duration_min"] - sb["duration_min"]),
        "abs_energy_diff": abs(sa["kWhDelivered"] - sb["kWhDelivered"]),
        "same_weekday": int(sa["weekday"] == sb["weekday"]),
        "same_weekend": int(sa["is_weekend"] == sb["is_weekend"]),
    }
    if observer == "v2":
        f["same_site"] = int(sa["site"] == sb["site"])
        f["abs_active_duration_diff"] = _absdiff(sa["active_duration_min"], sb["active_duration_min"])
        f["abs_idle_duration_diff"] = _absdiff(sa["idle_duration_min"], sb["idle_duration_min"])
        f["abs_avg_power_diff"] = _absdiff(sa["avg_power"], sb["avg_power"])
    return f


def _row(sa, sb, observer, split, label) -> Dict[str, Any]:
    row = {
        "session_id_a": sa["sessionID"], "session_id_b": sb["sessionID"],
        "user_hash_a": sa["user_hash"], "user_hash_b": sb["user_hash"],
        "site_a": sa["site"], "site_b": sb["site"], "split": split,
    }
    row.update(pair_features(sa, sb, observer))
    row["y"] = label
    return row


# --- hard-negative index -----------------------------------------------------
class _ArrivalIndex:
    """Per-site sessions sorted by arrival minute, for windowed hard negatives."""

    def __init__(self, sessions: List[Dict[str, Any]]):
        self.by_site: Dict[str, Tuple[List[int], List[Dict[str, Any]]]] = {}
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in sessions:
            buckets[s["site"]].append(s)
        for site, recs in buckets.items():
            recs.sort(key=lambda r: r["arrival_minute"])
            self.by_site[site] = ([r["arrival_minute"] for r in recs], recs)

    def _window(self, site: str, lo: int, hi: int) -> List[Dict[str, Any]]:
        arrivals, recs = self.by_site[site]
        i = bisect.bisect_left(arrivals, lo)
        j = bisect.bisect_right(arrivals, hi)
        return recs[i:j]

    def candidates(self, anchor: Dict[str, Any], window: int) -> List[Dict[str, Any]]:
        """Same-site sessions whose arrival is within +/-window (circular)."""
        site = anchor["site"]
        if site not in self.by_site:
            return []
        a = anchor["arrival_minute"]
        lo, hi = a - window, a + window
        out: List[Dict[str, Any]] = []
        if lo < 0:
            out += self._window(site, 0, hi)
            out += self._window(site, lo + 1440, 1439)
        elif hi > 1439:
            out += self._window(site, lo, 1439)
            out += self._window(site, 0, hi - 1440)
        else:
            out += self._window(site, lo, hi)
        return out


# --- pair construction for one split ----------------------------------------
def build_split_pairs(
    sessions: List[Dict[str, Any]],
    observer: str,
    split: str,
    rng: random.Random,
    positive_cap: int = 200,
    neg_ratio: int = 5,
    hard_frac: float = 0.60,
    window: int = 120,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Return (pair rows, per-identity positive counts) for one split."""
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sessions:
        by_id[s["user_hash"]].append(s)

    index = _ArrivalIndex(sessions)
    all_sessions = sessions
    rows: List[Dict[str, Any]] = []
    pos_counts: Dict[str, int] = {}
    seen: set = set()  # frozenset({id_a, id_b}) to dedupe

    def _key(x, y):
        return frozenset((x, y))

    for uh, recs in by_id.items():
        # ---- positives: same-identity pairs, capped ----
        pairs = list(itertools.combinations(recs, 2))
        if len(pairs) > positive_cap:
            pairs = rng.sample(pairs, positive_cap)
        n_pos = 0
        for sa, sb in pairs:
            k = _key(sa["sessionID"], sb["sessionID"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(_row(sa, sb, observer, split, 1))
            n_pos += 1
        pos_counts[uh] = n_pos
        if n_pos == 0:
            continue

        # ---- negatives: anchored on this identity's sessions ----
        n_neg = n_pos * neg_ratio
        n_hard = int(round(n_neg * hard_frac))
        made_hard = made_rand = 0
        attempts = 0
        max_attempts = 40 * n_neg + 50
        while (made_hard + made_rand) < n_neg and attempts < max_attempts:
            attempts += 1
            anchor = rng.choice(recs)
            want_hard = made_hard < n_hard
            if want_hard:
                cands = [c for c in index.candidates(anchor, window)
                         if c["user_hash"] != uh]
                if not cands:
                    other = rng.choice(all_sessions)  # fall back to random
                    is_hard = False
                else:
                    other = rng.choice(cands)
                    is_hard = True
            else:
                other = rng.choice(all_sessions)
                is_hard = False
            if other["user_hash"] == uh:
                continue
            k = _key(anchor["sessionID"], other["sessionID"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(_row(anchor, other, observer, split, 0))
            if is_hard:
                made_hard += 1
            else:
                made_rand += 1

    return rows, pos_counts


# --- IO ----------------------------------------------------------------------
def _to_float(v):
    return float(v) if v not in (None, "", "None") else None


def load_joined_sessions(splits_path: Path, features_path: Path) -> List[Dict[str, Any]]:
    """Join the split map (user_hash, split) to session features by sessionID."""
    feats: Dict[str, Dict[str, Any]] = {}
    with features_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            active = _to_float(r["active_duration_min"])
            energy = _to_float(r["kWhDelivered"])
            avg_power = (energy / (active / 60.0)) if (active and active > 0 and energy is not None) else None
            feats[r["sessionID"]] = {
                "sessionID": r["sessionID"],
                "site": r["site"],
                "arrival_minute": int(r["arrival_minute"]),
                "departure_minute": int(r["departure_minute"]),
                "weekday": int(r["weekday"]),
                "is_weekend": int(r["is_weekend"]),
                "duration_min": _to_float(r["duration_min"]),
                "active_duration_min": active,
                "idle_duration_min": _to_float(r["idle_duration_min"]),
                "kWhDelivered": energy,
                "avg_power": avg_power,
            }
    joined: List[Dict[str, Any]] = []
    with splits_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f = feats.get(r["sessionID"])
            if f is None:
                continue
            joined.append({**f, "user_hash": r["user_hash"], "split": r["split"]})
    return joined


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", required=True, help="splits_*.csv from split_sessions.")
    parser.add_argument("--features", default=os.environ.get(
        "INTERIM_DIR", "data/interim") + "/eligible_sessions.csv")
    parser.add_argument("--observer", choices=["v1", "v2"], default="v2")
    parser.add_argument("--out-dir", default=os.environ.get("PROCESSED_DIR", "data/processed"))
    parser.add_argument("--positive-cap", type=int, default=200)
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--hard-frac", type=float, default=0.60)
    parser.add_argument("--window", type=int, default=120, help="Hard-neg arrival window (min).")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    sessions = load_joined_sessions(Path(args.splits), Path(args.features))
    by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sessions:
        by_split[s["split"]].append(s)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = PROVENANCE_COLUMNS + FEATURE_COLUMNS[args.observer] + ["y"]

    # belt-and-braces before writing anything
    pp.assert_no_feature_leakage(FEATURE_COLUMNS[args.observer])

    tag = Path(args.splits).stem.replace("splits_", "")
    for split in ("train", "val", "test"):
        recs = by_split.get(split, [])
        if not recs:
            continue
        rng = random.Random(args.seed)  # per-split reproducibility
        rows, pos_counts = build_split_pairs(
            recs, args.observer, split, rng,
            positive_cap=args.positive_cap, neg_ratio=args.neg_ratio,
            hard_frac=args.hard_frac, window=args.window,
        )
        out_path = out_dir / f"pairs_{split}_{tag}_{args.observer}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        n_pos = sum(1 for r in rows if r["y"] == 1)
        counts = sorted(pos_counts.values())
        med = counts[len(counts) // 2] if counts else 0
        print(f"[{split}] {len(rows)} pairs ({n_pos} pos / {len(rows)-n_pos} neg) "
              f"across {len(pos_counts)} identities; median positives/identity={med} "
              f"-> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
