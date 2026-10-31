#!/usr/bin/env python3
"""Stage 2b: leakage-safe per-user chronological split (section 10 / 28.1).

For each user, sessions are ordered by connection time; the earliest 60% go to
train, the next 20% to validation, the latest 20% to test (floor boundaries).
Pairs are built LATER and only within a split, so no session ever crosses a
partition -- this is what prevents the optimistic leakage a random row split
would cause.

The cohort filter (>= --min-sessions eligible sessions per user) is applied
here, not in preprocess, so a single interim table serves the whole
10/15/20/30 sweep.

Outputs (data/processed/):
    splits_min{N}.csv        sessionID, site, user_hash, split, connection_epoch
    split_stats_min{N}.csv   per-site train/val/test session + user counts
Restricted vault (gitignored; raw userID for debugging only):
    ground_truth_min{N}.csv  sessionID, userID

Run (after preprocess):
    python src/split_sessions.py --min-sessions 20
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import math
import os
import secrets
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import preprocess as pp  # single source of truth for eligibility + leakage guards

SEED = 20260712  # fixed; recorded for reproducibility
DEFAULT_RATIOS = (0.60, 0.20, 0.20)

# The only columns the SHARED split map is allowed to carry. No raw userID:
# grouping is done via a salted, non-reversible user_hash.
PUBLIC_SPLIT_COLUMNS = ["sessionID", "site", "user_hash", "split", "connection_epoch"]


# --- pure, testable split logic ---------------------------------------------
def chronological_split(
    items: Sequence[Any], ratios: Tuple[float, float, float] = DEFAULT_RATIOS
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split an already-ordered sequence into (train, val, test) by floor rule."""
    n = len(items)
    n_train = int(math.floor(ratios[0] * n))
    n_val = int(math.floor(ratios[1] * n))
    return (
        list(items[:n_train]),
        list(items[n_train:n_train + n_val]),
        list(items[n_train + n_val:]),
    )


def _group_key(r: Dict[str, Any], group_by: str) -> str:
    """Identity used for grouping. 'user' = the person (global, spans sites);
    'user_site' = the person-at-a-site (keeps pairs within one site, matches a
    single-CPO threat model)."""
    return str(r["userID"]) if group_by == "user" else f'{r["userID"]}||{r["site"]}'


def assign_splits(
    records: List[Dict[str, Any]],
    min_sessions: int = 20,
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    group_by: str = "user_site",
) -> List[Dict[str, Any]]:
    """Return cohort records annotated with 'split' and 'group_key'.

    Groups below ``min_sessions`` are dropped. Each surviving group's sessions
    are sorted by (connection_epoch, sessionID) -- the sessionID tie-break keeps
    the ordering deterministic across runs.
    """
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_group[_group_key(r, group_by)].append(r)

    out: List[Dict[str, Any]] = []
    for gkey, recs in by_group.items():
        if len(recs) < min_sessions:
            continue
        recs_sorted = sorted(recs, key=lambda r: (float(r["connection_epoch"]), str(r["sessionID"])))
        train, val, test = chronological_split(recs_sorted, ratios)
        for split_name, group in (("train", train), ("val", val), ("test", test)):
            for r in group:
                out.append({**r, "split": split_name, "group_key": gkey})
    return out


# --- ground-truth hashing ----------------------------------------------------
def _resolve_salt(processed_dir: Path) -> bytes:
    """Read the HMAC salt from env, or generate+persist one (gitignored).

    A secret salt makes the user_hash non-reversible without it, satisfying the
    ethics statement, while staying stable across reruns on the same machine.
    """
    env = os.environ.get("EVLINK_SALT")
    if env:
        return env.encode()
    salt_file = processed_dir / ".gt_salt"
    if salt_file.exists():
        return salt_file.read_text().strip().encode()
    salt = secrets.token_hex(16)
    salt_file.write_text(salt)
    return salt.encode()


def _hash_user(uid: str, salt: bytes) -> str:
    return hmac.new(salt, str(uid).encode(), hashlib.sha256).hexdigest()[:16]


# --- CSV IO ------------------------------------------------------------------
def load_eligible(interim_dir: Path) -> List[Dict[str, Any]]:
    path = interim_dir / "eligible_sessions.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-sessions", type=int, default=20)
    parser.add_argument(
        "--group-by", choices=["user_site", "user"], default="user_site",
        help="Identity for grouping: user_site (within-site, default) or user (global).",
    )
    parser.add_argument("--interim-dir", default=os.environ.get("INTERIM_DIR", "data/interim"))
    parser.add_argument("--processed-dir", default=os.environ.get("PROCESSED_DIR", "data/processed"))
    args = parser.parse_args(argv)

    interim_dir = Path(args.interim_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    records = load_eligible(interim_dir)
    labeled = assign_splits(records, min_sessions=args.min_sessions, group_by=args.group_by)
    if not labeled:
        print(f"No groups clear the >= {args.min_sessions} threshold.")
        return 1

    salt = _resolve_salt(processed_dir)
    n = args.min_sessions
    tag = f"min{n}_{args.group_by}"

    # public split map (no raw userID) + restricted vault (raw userID)
    split_path = processed_dir / f"splits_{tag}.csv"
    vault_path = processed_dir / f"ground_truth_{tag}.csv"
    with split_path.open("w", newline="", encoding="utf-8") as sf, \
            vault_path.open("w", newline="", encoding="utf-8") as vf:
        sw = csv.DictWriter(sf, fieldnames=PUBLIC_SPLIT_COLUMNS)
        sw.writeheader()
        vw = csv.DictWriter(vf, fieldnames=["sessionID", "userID"])
        vw.writeheader()
        for r in labeled:
            sw.writerow({
                "sessionID": r["sessionID"],
                "site": r["site"],
                "user_hash": _hash_user(r["group_key"], salt),
                "split": r["split"],
                "connection_epoch": r["connection_epoch"],
            })
            vw.writerow({"sessionID": r["sessionID"], "userID": r["userID"]})

    # belt-and-braces: the file we just wrote must carry no raw identifiers
    pp.assert_no_raw_identifiers(PUBLIC_SPLIT_COLUMNS)

    # per-site split stats (feeds Table 2)
    stats: Dict[Tuple[str, str], Dict[str, set]] = defaultdict(lambda: {"users": set(), "n": 0})
    for r in labeled:
        cell = stats[(r["site"], r["split"])]
        cell["users"].add(r["userID"])
        cell["n"] += 1

    stats_path = processed_dir / f"split_stats_{tag}.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "split", "n_sessions", "n_users"])
        sites = sorted({s for s, _ in stats})
        for site in sites:
            for split in ("train", "val", "test"):
                cell = stats.get((site, split), {"users": set(), "n": 0})
                w.writerow([site, split, cell["n"], len(cell["users"])])

    cohort_users = len({r["group_key"] for r in labeled})
    print(f"min_sessions={n}, group_by={args.group_by}: "
          f"{cohort_users} cohort identities, {len(labeled)} sessions")
    print(f"  wrote {split_path}")
    print(f"  wrote {stats_path}")
    print(f"  wrote {vault_path}  (RESTRICTED / gitignored -- raw userID)")
    for site in sorted({r['site'] for r in labeled}):
        parts = {sp: sum(1 for r in labeled if r['site'] == site and r['split'] == sp)
                 for sp in ('train', 'val', 'test')}
        print(f"  [{site}] train={parts['train']}  val={parts['val']}  test={parts['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
