#!/usr/bin/env python3
"""Stage 2a: turn raw site JSON into a canonical table of eligible sessions.

This module is the SINGLE SOURCE OF TRUTH for parsing, timezone localization,
and the section-8 eligibility rules. split_sessions.py imports from here, so the
cohort that gets split is exactly the cohort that passed eligibility -- the two
can never drift apart.

Output (restricted; contains raw userID -> gitignored):
    data/interim/eligible_sessions.csv

Run:
    RAW_DIR=data/raw python src/preprocess.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytz  # already a dependency (acnportal + tests); portable on py3.6+

# --- section-8 eligibility, fixed in advance (identical to audit_data.py) ----
MIN_DURATION_MIN = 10.0
MAX_DURATION_HOURS = 24.0
MIN_ENERGY_KWH = 0.1
VALID_SITES = {"caltech", "jpl", "office001"}

# --- leakage vocabulary ------------------------------------------------------
# Raw identifiers must never appear in any SHARED artifact (split map, repo).
RAW_IDENTIFIERS = {"userID", "eMAID", "VIN", "contract_id"}
# Columns that must never enter a MODEL FEATURE matrix. This is a superset:
# grouping keys (user_hash, sessionID) are legitimate in the split map for
# building pairs, but must never be handed to a model as a feature.
FORBIDDEN_FEATURES = RAW_IDENTIFIERS | {"user_hash", "sessionID", "session_id"}

# Columns written to the canonical eligible-sessions table.
ELIGIBLE_COLUMNS = [
    "sessionID", "site", "userID", "timezone",
    "connection_iso", "connection_epoch",
    "arrival_minute", "departure_minute",
    "weekday", "is_weekend", "month",
    "duration_min", "active_duration_min", "idle_duration_min",
    "kWhDelivered",
]


def assert_no_raw_identifiers(columns) -> None:
    """Fail if a shared artifact would expose a raw identifier."""
    leaked = RAW_IDENTIFIERS & set(columns)
    assert not leaked, f"raw identifiers leaked into shared artifact: {sorted(leaked)}"


def assert_no_feature_leakage(model_features) -> None:
    """Fail if a model feature list contains an identifier or grouping key."""
    leaked = FORBIDDEN_FEATURES & set(model_features)
    assert not leaked, f"forbidden columns in model features: {sorted(leaked)}"


# --- IO / parsing (shared with the audit) ------------------------------------
def _load_sessions(path: Path) -> List[Dict[str, Any]]:
    """Return a list of session dicts regardless of the file's top-level shape."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data["_items"] if isinstance(data.get("_items"), list) else list(data.values())
    else:
        raise TypeError(f"{path.name}: unexpected top-level JSON {type(data).__name__}")
    return [s for s in items if isinstance(s, dict)]


def _parse(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts)
    except ValueError:
        return None


def _local(ts: dt.datetime, tz_name: str) -> dt.datetime:
    """Convert an aware timestamp to site-local wall clock (UTC or local input)."""
    return ts.astimezone(pytz.timezone(tz_name))


def clean_session(s: Dict[str, Any], site: str) -> Optional[Dict[str, Any]]:
    """Apply per-session eligibility and derive features. None if ineligible.

    Per-session only: the >= min-sessions-per-user cohort filter is applied
    later, in split_sessions, so one interim table serves every threshold in
    the 10/15/20/30 sweep.
    """
    uid = s.get("userID")
    if uid in (None, "", "null", "None"):
        return None
    conn = _parse(s.get("connectionTime"))
    disc = _parse(s.get("disconnectTime"))
    if conn is None or disc is None or disc <= conn:
        return None
    dur_min = (disc - conn).total_seconds() / 60.0
    if dur_min < MIN_DURATION_MIN or dur_min > MAX_DURATION_HOURS * 60:
        return None
    energy = s.get("kWhDelivered")
    if energy is None or energy <= MIN_ENERGY_KWH:
        return None

    tz_name = s.get("timezone") or "America/Los_Angeles"
    conn_l = _local(conn, tz_name)
    disc_l = _local(disc, tz_name)

    # active/idle only when doneChargingTime is present and sane (section 9.2:
    # do NOT silently equate a missing done-time to disconnect).
    done = _parse(s.get("doneChargingTime"))
    active = idle = None
    if done is not None and conn <= done <= disc:
        active = (done - conn).total_seconds() / 60.0
        idle = (disc - done).total_seconds() / 60.0

    return {
        "sessionID": s.get("sessionID"),
        "site": site,  # authoritative: the file's site, not the sometimes-int siteID field
        "userID": uid,
        "timezone": tz_name,
        "connection_iso": conn.isoformat(),
        "connection_epoch": conn.timestamp(),
        "arrival_minute": conn_l.hour * 60 + conn_l.minute,
        "departure_minute": disc_l.hour * 60 + disc_l.minute,
        "weekday": conn_l.weekday(),
        "is_weekend": int(conn_l.weekday() >= 5),
        "month": conn_l.month,
        "duration_min": round(dur_min, 3),
        "active_duration_min": round(active, 3) if active is not None else None,
        "idle_duration_min": round(idle, 3) if idle is not None else None,
        "kWhDelivered": energy,
    }


def iter_eligible(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    for site in sorted(VALID_SITES):
        p = raw_dir / f"{site}.json"
        if not p.exists():
            continue
        for s in _load_sessions(p):
            rec = clean_session(s, site=site)
            if rec is not None:
                yield rec


def main() -> int:
    raw_dir = Path(os.environ.get("RAW_DIR", "data/raw"))
    out_dir = Path(os.environ.get("INTERIM_DIR", "data/interim"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_eligible(raw_dir))
    if not rows:
        print(f"No eligible sessions found under {raw_dir}/")
        return 1

    out_path = out_dir / "eligible_sessions.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ELIGIBLE_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # sanity summary (cross-check against the audit's eligible counts)
    per_site: Dict[str, int] = {}
    users_per_site: Dict[str, set] = {}
    for r in rows:
        per_site[r["site"]] = per_site.get(r["site"], 0) + 1
        users_per_site.setdefault(r["site"], set()).add(r["userID"])
    print(f"Wrote {len(rows)} eligible sessions -> {out_path}")
    for site in sorted(per_site, key=str):
        print(f"  [{site}] eligible={per_site[site]}  users={len(users_per_site[site])}")
    print("NOTE: this file contains raw userID and is gitignored (restricted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
