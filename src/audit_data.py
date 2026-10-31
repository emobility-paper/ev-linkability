#!/usr/bin/env python3
"""Stage 1 data audit for the EV linkability study.

Reads the raw <site>.json files produced by download_acndata.py and answers the
questions that decide whether the study is viable, WITHOUT touching any modeling
code. Nothing here selects thresholds by looking at outcomes; it only describes
the data.

Outputs:
  * a timezone sanity report (raw stored offset vs. site-local wall clock),
  * per-site eligibility funnel counts,
  * the decision-gate table: users with >= {10,15,20,30} eligible sessions,
  * a Table-2-style summary written to outputs/tables/audit_summary.csv.
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz  # already a dependency (acnportal + test suite); works on py3.6+

# --- §8 eligibility thresholds (fixed in advance, NOT tuned on results) ------
MIN_DURATION_MIN = 10.0
MAX_DURATION_HOURS = 24.0
MIN_ENERGY_KWH = 0.1
SESSION_THRESHOLDS = [10, 15, 20, 30]  # decision-gate ladder


def _load_sessions(path: Path) -> List[Dict[str, Any]]:
    """Return a list of session dicts regardless of the file's top-level shape.

    Handles three shapes seen in the wild:
      * a plain JSON array (acnportal client / download_acndata.py output),
      * a Python-Eve envelope: {"_items": [...], "_meta": {...}},
      * a dict keyed by id: {"sess_1": {...}, ...}.
    Non-dict entries are skipped with a warning rather than crashing.
    """
    data = json.loads(path.read_text())
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if isinstance(data.get("_items"), list):
            items = data["_items"]
        else:
            items = list(data.values())
    else:
        raise TypeError(
            f"{path.name}: unexpected top-level JSON type {type(data).__name__}"
        )
    good = [s for s in items if isinstance(s, dict)]
    if len(good) != len(items):
        print(f"[warn] {path.name}: skipped {len(items) - len(good)} non-dict entries")
    return good


def _parse(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts)
    except ValueError:
        return None


def _local(ts: dt.datetime, tz_name: str) -> dt.datetime:
    """Convert a tz-aware timestamp to the site's local wall clock.

    This is correct whether the client handed back UTC (+00:00) or already
    site-local strings: astimezone() resolves both to the same wall clock.
    """
    return ts.astimezone(pytz.timezone(tz_name))


def audit_site(path: Path) -> Dict[str, Any]:
    sessions: List[Dict[str, Any]] = _load_sessions(path)
    site = path.stem

    raw = len(sessions)
    claimed = 0
    eligible_by_user: Dict[str, int] = defaultdict(int)
    reasons = Counter()
    raw_offsets = Counter()
    local_arrival_hours = Counter()

    for s in sessions:
        uid = s.get("userID")
        if uid in (None, "", "null", "None"):
            reasons["no_userID"] += 1
            continue
        claimed += 1

        conn = _parse(s.get("connectionTime"))
        disc = _parse(s.get("disconnectTime"))
        if conn is None or disc is None:
            reasons["bad_timestamp"] += 1
            continue

        # Timezone sanity: record the offset the client actually stored, and the
        # site-local arrival hour after explicit conversion.
        off = conn.utcoffset()
        raw_offsets[str(off)] += 1
        tz_name = s.get("timezone") or "America/Los_Angeles"
        local_arrival_hours[_local(conn, tz_name).hour] += 1

        if disc <= conn:
            reasons["disconnect<=connect"] += 1
            continue
        dur_min = (disc - conn).total_seconds() / 60.0
        if dur_min < MIN_DURATION_MIN:
            reasons["too_short"] += 1
            continue
        if dur_min > MAX_DURATION_HOURS * 60:
            reasons["too_long"] += 1
            continue
        energy = s.get("kWhDelivered")
        if energy is None or energy <= MIN_ENERGY_KWH:
            reasons["low_energy"] += 1
            continue

        eligible_by_user[uid] += 1

    eligible_sessions = sum(eligible_by_user.values())
    gate = {
        t: sum(1 for c in eligible_by_user.values() if c >= t)
        for t in SESSION_THRESHOLDS
    }
    counts = sorted(eligible_by_user.values())
    median = counts[len(counts) // 2] if counts else 0

    return {
        "site": site,
        "raw_sessions": raw,
        "claimed_sessions": claimed,
        "eligible_sessions": eligible_sessions,
        "eligible_users_any": len(eligible_by_user),
        "median_sessions_per_eligible_user": median,
        "gate": gate,
        "drop_reasons": dict(reasons),
        "raw_offsets": dict(raw_offsets),
        "arrival_hour_hist": dict(sorted(local_arrival_hours.items())),
    }


def main() -> int:
    raw_dir = Path(os.environ.get("RAW_DIR", "data/raw"))
    files = [Path(p) for p in glob.glob(str(raw_dir / "*.json"))
             if not p.endswith("download_manifest.json")]
    if not files:
        print(f"No site JSON files found in {raw_dir}/")
        return 1

    results = [audit_site(p) for p in sorted(files)]

    print("=" * 68)
    print("TIMEZONE SANITY  (the offset the client actually stored)")
    print("=" * 68)
    for r in results:
        print(f"[{r['site']}] connectionTime offsets seen: {r['raw_offsets']}")
        print(f"    -> if you see '0:00:00' the client returned UTC; local hours")
        print(f"       below are already corrected via the timezone field.")
        peak = sorted(r["arrival_hour_hist"].items(), key=lambda kv: -kv[1])[:3]
        print(f"    site-local peak arrival hours: {peak}")
    print()

    print("=" * 68)
    print("ELIGIBILITY FUNNEL  (§8 thresholds, fixed in advance)")
    print("=" * 68)
    for r in results:
        print(f"[{r['site']}] raw={r['raw_sessions']}  claimed={r['claimed_sessions']}"
              f"  eligible={r['eligible_sessions']}  eligible_users={r['eligible_users_any']}"
              f"  median_sess/user={r['median_sessions_per_eligible_user']}")
        print(f"    drops: {r['drop_reasons']}")
    print()

    print("=" * 68)
    print("DECISION GATE  (users with >= N eligible sessions)")
    print("=" * 68)
    header = "site      " + "".join(f"  >= {t:>2}" for t in SESSION_THRESHOLDS)
    print(header)
    for r in results:
        row = f"{r['site']:<10}" + "".join(f"  {r['gate'][t]:>4}" for t in SESSION_THRESHOLDS)
        print(row)
    tot = {t: sum(r["gate"][t] for r in results) for t in SESSION_THRESHOLDS}
    print(f"{'TOTAL':<10}" + "".join(f"  {tot[t]:>4}" for t in SESSION_THRESHOLDS))
    print()
    print(">>> Study is viable at threshold 20 if the '>= 20' TOTAL is large")
    print("    enough to split 60/20/20 and still leave enough test users.")

    out = Path("outputs/tables"); out.mkdir(parents=True, exist_ok=True)
    with (out / "audit_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "raw", "claimed", "eligible", "eligible_users",
                    "median_sess_per_user"] + [f"users_ge_{t}" for t in SESSION_THRESHOLDS])
        for r in results:
            w.writerow([r["site"], r["raw_sessions"], r["claimed_sessions"],
                        r["eligible_sessions"], r["eligible_users_any"],
                        r["median_sessions_per_eligible_user"]]
                       + [r["gate"][t] for t in SESSION_THRESHOLDS])
    print(f"Wrote {out/'audit_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
