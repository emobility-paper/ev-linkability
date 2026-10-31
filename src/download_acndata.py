#!/usr/bin/env python3
"""Download ACN-Data charging sessions using the maintained acnportal client.

This script pulls raw charging-session records for one or more ACN sites and
writes them to disk together with a reproducibility manifest. It is written
against the real ``acnportal.acndata.DataClient`` API (v0.3.3).

Design notes
------------
* The token is NEVER hardcoded. It is read from the ``ACNDATA_API_TOKEN``
  environment variable, or prompted for interactively (hidden input). A
  ``--token`` flag exists only as an escape hatch and is discouraged because it
  can leak into shell history.
* The acnportal client converts every RFC-1123 timestamp string into a
  timezone-aware ``datetime`` (localized to each site's timezone) *before*
  handing back the session dict. A naive ``json.dump`` therefore raises
  "Object of type datetime is not JSON serializable". We serialize every
  datetime as ISO-8601 *with UTC offset*, which preserves the tz-awareness the
  downstream pipeline depends on.
* The raw store keeps BOTH claimed and unclaimed sessions. Dropping unclaimed
  sessions is a cleaning-stage decision, not a download-stage one; the manifest
  records the claimed/unclaimed counts so the data-audit stage has them.

Usage
-----
    export ACNDATA_API_TOKEN="your_token_here"
    python download_acndata.py --sites caltech jpl --out-dir ../data/raw

    # optional date bounds (ISO 8601, interpreted as UTC if no offset given):
    python download_acndata.py --start 2018-05-01 --end 2020-01-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from acnportal.acndata import DataClient
except ImportError:  # pragma: no cover - environment-dependent
    DataClient = None  # type: ignore

DEFAULT_URL = "https://ev.caltech.edu/api/v1/"
VALID_SITES = {"caltech", "jpl", "office001"}
# Fields that identify a session as "claimed" (associated with a user account).
USER_ID_FIELD = "userID"


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    """Serialize datetimes (at any nesting depth) as ISO-8601 with offset.

    json.dump recurses into dicts/lists natively and only calls this hook for
    leaf objects it cannot serialize, so any tz-aware datetime nested anywhere
    in the session document is handled here.
    """
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _is_claimed(session: Dict[str, Any]) -> bool:
    val = session.get(USER_ID_FIELD)
    return val not in (None, "", "null", "None")


# --------------------------------------------------------------------------- #
# Token handling
# --------------------------------------------------------------------------- #
def resolve_token(cli_token: Optional[str]) -> str:
    if cli_token:
        sys.stderr.write(
            "WARNING: passing --token on the command line can leak the token "
            "into shell history. Prefer the ACNDATA_API_TOKEN env var.\n"
        )
        return cli_token
    env_token = os.environ.get("ACNDATA_API_TOKEN")
    if env_token:
        return env_token
    return getpass.getpass("ACN-Data API token (input hidden): ").strip()


# --------------------------------------------------------------------------- #
# Date parsing for CLI bounds
# --------------------------------------------------------------------------- #
def parse_cli_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    """Parse a CLI date/datetime. Naive inputs are treated as UTC.

    The client's get_sessions_by_time calls .astimezone() on these, which
    requires tz-aware datetimes, so we attach UTC when no offset is present.
    """
    if value is None:
        return None
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


# --------------------------------------------------------------------------- #
# Core download
# --------------------------------------------------------------------------- #
def iter_site_sessions(
    client: "DataClient",
    site: str,
    start: Optional[dt.datetime],
    end: Optional[dt.datetime],
    min_energy: Optional[float],
) -> Iterable[Dict[str, Any]]:
    """Yield sessions for a site, using time bounds when provided."""
    if start is not None or end is not None or min_energy is not None:
        yield from client.get_sessions_by_time(
            site, start=start, end=end, min_energy=min_energy, timeseries=False
        )
    else:
        yield from client.get_sessions(site, sort="connectionTime", timeseries=False)


def download_site(
    client: "DataClient",
    site: str,
    out_dir: Path,
    start: Optional[dt.datetime],
    end: Optional[dt.datetime],
    min_energy: Optional[float],
    progress_every: int = 1000,
) -> Dict[str, Any]:
    """Download one site, write <site>.json, and return manifest stats."""
    sessions: List[Dict[str, Any]] = []
    claimed = 0
    observed_fields: set[str] = set()
    first_conn: Optional[str] = None
    last_conn: Optional[str] = None

    for i, s in enumerate(iter_site_sessions(client, site, start, end, min_energy), 1):
        sessions.append(s)
        observed_fields.update(s.keys())
        if _is_claimed(s):
            claimed += 1
        conn = s.get("connectionTime")
        conn_iso = conn.isoformat() if isinstance(conn, dt.datetime) else conn
        if conn_iso is not None:
            if first_conn is None or conn_iso < first_conn:
                first_conn = conn_iso
            if last_conn is None or conn_iso > last_conn:
                last_conn = conn_iso
        if i % progress_every == 0:
            sys.stderr.write(f"  [{site}] {i} sessions...\n")
            sys.stderr.flush()

    out_path = out_dir / f"{site}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(sessions, fh, default=_json_default, ensure_ascii=False, indent=None)

    stats = {
        "site": site,
        "total_sessions": len(sessions),
        "claimed_sessions": claimed,
        "unclaimed_sessions": len(sessions) - claimed,
        "first_connection": first_conn,
        "last_connection": last_conn,
        "observed_fields": sorted(observed_fields),
        "output_file": out_path.name,
    }
    sys.stderr.write(
        f"[{site}] wrote {len(sessions)} sessions "
        f"({claimed} claimed) -> {out_path}\n"
    )
    return stats


def _acnportal_version() -> str:
    try:
        import acnportal  # noqa: WPS433

        return getattr(acnportal, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites", nargs="+", default=["caltech", "jpl"],
        help="ACN sites to download (choices: caltech, jpl, office001).",
    )
    parser.add_argument("--out-dir", default="data/raw", help="Output directory.")
    parser.add_argument("--start", default=None, help="ISO start (connectionTime >=).")
    parser.add_argument("--end", default=None, help="ISO end (connectionTime <=).")
    parser.add_argument(
        "--min-energy", type=float, default=None,
        help="Only sessions with kWhDelivered > this value.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL.")
    parser.add_argument(
        "--token", default=None,
        help="API token (DISCOURAGED; prefer ACNDATA_API_TOKEN env var).",
    )
    args = parser.parse_args(argv)

    if DataClient is None:
        sys.stderr.write(
            "ERROR: acnportal is not installed. Install it with:\n"
            "    pip install acnportal\n"
        )
        return 2

    bad = [s for s in args.sites if s not in VALID_SITES]
    if bad:
        sys.stderr.write(
            f"ERROR: invalid site(s) {bad}. Valid: {sorted(VALID_SITES)}\n"
        )
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = resolve_token(args.token)
    if not token:
        sys.stderr.write("ERROR: no API token provided.\n")
        return 2

    start = parse_cli_datetime(args.start)
    end = parse_cli_datetime(args.end)

    client = DataClient(api_token=token, url=args.url)

    manifest: Dict[str, Any] = {
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "api_url": args.url,
        "acnportal_version": _acnportal_version(),
        "query": {
            "start": args.start,
            "end": args.end,
            "min_energy": args.min_energy,
        },
        "sites": {},
    }

    for site in args.sites:
        stats = download_site(
            client, site, out_dir, start, end, args.min_energy
        )
        manifest["sites"][site] = stats

    manifest_path = out_dir / "download_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    sys.stderr.write(f"\nManifest written -> {manifest_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
