"""Offline tests for download_acndata serialization and claimed-detection.

These run WITHOUT API access. They build synthetic sessions carrying tz-aware
datetime objects identical in shape to what acnportal's parse_dates produces,
and verify the download script serializes them correctly.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import pytz

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import download_acndata as dl  # noqa: E402


def _synthetic_session(user_id, site="caltech", tz_name="America/Los_Angeles"):
    """Mimic a parsed ACN session: timestamps are tz-aware datetimes."""
    tz = pytz.timezone(tz_name)
    conn = tz.localize(dt.datetime(2018, 5, 1, 8, 23, 0))
    done = tz.localize(dt.datetime(2018, 5, 1, 11, 5, 0))
    disc = tz.localize(dt.datetime(2018, 5, 1, 17, 42, 0))
    return {
        "_id": "abc123",
        "sessionID": "2_39_target_session",
        "userID": user_id,
        "siteID": site,
        "stationID": "2-39-104",
        "connectionTime": conn,
        "doneChargingTime": done,
        "disconnectTime": disc,
        "kWhDelivered": 12.34,
        "timezone": tz_name,
        # Nested string-date field the client leaves as a string:
        "userInputs": [{"requestedDeparture": "Tue, 01 May 2018 18:00:00 GMT"}],
    }


def test_datetime_serialization_roundtrips():
    """tz-aware datetimes serialize to ISO-8601 with offset and round-trip."""
    session = _synthetic_session("user_042")
    encoded = json.dumps(session, default=dl._json_default)
    decoded = json.loads(encoded)
    # Offset preserved (Pacific in May = -07:00 DST):
    assert decoded["connectionTime"] == "2018-05-01T08:23:00-07:00"
    assert decoded["disconnectTime"] == "2018-05-01T17:42:00-07:00"
    # A tz-aware datetime survives a full parse back:
    reparsed = dt.datetime.fromisoformat(decoded["connectionTime"])
    assert reparsed.tzinfo is not None
    assert reparsed.utcoffset() == dt.timedelta(hours=-7)


def test_nested_string_dates_untouched():
    """String dates the client didn't convert must still serialize fine."""
    session = _synthetic_session("user_042")
    encoded = json.dumps(session, default=dl._json_default)
    decoded = json.loads(encoded)
    assert decoded["userInputs"][0]["requestedDeparture"].endswith("GMT")


def test_claimed_detection():
    assert dl._is_claimed(_synthetic_session("user_1")) is True
    assert dl._is_claimed(_synthetic_session(None)) is False
    assert dl._is_claimed(_synthetic_session("")) is False
    assert dl._is_claimed({"foo": "bar"}) is False


def test_naive_datetime_raises_on_default():
    """Non-datetime unserializable objects still raise (no silent pass)."""
    try:
        dl._json_default(object())
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected TypeError for unserializable object")


def test_cli_datetime_attaches_utc():
    parsed = dl.parse_cli_datetime("2019-01-01")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == dt.timedelta(0)
    assert dl.parse_cli_datetime(None) is None


def test_download_site_writes_file_and_stats(tmp_path, monkeypatch):
    """Simulate a DataClient and verify file output + manifest stats."""

    class FakeClient:
        def get_sessions(self, site, sort=None, timeseries=False):
            yield _synthetic_session("user_A", site=site)
            yield _synthetic_session("user_B", site=site)
            yield _synthetic_session(None, site=site)  # unclaimed

    stats = dl.download_site(
        FakeClient(), "caltech", tmp_path,
        start=None, end=None, min_energy=None,
    )
    assert stats["total_sessions"] == 3
    assert stats["claimed_sessions"] == 2
    assert stats["unclaimed_sessions"] == 1
    assert stats["first_connection"] == "2018-05-01T08:23:00-07:00"

    out_file = tmp_path / "caltech.json"
    assert out_file.exists()
    loaded = json.loads(out_file.read_text())
    assert len(loaded) == 3
    assert loaded[0]["connectionTime"] == "2018-05-01T08:23:00-07:00"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
