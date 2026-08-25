"""Tests for the per-call log record.

Two properties matter here and neither is about any single field. The record
must fill its own identity so no caller can forget to, and it must serialize
*every* field on every call -- including the ones that are None -- because the
rollup declares one schema over a glob of daily files and a key that appears
only on days when some branch fired would make that schema a lie.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import json
from dataclasses import fields

from chicago_crime_mcp.telemetry.record import CallRecord


def _record(**kwargs) -> CallRecord:
    """Build a record, defaulting the three required fields."""
    kwargs.setdefault("tool", "search_incidents")
    kwargs.setdefault("outcome", "ok")
    kwargs.setdefault("duration_ms", 1.0)
    return CallRecord(**kwargs)


def test_identity_is_generated_when_not_supplied():
    """A caller should not have to remember a timestamp and an id."""
    record = _record()
    assert record.ts.startswith("20")
    assert record.ts.endswith("+00:00"), "timestamps must be explicitly UTC"
    assert len(record.call_id) == 32


def test_call_ids_are_unique_per_record():
    """The join key has to actually distinguish calls."""
    assert len({_record().call_id for _ in range(50)}) == 50


def test_supplied_identity_is_kept():
    """Generation is a default, not an override -- tests need to pin both."""
    record = _record(ts="2026-08-24T12:00:00+00:00", call_id="fixed")
    assert (record.ts, record.call_id) == ("2026-08-24T12:00:00+00:00", "fixed")


def test_every_field_is_serialized_even_when_none():
    """The flat-schema contract: absent keys would break the rollup's glob.

    Asserted against the dataclass's own field list rather than a copied
    literal, so a field added to the record cannot silently go unlogged.
    """
    payload = _record().to_dict()
    assert set(payload) == {f.name for f in fields(CallRecord)}
    assert payload["route_store"] is None
    assert payload["error_code"] is None


def test_the_record_survives_a_json_round_trip():
    """It is written with json.dumps; nothing in it may be unserializable."""
    record = _record(
        args={"types": ["BATTERY"], "limit": 50},
        warning_codes=["provisional", "truncated"],
        route_store="duckdb",
    )
    restored = json.loads(json.dumps(record.to_dict(), default=str))
    assert restored["args"] == {"types": ["BATTERY"], "limit": 50}
    assert restored["warning_codes"] == ["provisional", "truncated"]


def test_two_records_do_not_share_a_mutable_default():
    """``args`` and ``warning_codes`` are per-record, not class-wide."""
    first, second = _record(), _record()
    first.args["x"] = 1
    first.warning_codes.append("truncated")
    assert second.args == {}
    assert second.warning_codes == []


# --- the one-line human form -------------------------------------------------


def test_summary_of_a_successful_call_names_rows_and_route():
    line = _record(row_count=42, route_store="duckdb", route_tier="rollup").summary()
    assert "search_incidents ok" in line
    assert "42 rows" in line
    assert "duckdb/rollup" in line
    assert "truncated" not in line


def test_summary_flags_truncation_only_when_truncated():
    """The negative half: an untruncated page must not claim to be capped."""
    capped = _record(row_count=50, truncated=True).summary()
    whole = _record(row_count=50, truncated=False).summary()
    assert "truncated" in capped
    assert "truncated" not in whole


def test_summary_of_an_error_names_the_code_and_field_instead_of_rows():
    line = _record(
        outcome="error", error_code="unknown_value", error_field="types", row_count=None
    ).summary()
    assert "unknown_value:types" in line
    assert "rows" not in line


def test_summary_omits_the_tier_for_a_store_that_has_none():
    """Postgres has one tier, so naming one would be inventing a fact."""
    assert "postgres " in _record(route_store="postgres").summary()
    assert "postgres/" not in _record(route_store="postgres").summary()
