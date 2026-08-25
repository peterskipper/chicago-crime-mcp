"""Tests for the batch rollup over the JSONL logs.

Two kinds of claim here. The first is that the declared schema and the record
stay in step -- the failure otherwise is a column that reads NULL forever and
nobody notices. The second is that each report actually separates the thing it
claims to separate, which needs fixtures containing the *near misses*: a soft
resolve miss next to a successful one, an empty result next to a full one from
the same filters, a bug next to a teaching error.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

import pytest

from chicago_crime_mcp.telemetry import rollup
from chicago_crime_mcp.telemetry.record import CallRecord
from chicago_crime_mcp.telemetry.sink import TelemetryConfig, TelemetrySink


def _write(directory, records):
    """Write records into a log directory and return it."""
    sink = TelemetrySink(TelemetryConfig(log_dir=directory))
    for record in records:
        sink.write(record)
    sink.close()
    return directory


def _call(**kwargs) -> CallRecord:
    """Build a record with the fields these tests do not vary."""
    kwargs.setdefault("tool", "search_incidents")
    kwargs.setdefault("outcome", "ok")
    kwargs.setdefault("duration_ms", 10.0)
    kwargs.setdefault("ts", "2026-08-24T12:00:00+00:00")
    return CallRecord(**kwargs)


# --- the schema contract -----------------------------------------------------


def test_the_declared_schema_covers_every_record_field():
    """A field added to the record but not here would read NULL forever."""
    assert set(rollup.COLUMNS) == {f.name for f in fields(CallRecord)}


def test_every_record_field_round_trips_through_the_reader(tmp_path):
    """Not just present in the schema -- actually readable with its own type."""
    record = _call(
        outcome="empty",
        args={"types": ["BATTERY"], "limit": 50},
        session_id="s-1",
        client_id="c-1",
        transport="http",
        route_store="duckdb",
        route_tier="rollup",
        route_table="monthly",
        route_reason="month grain",
        route_elapsed_ms=4.5,
        row_count=0,
        truncated=False,
        cursor_issued=True,
        taxonomy_mode="comparable",
        warning_codes=["empty_result", "provisional"],
        result_bytes=1234,
        resolution_kind=None,
        error_code=None,
    )
    conn = rollup.connect(_write(tmp_path / "logs", [record]))
    row = rollup._rows(conn, "SELECT * FROM calls")[0]
    conn.close()

    assert row["warning_codes"] == ["empty_result", "provisional"]
    assert row["route_elapsed_ms"] == 4.5
    assert row["truncated"] is False
    assert row["cursor_issued"] is True
    assert row["result_bytes"] == 1234
    assert json.loads(row["args"]) == {"types": ["BATTERY"], "limit": 50}
    assert str(row["at"]).startswith("2026-08-24")


def test_multiple_days_are_read_as_one_view(tmp_path):
    """Retention is per file; analysis is over the glob."""
    directory = tmp_path / "logs"
    _write(directory, [_call(ts="2026-08-23T12:00:00+00:00")])
    _write(directory, [_call(ts="2026-08-24T12:00:00+00:00")])
    conn = rollup.connect(directory)
    assert conn.execute("SELECT count(*) FROM calls").fetchone()[0] == 2
    conn.close()


def test_a_single_file_can_be_read_directly(tmp_path):
    directory = _write(tmp_path / "logs", [_call()])
    conn = rollup.connect(directory / "calls-2026-08-24.jsonl")
    assert conn.execute("SELECT count(*) FROM calls").fetchone()[0] == 1
    conn.close()


def test_an_empty_directory_is_an_error_not_an_empty_report(tmp_path):
    """Silently reporting on nothing would read as 'all healthy'."""
    (tmp_path / "logs").mkdir()
    with pytest.raises(FileNotFoundError):
        rollup.connect(tmp_path / "logs")


# --- overview ----------------------------------------------------------------


def test_overview_separates_empty_from_error(tmp_path):
    """They mean different things and get fixed in different places."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _call(outcome="ok"),
                _call(outcome="empty"),
                _call(outcome="empty"),
                _call(outcome="error", error_code="unknown_value"),
            ],
        )
    )
    row = rollup.overview(conn).rows[0]
    conn.close()
    assert (row["calls"], row["empty"], row["errors"]) == (4, 2, 1)
    assert row["empty_rate"] == 0.5
    assert row["error_rate"] == 0.25


def test_overview_ranks_by_traffic(tmp_path):
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [_call(tool="quiet")] + [_call(tool="busy") for _ in range(3)],
        )
    )
    assert [r["tool"] for r in rollup.overview(conn).rows] == ["busy", "quiet"]
    conn.close()


# --- empty results by filter combination -------------------------------------


def test_empty_by_filters_groups_the_same_filters_regardless_of_order(tmp_path):
    """A combination is a set; two orderings are one row, not two."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _call(outcome="empty", args={"types": ["X"], "district": "031"}),
                _call(outcome="empty", args={"district": "031", "types": ["X"]}),
            ],
        )
    )
    rows = rollup.empty_by_filters(conn).rows
    conn.close()
    assert len(rows) == 1
    assert rows[0]["filters"] == ["district", "types"]
    assert rows[0]["empty"] == 2


def test_empty_by_filters_ignores_mechanism_arguments(tmp_path):
    """A page size does not explain why nothing matched."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _call(outcome="empty", args={"types": ["X"], "limit": 50}),
                _call(outcome="empty", args={"types": ["X"], "limit": 200, "cursor": "c"}),
            ],
        )
    )
    rows = rollup.empty_by_filters(conn).rows
    conn.close()
    assert len(rows) == 1, "limit split one real combination into two"
    assert rows[0]["filters"] == ["types"]


def test_empty_by_filters_reports_a_rate_not_only_a_count(tmp_path):
    """Always-empty and sometimes-empty are different findings."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _call(outcome="empty", args={"types": ["NEVER"]}),
                _call(outcome="empty", args={"types": ["NEVER"]}),
                _call(outcome="empty", args={"district": "031"}),
                _call(outcome="ok", args={"district": "031"}),
                _call(outcome="ok", args={"district": "031"}),
            ],
        )
    )
    rows = rollup.empty_by_filters(conn).rows
    conn.close()
    assert rows[0]["filters"] == ["types"]
    assert rows[0]["empty_rate"] == 1.0
    assert rows[1]["empty_rate"] < 1.0


def test_empty_by_filters_omits_combinations_that_never_came_back_empty(tmp_path):
    """The negative half: a healthy combination is not a finding."""
    conn = rollup.connect(
        _write(tmp_path / "logs", [_call(outcome="ok", args={"types": ["BATTERY"]})])
    )
    assert rollup.empty_by_filters(conn).rows == []
    conn.close()


def test_empty_by_filters_honours_min_calls(tmp_path):
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _call(outcome="empty", args={"types": ["ONE"]}),
                _call(outcome="empty", args={"district": "031"}),
                _call(outcome="empty", args={"district": "031"}),
            ],
        )
    )
    rows = rollup.empty_by_filters(conn, min_calls=2).rows
    conn.close()
    assert [r["filters"] for r in rows] == [["district"]]


def test_empty_by_filters_excludes_errored_calls(tmp_path):
    """A rejected call never ran a query, so it is not evidence about filters."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _call(outcome="error", args={"types": ["BATERY"]}, error_code="unknown_value"),
                _call(outcome="empty", args={"types": ["BATERY"]}),
            ],
        )
    )
    rows = rollup.empty_by_filters(conn).rows
    conn.close()
    assert rows[0]["calls"] == 1


# --- the alias backlog -------------------------------------------------------


def _resolve(name, kind, **kwargs):
    """A resolve_neighborhood record."""
    outcome = "error" if kind == "none" else "ok"
    return _call(
        tool="resolve_neighborhood",
        outcome=outcome,
        args={"name": name},
        resolution_kind=kind,
        **kwargs,
    )


def test_resolution_misses_counts_soft_misses_that_succeeded(tmp_path):
    """The whole point: a suggestion is a successful call and an unmet need."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [_resolve("Bronzeville", "suggestion") for _ in range(3)],
        )
    )
    rows = rollup.resolution_misses(conn).rows
    conn.close()
    assert rows[0]["name"] == "Bronzeville"
    assert rows[0]["asked"] == 3


def test_resolution_misses_excludes_names_that_resolved(tmp_path):
    """The negative half, and the one that matters: a hit is not a backlog item.

    'containing' in particular is a *success* -- a real place answered by the
    community area holding it -- and putting it on the backlog would send
    somebody off to write an alias that already works.
    """
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _resolve("Wicker Park", "exact"),
                _resolve("Lakeview", "alias"),
                _resolve("Pilsen", "containing"),
                _resolve("Bronzeville", "suggestion"),
            ],
        )
    )
    rows = rollup.resolution_misses(conn).rows
    conn.close()
    assert [r["name"] for r in rows] == ["Bronzeville"]


def test_resolution_misses_counts_distinct_sessions(tmp_path):
    """One person asking ten times is weaker evidence than ten people asking once."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _resolve("Bronzeville", "suggestion", session_id="s1"),
                _resolve("Bronzeville", "suggestion", session_id="s1"),
                _resolve("Bronzeville", "suggestion", session_id="s2"),
            ],
        )
    )
    row = rollup.resolution_misses(conn).rows[0]
    conn.close()
    assert (row["asked"], row["sessions"]) == (3, 2)


def test_resolution_misses_ranks_by_demand(tmp_path):
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [_resolve("Rare", "none")] + [_resolve("Common", "suggestion") for _ in range(4)],
        )
    )
    assert [r["name"] for r in rollup.resolution_misses(conn).rows] == ["Common", "Rare"]
    conn.close()


def test_the_backlog_is_written_as_an_alias_file_stub(tmp_path):
    """Shaped for a diff against the real reference file, not for retyping."""
    conn = rollup.connect(
        _write(tmp_path / "logs", [_resolve("Bronzeville", "suggestion") for _ in range(2)])
    )
    rows = rollup.alias_backlog_rows(rollup.resolution_misses(conn))
    conn.close()

    assert list(rows[0]) == list(rollup.ALIAS_COLUMNS)
    assert rows[0]["alias"] == "Bronzeville"
    assert "asked 2x" in rows[0]["note"]
    # The target is a judgement about Chicago; the job must not guess at it.
    assert rows[0]["target_value"] == ""
    assert rows[0]["match_kind"] == ""


def test_the_backlog_stub_matches_the_real_reference_files_columns():
    """If the reference file gains a column, the stub stops being diffable."""
    from chicago_crime_mcp import reference

    path = Path(reference.__file__).parent / "neighborhood_aliases.csv"
    with path.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == list(rollup.ALIAS_COLUMNS)


# --- malformed arguments -----------------------------------------------------


def test_arg_failures_collects_the_invented_values(tmp_path):
    """The values are the useful part -- they name the missing synonym."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _call(outcome="error", error_code="unknown_value", error_field="types",
                      error_received="BATERY", error_nearest_match="BATTERY"),
                _call(outcome="error", error_code="unknown_value", error_field="types",
                      error_received="ASSUALT"),
            ],
        )
    )
    row = rollup.arg_failures(conn).rows[0]
    conn.close()
    assert row["field"] == "types"
    assert row["errors"] == 2
    assert set(row["invented_values"]) == {"BATERY", "ASSUALT"}
    assert row["suggested"] == 1


def test_arg_failures_keeps_bugs_separable_from_teaching_errors(tmp_path):
    """A self-correcting loop and a crash must not add up into one number."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _call(outcome="error", error_code="unknown_value", error_field="types"),
                _call(outcome="error", error_code="unhandled"),
            ],
        )
    )
    codes = {r["error_code"] for r in rollup.arg_failures(conn).rows}
    conn.close()
    assert codes == {"unknown_value", "unhandled"}


def test_arg_failures_ignores_successful_calls(tmp_path):
    conn = rollup.connect(_write(tmp_path / "logs", [_call(outcome="ok")]))
    assert rollup.arg_failures(conn).rows == []
    conn.close()


# --- latency and warnings ----------------------------------------------------


def test_latency_separates_our_overhead_from_the_query(tmp_path):
    """A slow serializer and a slow database want opposite fixes."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [_call(duration_ms=100.0, route_elapsed_ms=30.0, route_store="postgres")],
        )
    )
    row = rollup.latency_by_route(conn).rows[0]
    conn.close()
    assert row["p50_ms"] == 100.0
    assert row["overhead_ms"] == 70.0


def test_latency_groups_by_route_not_by_tool(tmp_path):
    """One tool can take two routes; the route is what explains the time."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [
                _call(tool="aggregate_incidents", route_store="duckdb", route_tier="rollup"),
                _call(tool="aggregate_incidents", route_store="duckdb", route_tier="scan"),
            ],
        )
    )
    rows = rollup.latency_by_route(conn).rows
    conn.close()
    assert {r["tier"] for r in rows} == {"rollup", "scan"}


def test_warning_frequency_reports_a_share_of_all_calls(tmp_path):
    """A count alone cannot say whether a warning has become background noise."""
    conn = rollup.connect(
        _write(
            tmp_path / "logs",
            [_call(warning_codes=["provisional"]) for _ in range(3)] + [_call()],
        )
    )
    row = rollup.warning_frequency(conn).rows[0]
    conn.close()
    assert (row["code"], row["attached"], row["share_of_calls"]) == ("provisional", 3, 0.75)


# --- rendering and the CLI ---------------------------------------------------


def test_render_says_so_when_a_report_is_empty():
    """A blank section reads as 'not run'; it should read as 'nothing found'."""
    text = rollup.render([rollup.Report(title="T", note="n", rows=[])])
    assert "(nothing)" in text


def test_render_shows_null_as_a_dash_not_as_the_word_none():
    text = rollup.render([rollup.Report(title="T", note="n", rows=[{"a": None}])])
    assert "None" not in text
    assert "-" in text


def test_the_cli_reports_and_writes_a_backlog(tmp_path, capsys):
    directory = _write(
        tmp_path / "logs",
        [_resolve("Bronzeville", "suggestion"), _call(outcome="empty", args={"types": ["X"]})],
    )
    backlog = tmp_path / "out" / "backlog.csv"
    code = rollup.main([str(directory), "--alias-backlog", str(backlog)])
    out = capsys.readouterr().out

    assert code == 0
    assert "2 tool call(s)" in out
    assert "Bronzeville" in out
    with backlog.open(encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert [r["alias"] for r in written] == ["Bronzeville"]


def test_the_cli_fails_cleanly_when_there_are_no_logs(tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    assert rollup.main([str(tmp_path / "logs")]) == 1
    assert "no telemetry log files" in capsys.readouterr().err


def test_the_cli_falls_back_to_the_configured_log_directory(tmp_path, monkeypatch, capsys):
    """Running it with no argument on the box should just work."""
    directory = _write(tmp_path / "logs", [_call()])
    monkeypatch.setenv("TELEMETRY_LOG_DIR", str(directory))
    assert rollup.main([]) == 0
    assert "1 tool call(s)" in capsys.readouterr().out
