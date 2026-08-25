"""Tests for the JSONL sink: file layout, rotation, and refusing to fail loudly.

The sink has one hard obligation beyond writing correctly -- it must never be
the reason a tool call fails. Half these tests are about what happens when the
filesystem does not cooperate.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import json
import threading

import pytest

from chicago_crime_mcp.telemetry.record import CallRecord
from chicago_crime_mcp.telemetry.sink import TelemetryConfig, TelemetrySink


def _record(day: str = "2026-08-24", **kwargs) -> CallRecord:
    """Build a record pinned to a given day."""
    kwargs.setdefault("tool", "search_incidents")
    kwargs.setdefault("outcome", "ok")
    kwargs.setdefault("duration_ms", 1.0)
    kwargs.setdefault("ts", f"{day}T12:00:00+00:00")
    return CallRecord(**kwargs)


@pytest.fixture
def sink(tmp_path):
    """A sink writing into a temporary directory."""
    s = TelemetrySink(TelemetryConfig(enabled=True, log_dir=tmp_path / "telemetry"))
    yield s
    s.close()


# --- configuration -----------------------------------------------------------


def test_config_defaults_need_no_environment():
    """A fresh checkout with no .env still logs, like StoreConfig."""
    config = TelemetryConfig.from_env({})
    assert config.enabled is True
    assert config.log_dir.as_posix() == "data/telemetry"


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " Off "])
def test_config_disables_on_falsey_spellings(raw):
    assert TelemetryConfig.from_env({"TELEMETRY_ENABLED": raw}).enabled is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "anything"])
def test_config_stays_enabled_otherwise(raw):
    """The negative half: only the listed spellings turn it off."""
    assert TelemetryConfig.from_env({"TELEMETRY_ENABLED": raw}).enabled is True


def test_config_reads_the_directory_override():
    config = TelemetryConfig.from_env({"TELEMETRY_LOG_DIR": "/var/log/crime"})
    assert config.log_dir.as_posix() == "/var/log/crime"


# --- writing -----------------------------------------------------------------


def test_nothing_is_created_until_something_is_written(tmp_path):
    """A server that is never called leaves no empty artifacts behind."""
    directory = tmp_path / "telemetry"
    TelemetrySink(TelemetryConfig(log_dir=directory))
    assert not directory.exists()


def test_each_record_is_one_json_line(sink):
    sink.write(_record(tool="a"))
    sink.write(_record(tool="b"))
    lines = sink.path_for("2026-08-24").read_text().splitlines()
    assert [json.loads(line)["tool"] for line in lines] == ["a", "b"]


def test_the_records_own_day_decides_the_file_not_the_clock(sink):
    """A call at 23:59:59.9 belongs to the day it happened on."""
    sink.write(_record(day="2026-08-24"))
    sink.write(_record(day="2026-08-25"))
    assert sink.path_for("2026-08-24").exists()
    assert sink.path_for("2026-08-25").exists()
    assert len(sink.path_for("2026-08-24").read_text().splitlines()) == 1


def test_reopening_a_day_appends_rather_than_truncates(tmp_path):
    """A restarted server must not erase the morning's records."""
    config = TelemetryConfig(log_dir=tmp_path / "telemetry")
    first = TelemetrySink(config)
    first.write(_record(tool="before"))
    first.close()

    second = TelemetrySink(config)
    second.write(_record(tool="after"))
    second.close()

    lines = second.path_for("2026-08-24").read_text().splitlines()
    assert [json.loads(line)["tool"] for line in lines] == ["before", "after"]


def test_records_are_readable_while_the_sink_is_still_open(sink):
    """Flushed per write, so a rollup can read a file the server is writing."""
    sink.write(_record())
    assert sink.path_for("2026-08-24").read_text().endswith("\n")


def test_disabled_sink_writes_no_file_but_still_logs(tmp_path, caplog):
    """Turning off the file must not turn off the operator's stderr line."""
    directory = tmp_path / "telemetry"
    disabled = TelemetrySink(TelemetryConfig(enabled=False, log_dir=directory))
    with caplog.at_level("INFO", logger="chicago_crime_mcp.telemetry.sink"):
        disabled.write(_record())
    assert not directory.exists()
    assert any("search_incidents ok" in r.getMessage() for r in caplog.records)


def test_concurrent_writers_do_not_interleave_lines(sink):
    """Every line must still parse: the lock is what makes that true."""
    threads = [
        threading.Thread(target=lambda i=i: sink.write(_record(tool=f"t{i}")))
        for i in range(40)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = sink.path_for("2026-08-24").read_text().splitlines()
    assert len(lines) == 40
    assert {json.loads(line)["tool"] for line in lines} == {f"t{i}" for i in range(40)}


# --- failing safely ----------------------------------------------------------


def test_a_write_failure_does_not_raise_and_disables_the_sink(tmp_path, caplog):
    """Telemetry is never the reason a tool call fails.

    The directory path is occupied by a *file*, so ``mkdir`` raises -- the
    realistic shape of a bad TELEMETRY_LOG_DIR or a read-only volume.
    """
    blocker = tmp_path / "telemetry"
    blocker.write_text("not a directory")
    sink = TelemetrySink(TelemetryConfig(log_dir=blocker))

    with caplog.at_level("WARNING", logger="chicago_crime_mcp.telemetry.sink"):
        sink.write(_record())  # must not raise

    assert blocker.is_file(), "the sink must not have clobbered what was in its way"
    assert any("sink disabled" in r.getMessage() for r in caplog.records)


def test_a_broken_sink_warns_once_not_per_call(tmp_path, caplog):
    """A sink failing on every call would bury the errors it exists to surface."""
    blocker = tmp_path / "telemetry"
    blocker.write_text("not a directory")
    sink = TelemetrySink(TelemetryConfig(log_dir=blocker))

    with caplog.at_level("WARNING", logger="chicago_crime_mcp.telemetry.sink"):
        for _ in range(5):
            sink.write(_record())

    warnings = [r for r in caplog.records if "sink disabled" in r.getMessage()]
    assert len(warnings) == 1


def test_close_is_idempotent(sink):
    sink.write(_record())
    sink.close()
    sink.close()
