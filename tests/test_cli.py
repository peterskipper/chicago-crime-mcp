"""Unit tests for the ingestion CLI.

Argument parsing is tested directly; command dispatch is tested with the
SodaClient and the ingest routines monkeypatched, so nothing hits the network.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pytest

from chicago_crime_mcp.ingest import cli


class _DummyClient:
    """Stand-in for SodaClient's context manager (no network)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _no_client_or_dotenv(monkeypatch):
    """Neutralize network client construction and .env loading in every test."""
    monkeypatch.setattr(cli, "SodaClient", lambda *a, **k: _DummyClient())
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)


# -- parsing ---------------------------------------------------------------


def test_backfill_defaults_to_working_window():
    args = cli.build_parser().parse_args(["backfill"])
    assert (args.command, args.start_year, args.end_year, args.force) == (
        "backfill", cli.DEFAULT_START_YEAR, cli.DEFAULT_END_YEAR, False,
    )


def test_backfill_end_year_tracks_the_calendar():
    """The default end year must follow the clock, not a pinned literal.

    Regression guard: a hardcoded end year silently stops managing the current
    year when the calendar rolls over, and `incremental` only refreshes years
    that already have a partition -- so the gap goes unnoticed until someone
    asks for this year's data and gets nothing.
    """
    assert cli.DEFAULT_END_YEAR == date.today().year
    assert cli.DEFAULT_START_YEAR < cli.DEFAULT_END_YEAR


def test_backfill_accepts_overrides():
    args = cli.build_parser().parse_args(
        ["backfill", "--start-year", "2021", "--end-year", "2022", "--force"]
    )
    assert (args.start_year, args.end_year, args.force) == (2021, 2022, True)


def test_incremental_parses_overlap():
    args = cli.build_parser().parse_args(["incremental", "--overlap-days", "3"])
    assert args.command == "incremental"
    assert args.overlap_days == 3


def test_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


# -- dispatch --------------------------------------------------------------


def test_main_dispatches_backfill(monkeypatch, caplog):
    calls = {}

    def fake_backfill(client, start, end, force=False):
        calls["args"] = (start, end, force)
        return [
            {"year": 2021, "rows": 100, "expected": 100, "skipped": False},
            {"year": 2022, "rows": 50, "expected": 50, "skipped": True},
        ]

    monkeypatch.setattr(cli.backfill_mod, "backfill", fake_backfill)
    with caplog.at_level(logging.INFO, logger="chicago_crime_mcp.ingest.cli"):
        cli.main(["backfill", "--start-year", "2021", "--end-year", "2022"])

    assert calls["args"] == (2021, 2022, False)
    assert "150 rows across 2 years" in caplog.text
    assert "2022: 50 rows (skipped (complete))" in caplog.text


def test_main_dispatches_incremental(monkeypatch, caplog):
    calls = {}

    def fake_sync(client, overlap=None):
        calls["overlap"] = overlap
        return {
            "pulled": 5, "cutoff": "2026-07-19T00:00:00",
            "years": {2023: {"applied": 5, "partition_rows": 42}},
            "skipped_unmanaged": 2, "watermark": "2026-07-21T00:00:00",
        }

    monkeypatch.setattr(cli.incremental_mod, "incremental_sync", fake_sync)
    with caplog.at_level(logging.INFO, logger="chicago_crime_mcp.ingest.cli"):
        cli.main(["incremental", "--overlap-days", "3"])

    assert calls["overlap"] == timedelta(days=3)
    assert "pulled 5 rows" in caplog.text
    assert "2023: applied 5 -> 42 rows" in caplog.text
    assert "skipped 2 rows for unmanaged years" in caplog.text
    assert "watermark now 2026-07-21T00:00:00" in caplog.text
