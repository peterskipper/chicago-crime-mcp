"""Tests for the server's connection lifecycle.

These exercise the **DuckDB half** of :class:`ServerContext` directly, through
``_open_duckdb``, because the public ``open()`` also builds a Postgres pool and
that needs a live server. The DuckDB half is where all the behaviour worth
testing lives -- the inode swap, the vocabulary cache, the fail-loud checks --
and DuckDB is embedded, so these stay unit tests.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime

import pytest

from chicago_crime_mcp.server.context import (
    CHECKED_RELATIONS,
    ServerContext,
    get_context,
    set_context,
    use_context,
)
from chicago_crime_mcp.server.errors import DataUnavailableError
from chicago_crime_mcp.store.config import StoreConfig
from chicago_crime_mcp.store.duckdb import rollups
from tests.helpers import row as _row
from tests.helpers import write_partition as _write_partition


def _build(base, rows):
    """Build a rollup database over ``rows`` and return its config.

    Args:
        base: Directory to build under.
        rows: Incident dicts for a single 2024 partition.

    Returns:
        A :class:`StoreConfig` pointing at the built database.
    """
    _write_partition(base / "parquet", 2024, rows)
    path = base / "db" / "crime.duckdb"
    conn = rollups.connect(duckdb_path=path, parquet_root=base / "parquet")
    rollups.build(conn)
    conn.close()
    return StoreConfig(duckdb_path=path, parquet_root=base / "parquet")


@pytest.fixture
def context(tmp_path):
    """An open DuckDB-only context over a two-row fixture dataset."""
    config = _build(
        tmp_path,
        [
            _row(id=1, date=datetime(2024, 1, 5), community_area=1),
            _row(id=2, date=datetime(2024, 2, 6), primary_type_canonical="THEFT",
                 stable_category="THEFT", community_area=2, district="011"),
        ],
    )
    ctx = ServerContext(config)
    ctx._open_duckdb()
    yield ctx
    ctx.close()


# --- the context provider ---------------------------------------------------


def test_get_context_without_one_is_a_teaching_error():
    """A tool called before startup finished says so, rather than crashing."""
    set_context(None)
    with pytest.raises(DataUnavailableError) as exc:
        get_context()
    assert "not a problem with the request" in str(exc.value)


def test_use_context_restores_the_previous_one(context):
    """Nesting is safe, so a test never leaks its context into the next one."""
    set_context(None)
    with use_context(context):
        assert get_context() is context
    with pytest.raises(DataUnavailableError):
        get_context()


# --- the vocabulary cache ---------------------------------------------------


def test_vocabulary_is_cached_between_calls(context):
    """Loaded once per build, not once per tool call."""
    assert context.vocabulary() is context.vocabulary()


def test_vocabulary_reads_the_data_not_a_constant(context):
    """Categories and geography values come from the fixture, not a hardcoded list."""
    vocabulary = context.vocabulary()
    assert vocabulary.categories_for("source") == ("BATTERY", "THEFT")
    assert vocabulary.values_for("district") == ("010", "011")
    assert vocabulary.values_for("community_area") == (1, 2)
    # citywide has no column, so it has no values -- not an oversight.
    assert vocabulary.values_for("citywide") == ()


def test_vocabulary_distinguishes_the_two_taxonomies(tmp_path):
    """A curated remap shows up under 'comparable' and not under 'source'."""
    config = _build(
        tmp_path,
        [
            _row(id=1, date=datetime(2024, 1, 5), primary_type_canonical="BURGLARY",
                 stable_category="THEFT"),
        ],
    )
    ctx = ServerContext(config)
    ctx._open_duckdb()
    try:
        assert ctx.vocabulary().categories_for("source") == ("BURGLARY",)
        assert ctx.vocabulary().categories_for("comparable") == ("THEFT",)
    finally:
        ctx.close()


# --- the nightly file swap --------------------------------------------------


def test_swapping_the_database_file_reopens_and_reloads(context, tmp_path):
    """The rebuild replaces the file; the next call must see the new build.

    This is the failure the inode check exists for. Without it the server holds
    the old inode after ``os.replace`` and serves correct-but-stale data
    indefinitely -- silently, because nothing errors.
    """
    before = context.vocabulary()
    assert "ARSON" not in before.categories_for("source")

    replacement = _build(
        tmp_path / "next",
        [_row(id=3, date=datetime(2024, 3, 7), primary_type_canonical="ARSON",
              stable_category="ARSON")],
    )
    os.replace(replacement.duckdb_path, context.config.duckdb_path)

    after = context.vocabulary()
    assert after is not before
    assert after.categories_for("source") == ("ARSON",)


def test_swap_waits_for_an_in_flight_cursor(context, tmp_path):
    """A reader holding a cursor blocks the swap until it finishes.

    The swap must close the old connection -- DuckDB hands back the same cached
    instance otherwise -- and closing it would invalidate a cursor a running
    request is still reading from. So the swap drains first. This asserts the
    reader is not disturbed and the swap lands once it lets go.
    """
    replacement = _build(tmp_path / "next", [_row(id=3, date=datetime(2024, 3, 7),
                                                  primary_type_canonical="ARSON",
                                                  stable_category="ARSON")])
    swapped = threading.Event()
    finished = threading.Event()

    with context.duckdb() as reader:
        def swap():
            os.replace(replacement.duckdb_path, context.config.duckdb_path)
            swapped.set()
            context.vocabulary()  # blocks until the reader below lets go
            finished.set()

        thread = threading.Thread(target=swap)
        thread.start()
        assert swapped.wait(5)
        # The reader's cursor keeps working while the swap is pending.
        for _ in range(20):
            assert reader.execute("SELECT count(*) FROM rollup_citywide").fetchone()[0] == 2
        assert not finished.is_set(), "swap closed the connection out from under a reader"

    thread.join(timeout=10)
    assert finished.is_set()
    assert context.vocabulary().categories_for("source") == ("ARSON",)


def test_unchanged_file_does_not_reopen(context):
    """No swap, no churn -- the connection and its cache survive."""
    first = context.vocabulary()
    inode = context._duck_inode
    with context.duckdb() as conn:
        conn.execute("SELECT 1").fetchone()
    assert context.vocabulary() is first
    assert context._duck_inode == inode


def test_readers_are_released_even_when_a_tool_raises(context):
    """A failed query must not leave a reader counted, or the next swap hangs."""
    import duckdb

    with pytest.raises(duckdb.Error):
        with context.duckdb() as conn:
            conn.execute("SELECT * FROM no_such_table")
    assert context._readers == 0


def test_missing_file_during_a_swap_keeps_the_open_connection(context):
    """The instant between unlink and rename must not fail a request.

    ``_current_inode`` reports the last known inode when the path does not
    resolve, so a request landing inside that window keeps serving from the
    connection it already has.
    """
    context.config.duckdb_path.unlink()
    assert context._current_inode() == context._duck_inode
    assert context.vocabulary() is not None


# --- failing loud -----------------------------------------------------------


def test_missing_database_is_reported_at_startup(tmp_path):
    """An unbuilt database fails where an operator sees it, naming the remedy."""
    ctx = ServerContext(StoreConfig(duckdb_path=tmp_path / "absent.duckdb"))
    with pytest.raises(DataUnavailableError) as exc:
        ctx._open_duckdb()
    assert "chicago-crime-rollup" in str(exc.value)


def test_database_without_rollups_is_reported_at_startup(tmp_path):
    """A file that exists but was never built is caught by the relation check."""
    import duckdb

    path = tmp_path / "empty.duckdb"
    duckdb.connect(str(path)).close()
    ctx = ServerContext(StoreConfig(duckdb_path=path))
    with pytest.raises(DataUnavailableError) as exc:
        ctx._open_duckdb()
    assert "chicago-crime-rollup" in str(exc.value)


def test_checked_relations_covers_every_rollup_table():
    """A rollup table added to the builder is checked here without a second list."""
    assert set(rollups.ROLLUP_TABLES) <= set(CHECKED_RELATIONS)
    assert {rollups.CODE_MONTH_TABLE, rollups.COVERAGE_TABLE, "rollup_meta"} <= set(
        CHECKED_RELATIONS
    )


def test_duckdb_connection_is_read_only(context):
    """A tool cannot mutate the rollups, structurally rather than by convention."""
    import duckdb

    with context.duckdb() as conn:
        with pytest.raises(duckdb.Error):
            conn.execute("CREATE TABLE scribble (x INTEGER)")
