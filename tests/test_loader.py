"""Tests for the Postgres bulk loader.

The Parquet-reading helpers are pure and always run. The full ``load`` round-trip
needs a live Postgres and is marked ``integration`` (skipped by default, and
auto-skipped if no database is reachable).
"""

from __future__ import annotations

from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chicago_crime_mcp.store.postgres import loader
from tests.helpers import SCHEMA
from tests.helpers import row as _row
from tests.helpers import write_partition as _write_partition

# --- unit tests: Parquet reading (no database) ------------------------------


def test_discover_partitions_sorted_by_year(tmp_path):
    _write_partition(tmp_path, 2021, [_row()])
    _write_partition(tmp_path, 2019, [_row()])
    _write_partition(tmp_path, 2020, [_row()])
    found = loader.discover_partitions(tmp_path)
    assert [p.parent.name for p in found] == ["year=2019", "year=2020", "year=2021"]


def test_discover_partitions_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.discover_partitions(tmp_path)


def test_partition_columns_match_file(tmp_path):
    path = _write_partition(tmp_path, 2025, [_row()])
    assert loader.partition_columns(path) == SCHEMA.names


def test_select_partitions_filters_by_year(tmp_path):
    _write_partition(tmp_path, 2024, [_row()])
    _write_partition(tmp_path, 2025, [_row()])
    _write_partition(tmp_path, 2026, [_row()])
    sel = loader.select_partitions(tmp_path, years=[2025, 2026])
    assert [p.parent.name for p in sel] == ["year=2025", "year=2026"]


def test_select_partitions_none_returns_all(tmp_path):
    _write_partition(tmp_path, 2024, [_row()])
    _write_partition(tmp_path, 2025, [_row()])
    assert len(loader.select_partitions(tmp_path)) == 2


def test_select_partitions_unknown_year_omitted(tmp_path):
    _write_partition(tmp_path, 2025, [_row()])
    assert loader.select_partitions(tmp_path, years=[2099]) == []


def test_iter_rows_orders_columns_and_maps_nulls(tmp_path):
    path = _write_partition(
        tmp_path,
        2025,
        [
            _row(id=10, primary_type="THEFT"),
            _row(id=11, latitude=None, longitude=None),  # ungeocoded
        ],
    )
    rows = list(loader.iter_rows(path, ["id", "latitude", "primary_type"]))
    assert rows[0] == (10, 41.8781, "THEFT")
    assert rows[1] == (11, None, "BATTERY")  # null lat -> None, order preserved


# --- integration tests: need Postgres; use the dedicated test DB (see conftest) --


@pytest.mark.integration
def test_load_full_refresh_round_trip(tmp_path, pg_conn):
    _write_partition(tmp_path, 2024, [_row(id=1), _row(id=2, latitude=None, longitude=None)])
    _write_partition(tmp_path, 2025, [_row(id=3)])

    summary = loader.load(pg_conn, parquet_root=tmp_path)
    assert summary == {"partitions": 2, "rows": 3}

    # All rows present; geom derived for geocoded rows, NULL for the ungeocoded one.
    assert pg_conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 3
    assert pg_conn.execute(
        "SELECT count(*) FROM incidents WHERE geom IS NULL"
    ).fetchone()[0] == 1
    lon = pg_conn.execute(
        "SELECT ST_X(geom::geometry) FROM incidents WHERE id = 1"
    ).fetchone()[0]
    assert round(lon, 4) == -87.6298

    # Secondary indexes were recreated after the load. Asserted by name, not
    # count: a rename would slip past a count and take a tool's index with it.
    idx = {
        name
        for (name,) in pg_conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'incidents'"
        ).fetchall()
    }
    assert idx == {
        "incidents_pkey",
        "incidents_geom_gix",
        "incidents_date_idx",
        "incidents_case_idx",
        "incidents_ptc_date_idx",
        "incidents_stable_date_idx",
    }

    # Full refresh is idempotent: a second run replaces, not appends.
    loader.load(pg_conn, parquet_root=tmp_path)
    assert pg_conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 3


@pytest.mark.integration
def test_upsert_updates_and_inserts(tmp_path, pg_conn):
    # Initial state: two 2025 rows.
    _write_partition(
        tmp_path,
        2025,
        [_row(id=1, primary_type="THEFT", primary_type_canonical="THEFT"), _row(id=2)],
    )
    loader.load(pg_conn, parquet_root=tmp_path)

    # Incremental rewrites the 2025 partition (id=1 reclassified + moved, id=3 new)
    # and lands a brand-new current-year partition (id=4 in 2026).
    _write_partition(
        tmp_path,
        2025,
        [
            _row(id=1, latitude=41.9, longitude=-87.7),  # default primary_type BATTERY
            _row(id=2),
            _row(id=3),
        ],
    )
    _write_partition(tmp_path, 2026, [_row(id=4, date=datetime(2026, 1, 1, 0, 0))])

    summary = loader.upsert(pg_conn, parquet_root=tmp_path)
    assert summary == {"partitions": 2, "rows": 4}

    # id=1 overwritten in place (not duplicated); id=3/4 inserted.
    assert pg_conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 4
    assert (
        pg_conn.execute("SELECT primary_type FROM incidents WHERE id = 1").fetchone()[0]
        == "BATTERY"
    )
    # geom is regenerated from the new coordinates.
    lon = pg_conn.execute(
        "SELECT ST_X(geom::geometry) FROM incidents WHERE id = 1"
    ).fetchone()[0]
    assert round(lon, 1) == -87.7
    assert (
        pg_conn.execute("SELECT count(*) FROM incidents WHERE id IN (3, 4)").fetchone()[0] == 2
    )


@pytest.mark.integration
def test_upsert_years_filter_scopes_to_requested_year(tmp_path, pg_conn):
    _write_partition(
        tmp_path,
        2025,
        [_row(id=1, primary_type="THEFT", primary_type_canonical="THEFT")],
    )
    loader.load(pg_conn, parquet_root=tmp_path)

    # Both partitions change on disk, but we only upsert 2026.
    _write_partition(tmp_path, 2025, [_row(id=1)])  # would flip THEFT -> BATTERY
    _write_partition(tmp_path, 2026, [_row(id=2, date=datetime(2026, 1, 1, 0, 0))])

    summary = loader.upsert(pg_conn, parquet_root=tmp_path, years=[2026])
    assert summary == {"partitions": 1, "rows": 1}

    # 2026 row inserted; the 2025 change was NOT applied.
    assert pg_conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 2
    assert (
        pg_conn.execute("SELECT primary_type FROM incidents WHERE id = 1").fetchone()[0]
        == "THEFT"
    )


@pytest.mark.integration
def test_refresh_picks_up_a_schema_column_added_since_the_table_was_created(tmp_path, pg_conn):
    """The 2026-07-31 failure: schema.sql gains a column, the live table never gets it.

    `CREATE TABLE IF NOT EXISTS` is a no-op against an existing table, so under
    the old truncate-based refresh the new column never arrived and the load died
    at COPY with `UndefinedColumn`. Reproduced here by creating the table from a
    schema with `stable_category` stripped out, then refreshing with the real one.
    """
    # An "old" schema: the current file minus stable_category and its index. Built
    # from the real file rather than hand-written, so it cannot drift from it.
    real_schema = loader.SCHEMA_PATH.read_text()
    old_schema = "\n".join(
        line
        for line in real_schema.splitlines()
        if "stable_category" not in line and "incidents_stable_date_idx" not in line
    )
    old_path = tmp_path / "old_schema.sql"
    old_path.write_text(old_schema)

    loader.apply_schema(pg_conn, schema_path=old_path)
    columns = "SELECT count(*) FROM information_schema.columns WHERE table_name='incidents'"
    assert pg_conn.execute(f"{columns} AND column_name='stable_category'").fetchone()[0] == 0

    _write_partition(tmp_path, 2025, [_row(id=1), _row(id=2)])
    summary = loader.load(pg_conn, parquet_root=tmp_path)

    assert summary == {"partitions": 1, "rows": 2}
    assert pg_conn.execute(f"{columns} AND column_name='stable_category'").fetchone()[0] == 1
    assert pg_conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 2


@pytest.mark.integration
def test_failed_refresh_rolls_the_drop_back(tmp_path, pg_conn):
    """Postgres has transactional DDL, so the DROP is undone with everything else.

    The refresh now drops the table before recreating it, which would be an
    unacceptable trade if a mid-load failure could leave no table at all. It
    cannot: the DROP is inside the same transaction as the COPY.
    """
    import psycopg  # lazy: unit-only runs must not need the `store` extras

    _write_partition(tmp_path, 2025, [_row(id=1), _row(id=2)])
    loader.load(pg_conn, parquet_root=tmp_path)
    assert pg_conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 2

    # A partition carrying a column the table has no home for: COPY fails after
    # the DROP has already run.
    bogus = pa.Table.from_pylist(
        [{**_row(id=3), "not_a_real_column": "x"}],
        schema=SCHEMA.append(pa.field("not_a_real_column", pa.string())),
    )
    (tmp_path / "year=2025").mkdir(parents=True, exist_ok=True)
    pq.write_table(bogus, tmp_path / "year=2025" / "part.parquet")

    with pytest.raises(psycopg.errors.UndefinedColumn):
        loader.load(pg_conn, parquet_root=tmp_path)
    pg_conn.rollback()

    # Table and rows survived the failed refresh.
    assert pg_conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 2


@pytest.mark.integration
def test_upsert_never_drops_the_table(tmp_path, pg_conn):
    """The nightly path must stay non-destructive -- the DROP belongs to refresh only.

    Checked by relfilenode: a drop-and-recreate gives the table a new one even
    though the name and row count would look unchanged.
    """
    _write_partition(tmp_path, 2025, [_row(id=1)])
    loader.load(pg_conn, parquet_root=tmp_path)
    before = pg_conn.execute("SELECT 'incidents'::regclass::oid").fetchone()[0]

    _write_partition(tmp_path, 2025, [_row(id=1), _row(id=2)])
    loader.upsert(pg_conn, parquet_root=tmp_path)

    assert pg_conn.execute("SELECT 'incidents'::regclass::oid").fetchone()[0] == before
    assert pg_conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 2
