"""Tests for the Postgres bulk loader.

The Parquet-reading helpers are pure and always run. The full ``load`` round-trip
needs a live Postgres and is marked ``integration`` (skipped by default, and
auto-skipped if no database is reachable).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from chicago_crime_mcp.store.postgres import loader

# Full incident schema, matching the columns/types ingest writes to Parquet.
SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("case_number", pa.string()),
        ("date", pa.timestamp("us")),
        ("block", pa.string()),
        ("iucr", pa.string()),
        ("primary_type", pa.string()),
        ("description", pa.string()),
        ("location_description", pa.string()),
        ("arrest", pa.bool_()),
        ("domestic", pa.bool_()),
        ("beat", pa.string()),
        ("district", pa.string()),
        ("ward", pa.int64()),
        ("community_area", pa.int64()),
        ("fbi_code", pa.string()),
        ("x_coordinate", pa.float64()),
        ("y_coordinate", pa.float64()),
        ("updated_on", pa.timestamp("us")),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("primary_type_canonical", pa.string()),
    ]
)


def _row(**overrides) -> dict:
    """A complete incident row with sane defaults; override only what matters."""
    base = dict(
        id=1,
        case_number="JF100001",
        date=datetime(2025, 1, 1, 3, 0),
        block="001XX N STATE ST",
        iucr="0486",
        primary_type="BATTERY",
        description="DOMESTIC BATTERY SIMPLE",
        location_description="APARTMENT",
        arrest=False,
        domestic=True,
        beat="1011",
        district="010",
        ward=1,
        community_area=29,
        fbi_code="08B",
        x_coordinate=1150000.0,
        y_coordinate=1900000.0,
        updated_on=datetime(2025, 1, 2, 0, 0),
        latitude=41.8781,
        longitude=-87.6298,
        primary_type_canonical="BATTERY",
    )
    base.update(overrides)
    return base


def _write_partition(base: Path, year: int, rows: list[dict]) -> Path:
    """Write ``rows`` to ``base/year=<year>/part.parquet`` and return the path."""
    path = base / f"year={year}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    return path


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

    # Secondary indexes were recreated after the load (pkey + the 4 in schema.sql).
    idx = pg_conn.execute(
        "SELECT count(*) FROM pg_indexes WHERE tablename = 'incidents'"
    ).fetchone()[0]
    assert idx == 5

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
