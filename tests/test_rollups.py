"""Tests for the DuckDB rollup builder.

DuckDB is embedded -- no server, no fixtures beyond a temp directory -- so these
all run as unit tests in the default suite, unlike the Postgres loader's
round-trip tests.

The load-bearing assertions are the invariants the rollup design rests on:
every incident is counted exactly once in every table, null geographies keep
their own bucket, and beat totals do not depend on ``district``.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from chicago_crime_mcp.store.duckdb import rollups
from tests.helpers import row as _row
from tests.helpers import write_partition as _write_partition


@pytest.fixture
def built(tmp_path):
    """Build rollups over a small fixture dataset; yield the open connection.

    The fixture rows deliberately cover every edge the SQL has to handle: two
    months, two years, arrest/domestic/geocoded flags, null ward and community
    area, and one beat recorded under two different districts.
    """
    _write_partition(
        tmp_path / "parquet",
        2024,
        [
            # 2024-01, BATTERY, beat 2422 -- three incidents, but the source puts
            # one of them in district 006 and two in district 024.
            _row(id=1, date=datetime(2024, 1, 5), beat="2422", district="024", arrest=True),
            _row(id=2, date=datetime(2024, 1, 9), beat="2422", district="024", domestic=False),
            _row(id=3, date=datetime(2024, 1, 20), beat="2422", district="006", domestic=False),
            # 2024-02, THEFT, ungeocoded + null ward/community_area.
            _row(
                id=4,
                date=datetime(2024, 2, 2),
                primary_type_canonical="THEFT",
                beat="1011",
                district="010",
                ward=None,
                community_area=None,
                latitude=None,
                longitude=None,
                domestic=False,
            ),
        ],
    )
    _write_partition(
        tmp_path / "parquet",
        2025,
        [_row(id=5, date=datetime(2025, 6, 1), arrest=True, domestic=False)],
    )

    conn = rollups.connect(
        duckdb_path=tmp_path / "db" / "crime.duckdb",
        parquet_root=tmp_path / "parquet",
    )
    rollups.build(conn)
    yield conn
    conn.close()


# --- view + connection ------------------------------------------------------


def test_ensure_parquet_view_rejects_empty_root(tmp_path):
    """A root with no partitions fails loudly, not at first query."""
    import duckdb as duckdb_pkg

    conn = duckdb_pkg.connect()
    with pytest.raises(FileNotFoundError):
        rollups.ensure_parquet_view(conn, tmp_path / "empty")
    conn.close()


def test_connect_propagates_empty_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        rollups.connect(duckdb_path=tmp_path / "db.duckdb", parquet_root=tmp_path / "empty")


def test_connect_creates_parent_directory(tmp_path):
    _write_partition(tmp_path / "parquet", 2025, [_row()])
    target = tmp_path / "nested" / "dir" / "crime.duckdb"
    conn = rollups.connect(duckdb_path=target, parquet_root=tmp_path / "parquet")
    conn.close()
    assert target.exists()


def test_view_reconstructs_hive_year_column(tmp_path):
    _write_partition(tmp_path / "parquet", 2024, [_row(id=1, date=datetime(2024, 3, 1))])
    _write_partition(tmp_path / "parquet", 2025, [_row(id=2, date=datetime(2025, 3, 1))])
    conn = rollups.connect(duckdb_path=tmp_path / "db.duckdb", parquet_root=tmp_path / "parquet")
    # `year` is not stored in the files; hive_partitioning=1 derives it from the path.
    years = conn.execute("SELECT year FROM incidents ORDER BY year").fetchall()
    conn.close()
    assert [int(y[0]) for y in years] == [2024, 2025]


def test_view_is_rebuilt_against_a_new_parquet_root(tmp_path):
    """A moved parquet_root must not leave the previous path baked into the view."""
    _write_partition(tmp_path / "first", 2025, [_row(id=1)])
    _write_partition(tmp_path / "second", 2025, [_row(id=1), _row(id=2)])
    db = tmp_path / "db.duckdb"

    conn = rollups.connect(duckdb_path=db, parquet_root=tmp_path / "first")
    assert conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 1
    conn.close()

    conn = rollups.connect(duckdb_path=db, parquet_root=tmp_path / "second")
    assert conn.execute("SELECT count(*) FROM incidents").fetchone()[0] == 2
    conn.close()


# --- the invariants ---------------------------------------------------------


def test_every_table_counts_every_incident_exactly_once(built):
    """SUM(incidents) == source rows on every table -- nothing dropped, nothing double-counted.

    Includes rollup_code_month: it is not a geography rollup, but it is the
    denominator the coverage share is measured against, so a row it drops would
    silently overstate how much of a span drifting codes account for.
    """
    source_rows = built.execute("SELECT source_rows FROM rollup_meta").fetchone()[0]
    assert source_rows == 5
    for table in (*rollups.ROLLUP_TABLES, rollups.CODE_MONTH_TABLE):
        total = built.execute(f"SELECT sum(incidents) FROM {table}").fetchone()[0]
        assert total == source_rows, f"{table} lost or duplicated incidents"


def test_measures_sum_to_source_totals(built):
    """Every measure, not just incidents, survives the rollup intact.

    Fixture totals: 2 arrests (ids 1, 5), 1 domestic (id 1), 4 geocoded (id 4 has
    null coordinates).
    """
    for table in rollups.ROLLUP_TABLES:
        arrests, domestic, geocoded = built.execute(
            f"SELECT sum(arrests), sum(domestic), sum(geocoded) FROM {table}"
        ).fetchone()
        assert (arrests, domestic, geocoded) == (2, 1, 4), table


def test_null_geography_keeps_its_own_bucket(built):
    """The ungeocoded THEFT row has null ward/community_area; it must still be counted."""
    for table, column in (("rollup_ward", "ward"), ("rollup_community_area", "community_area")):
        null_rows = built.execute(
            f"SELECT sum(incidents) FROM {table} WHERE {column} IS NULL"
        ).fetchone()[0]
        assert null_rows == 1, f"{table} dropped its null-{column} bucket"


def test_beat_totals_do_not_depend_on_district(built):
    """The design decision: one geography per table, so a beat cannot split.

    Beat 2422 appears under districts 024 and 006 in the same month and type (a
    source data-entry error). Carrying `district` in rollup_beat would split that
    into two rows and make a beat+district filter silently undercount.
    """
    assert "district" not in [c[0] for c in built.execute("DESCRIBE rollup_beat").fetchall()]

    beat_rows = built.execute(
        "SELECT month, primary_type_canonical, incidents FROM rollup_beat "
        "WHERE beat = '2422' ORDER BY month"
    ).fetchall()
    assert len(beat_rows) == 1, "beat bucket split across districts"
    assert beat_rows[0][2] == 3


def test_district_table_reports_the_source_field_faithfully(built):
    """The same rows, by district, keep the source's (mistaken) assignment."""
    rows = built.execute(
        "SELECT district, incidents FROM rollup_district "
        "WHERE primary_type_canonical = 'BATTERY' AND month = DATE '2024-01-01' "
        "ORDER BY district"
    ).fetchall()
    assert rows == [("006", 1), ("024", 2)]


# --- grain + measures -------------------------------------------------------


def test_month_bucketing_collapses_within_a_month_only(built):
    """Incidents in the same month share a bucket; different months do not."""
    months = built.execute(
        "SELECT month, sum(incidents) FROM rollup_citywide GROUP BY month ORDER BY month"
    ).fetchall()
    assert [(m.strftime("%Y-%m"), n) for m, n in months] == [
        ("2024-01", 3),
        ("2024-02", 1),
        ("2025-06", 1),
    ]


def test_citywide_splits_by_type(built):
    rows = built.execute(
        "SELECT primary_type_canonical, sum(incidents) FROM rollup_citywide "
        "GROUP BY 1 ORDER BY 1"
    ).fetchall()
    assert rows == [("BATTERY", 4), ("THEFT", 1)]


def test_geocoded_measure_excludes_null_coordinates(built):
    """The ungeocoded THEFT row counts as an incident but not as geocoded."""
    incidents, geocoded = built.execute(
        "SELECT sum(incidents), sum(geocoded) FROM rollup_citywide "
        "WHERE primary_type_canonical = 'THEFT'"
    ).fetchone()
    assert (incidents, geocoded) == (1, 0)


def test_no_rate_columns_are_stored(built):
    """Rates must be derived at read time; storing them breaks under summing."""
    for table in rollups.ROLLUP_TABLES:
        columns = [c[0] for c in built.execute(f"DESCRIBE {table}").fetchall()]
        assert not any("rate" in c for c in columns), f"{table} stores a rate"


# --- taxonomy: stable_category + code_coverage ------------------------------


@pytest.fixture
def drifted(tmp_path):
    """Build rollups over rows that exercise the real IUCR taxonomy drift.

    Uses genuine codes, not invented ones, because the join is against the
    committed reference snapshot: ``0610`` (BURGLARY / FORCIBLE ENTRY, no curated
    override) and ``0760`` (BURGLARY FROM MOTOR VEHICLE, curated to THEFT because
    it was minted mid-dataset and moved car break-ins out of THEFT). One row
    carries an IUCR absent from the reference entirely.
    """
    _write_partition(
        tmp_path / "parquet",
        2019,
        [
            _row(id=1, date=datetime(2019, 5, 1), iucr="0610",
                 primary_type_canonical="BURGLARY", community_area=22),
        ],
    )
    _write_partition(
        tmp_path / "parquet",
        2025,
        [
            _row(id=2, date=datetime(2025, 5, 1), iucr="0610",
                 primary_type_canonical="BURGLARY", community_area=22),
            _row(id=3, date=datetime(2025, 5, 2), iucr="0760",
                 primary_type_canonical="BURGLARY", community_area=22),
            _row(id=4, date=datetime(2025, 6, 3), iucr="0760",
                 primary_type_canonical="BURGLARY", community_area=22),
            # An IUCR the snapshot has never heard of: the LEFT JOIN must keep it.
            _row(id=5, date=datetime(2025, 6, 4), iucr="9999",
                 primary_type_canonical="OTHER OFFENSE", community_area=22),
        ],
    )
    conn = rollups.connect(
        duckdb_path=tmp_path / "db" / "crime.duckdb",
        parquet_root=tmp_path / "parquet",
    )
    rollups.build(conn)
    yield conn
    conn.close()


def test_stable_category_defaults_to_the_canonical_type(drifted):
    """A code with no curated override keeps its canonical type in both columns."""
    rows = drifted.execute(
        "SELECT DISTINCT primary_type_canonical, stable_category FROM rollup_citywide "
        "WHERE stable_category = 'BURGLARY'"
    ).fetchall()
    assert rows == [("BURGLARY", "BURGLARY")]


def test_unknown_iucr_is_not_dropped_by_the_reference_join(drifted):
    """The LEFT JOIN plus coalesce: an unmapped code falls back, never disappears."""
    row = drifted.execute(
        "SELECT primary_type_canonical, stable_category, incidents FROM rollup_citywide "
        "WHERE primary_type_canonical = 'OTHER OFFENSE'"
    ).fetchall()
    assert row == [("OTHER OFFENSE", "OTHER OFFENSE", 1)]


def test_curated_code_splits_the_two_type_dimensions(drifted):
    """`0760` is BURGLARY to the city and THEFT for comparison across time."""
    rows = drifted.execute(
        "SELECT primary_type_canonical, stable_category, sum(incidents) FROM rollup_citywide "
        "WHERE stable_category = 'THEFT' GROUP BY 1, 2"
    ).fetchall()
    assert rows == [("BURGLARY", "THEFT", 2)]


def test_both_taxonomies_are_answerable_from_every_table(drifted):
    """The point of carrying both dimensions: source and comparable, one build.

    Source counts 4 burglaries in Logan Square (community area 22); comparable
    counts 2, because the two `0760` rows are car break-ins that lived under
    THEFT before the code existed.
    """
    for table in rollups.ROLLUP_TABLES:
        source, comparable = drifted.execute(
            f"SELECT sum(incidents) FILTER (WHERE primary_type_canonical = 'BURGLARY'), "
            f"       sum(incidents) FILTER (WHERE stable_category = 'BURGLARY') "
            f"FROM {table}"
        ).fetchone()
        assert (source, comparable) == (4, 2), table


def test_adding_the_dimension_does_not_change_totals(drifted):
    """Summing over stable_category must reproduce the pre-taxonomy counts."""
    for table in rollups.ROLLUP_TABLES:
        total = drifted.execute(f"SELECT sum(incidents) FROM {table}").fetchone()[0]
        assert total == 5, table


def test_code_coverage_reports_each_code_lifespan(drifted):
    """First/last month per code -- what the tool layer warns from."""
    rows = drifted.execute(
        "SELECT iucr, primary_type_canonical, stable_category, first_month, last_month, "
        "incidents FROM code_coverage ORDER BY iucr"
    ).fetchall()
    assert rows == [
        # spans the whole fixture window: no warning
        ("0610", "BURGLARY", "BURGLARY", datetime(2019, 5, 1), datetime(2025, 5, 1), 2),
        # introduced mid-window: any span starting before 2025-05 is not comparable
        ("0760", "BURGLARY", "THEFT", datetime(2025, 5, 1), datetime(2025, 6, 1), 2),
        ("9999", "OTHER OFFENSE", "OTHER OFFENSE", datetime(2025, 6, 1), datetime(2025, 6, 1), 1),
    ]


def test_code_coverage_accounts_for_every_incident(drifted):
    """No code is missing, so a share-of-rows warning can be computed from it."""
    source_rows = drifted.execute("SELECT source_rows FROM rollup_meta").fetchone()[0]
    covered = drifted.execute("SELECT sum(incidents) FROM code_coverage").fetchone()[0]
    assert covered == source_rows


# --- the reference table ----------------------------------------------------


def test_reference_table_preserves_zero_padded_codes(built):
    """IUCR codes are identifiers: `0110` must not arrive as the integer 110."""
    codes = built.execute(
        "SELECT iucr FROM iucr_reference WHERE iucr IN ('0110', '0760')"
    ).fetchall()
    assert sorted(c[0] for c in codes) == ["0110", "0760"]


def test_reference_table_reads_blank_curation_as_null(built):
    """Blank `stable_category` must be NULL so the coalesce falls through."""
    assert built.execute(
        "SELECT stable_category FROM iucr_reference WHERE iucr = '0110'"
    ).fetchone()[0] is None
    assert built.execute(
        "SELECT stable_category FROM iucr_reference WHERE iucr = '0760'"
    ).fetchone()[0] == "THEFT"


def test_ensure_iucr_reference_rejects_a_missing_snapshot(tmp_path):
    import duckdb as duckdb_pkg

    conn = duckdb_pkg.connect()
    with pytest.raises(FileNotFoundError):
        rollups.ensure_iucr_reference(conn, tmp_path / "absent.csv")
    conn.close()


# --- meta + rebuild ---------------------------------------------------------


def test_meta_describes_the_build(built):
    built_at, source_rows, partitions, min_date, max_date = built.execute(
        "SELECT built_at, source_rows, partitions, min_date, max_date FROM rollup_meta"
    ).fetchone()
    assert (source_rows, partitions) == (5, 2)
    assert min_date == datetime(2024, 1, 5)
    assert max_date == datetime(2025, 6, 1)
    assert built_at.tzinfo is None  # naive UTC; readable without pytz


def test_build_summary_reports_row_counts(tmp_path):
    _write_partition(tmp_path / "parquet", 2025, [_row(id=1), _row(id=2, beat="2000")])
    conn = rollups.connect(duckdb_path=tmp_path / "db.duckdb", parquet_root=tmp_path / "parquet")
    summary = rollups.build(conn)
    conn.close()
    assert summary["source_rows"] == 2
    assert summary["rollup_citywide"] == 1  # same month + type
    assert summary["rollup_beat"] == 2  # two beats
    assert summary[rollups.COVERAGE_TABLE] == 1  # both rows share an IUCR
    assert summary[rollups.CODE_MONTH_TABLE] == 1  # same IUCR, same month
    assert set(summary) == set(rollups.ROLLUP_TABLES) | {
        rollups.COVERAGE_TABLE,
        rollups.CODE_MONTH_TABLE,
        "source_rows",
    }


def test_rebuild_replaces_rather_than_appends(tmp_path):
    """CREATE OR REPLACE: a second build over more data must not double-count."""
    parquet_root = tmp_path / "parquet"
    _write_partition(parquet_root, 2025, [_row(id=1)])
    conn = rollups.connect(duckdb_path=tmp_path / "db.duckdb", parquet_root=parquet_root)
    assert rollups.build(conn)["source_rows"] == 1

    _write_partition(parquet_root, 2025, [_row(id=1), _row(id=2)])
    summary = rollups.build(conn)
    assert summary["source_rows"] == 2
    assert conn.execute("SELECT sum(incidents) FROM rollup_citywide").fetchone()[0] == 2
    conn.close()


def test_build_is_idempotent(built):
    """Rebuilding unchanged data changes nothing."""
    before = built.execute("SELECT sum(incidents) FROM rollup_beat").fetchone()[0]
    rollups.build(built)
    assert built.execute("SELECT sum(incidents) FROM rollup_beat").fetchone()[0] == before
