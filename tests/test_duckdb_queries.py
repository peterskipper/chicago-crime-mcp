"""Tests for the DuckDB aggregate read API.

DuckDB is embedded, so these run as ordinary unit tests over a temp-directory
fixture -- no marker, no live server.

The load-bearing assertions are the properties the design rests on: the two
tiers return identical numbers, the router picks a tier for a stated reason,
normalization saves a caller from a silently empty answer, and the coverage
report's share is measured against rows that are actually in the span.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime

import pytest

from chicago_crime_mcp.store.duckdb import queries, rollups
from tests.helpers import row as _row
from tests.helpers import write_partition as _write_partition


@pytest.fixture
def conn(tmp_path):
    """Build rollups over a fixture dataset; yield the open connection.

    The rows cover what the reader has to get right: three years, several
    months, arrest/domestic/geocoded variation, two community areas, and two
    codes the coverage report has to treat differently -- `0810`, which appears
    only in the last year (an onset), and `1310`, which appears only in the
    first and is silent long enough afterwards to count as retired.
    """
    _write_partition(
        tmp_path / "parquet",
        2023,
        [
            _row(id=8, date=datetime(2023, 6, 15), iucr="1310",
                 primary_type="CRIMINAL DAMAGE", primary_type_canonical="CRIMINAL DAMAGE",
                 description="TO PROPERTY", community_area=1, domestic=False),
        ],
    )
    _write_partition(
        tmp_path / "parquet",
        2024,
        [
            _row(id=1, date=datetime(2024, 1, 5), arrest=True, community_area=1),
            _row(id=2, date=datetime(2024, 1, 20), community_area=1),
            _row(id=3, date=datetime(2024, 2, 10), community_area=2, domestic=False),
            _row(id=4, date=datetime(2024, 3, 15), community_area=1, latitude=None,
                 longitude=None, domestic=False),
        ],
    )
    _write_partition(
        tmp_path / "parquet",
        2025,
        [
            _row(id=5, date=datetime(2025, 1, 8), arrest=True, community_area=1,
                 domestic=False),
            # A THEFT code that exists only in 2025 -- the coverage report should
            # flag it as entering any span that opens in 2024.
            _row(id=6, date=datetime(2025, 2, 3), iucr="0810", primary_type="THEFT",
                 primary_type_canonical="THEFT", description="THEFT OVER $500",
                 community_area=2, domestic=False),
            _row(id=7, date=datetime(2025, 2, 27), iucr="0810", primary_type="THEFT",
                 primary_type_canonical="THEFT", description="THEFT OVER $500",
                 community_area=2, domestic=False),
        ],
    )

    connection = rollups.connect(
        duckdb_path=tmp_path / "db" / "crime.duckdb",
        parquet_root=tmp_path / "parquet",
    )
    rollups.build(connection)
    yield connection
    connection.close()


def _totals(result) -> int:
    """Sum the incidents across every returned bucket."""
    return sum(r.incidents for r in result.rows)


def test_month_aligned_span_routes_to_the_rollup(conn):
    """A whole-months span is summed out of the materialized table."""
    result = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2024, 1, 1), end=date(2024, 3, 31))
    )
    assert result.route.tier == "rollup"
    assert result.route.table == "rollup_citywide"
    assert "month-aligned" in result.route.reason
    assert _totals(result) == 4


def test_mid_month_span_falls_through_to_a_live_scan(conn):
    """A span the month grain cannot express is scanned rather than rounded."""
    result = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2024, 1, 10), end=date(2024, 2, 15))
    )
    assert result.route.tier == "scan"
    assert result.route.table == rollups.TAGGED_VIEW
    # Excludes id=1 (Jan 5, before the span) and id=4 (Mar 15, after it).
    assert _totals(result) == 2


def test_both_tiers_return_identical_numbers(conn):
    """The tiers must agree: same span, same buckets, same measures.

    Forced by building the same query twice and overriding the tier, so this
    compares the two SQL shapes rather than two different questions.
    """
    query = queries.AggregateQuery(
        start=date(2024, 1, 1), end=date(2025, 12, 31), geography="community_area"
    )
    rollup_sql, rollup_params = queries._build_aggregate_sql(query, "rollup")
    scan_sql, scan_params = queries._build_aggregate_sql(query, "scan")

    assert conn.execute(rollup_sql, rollup_params).fetchall() == (
        conn.execute(scan_sql, scan_params).fetchall()
    )


def test_grain_rolls_months_up_without_changing_totals(conn):
    """Quarter and year buckets are sums of the same months."""
    span = dict(start=date(2024, 1, 1), end=date(2024, 12, 31), breakdown_by_type=False)
    by_month = queries.aggregate(conn, queries.AggregateQuery(grain="month", **span))
    by_quarter = queries.aggregate(conn, queries.AggregateQuery(grain="quarter", **span))
    by_year = queries.aggregate(conn, queries.AggregateQuery(grain="year", **span))

    assert _totals(by_month) == _totals(by_quarter) == _totals(by_year) == 4
    assert len(by_year.rows) == 1
    assert by_year.rows[0].period == date(2024, 1, 1)
    assert by_quarter.rows[0].period == date(2024, 1, 1)


def test_taxonomy_selects_the_grouping_column(conn):
    """`source` and `comparable` are a choice of column, not a fallback path."""
    query = queries.AggregateQuery(start=date(2025, 1, 1), end=date(2025, 12, 31))
    source = queries.aggregate(conn, replace(query, taxonomy="source"))
    comparable = queries.aggregate(conn, replace(query, taxonomy="comparable"))

    assert _totals(source) == _totals(comparable) == 3
    assert source.query.taxonomy == "source"
    assert comparable.query.taxonomy == "comparable"


def test_measures_are_counts_not_rates(conn):
    """Arrests, domestic and geocoded come back as countable rows."""
    result = queries.aggregate(
        conn,
        queries.AggregateQuery(
            start=date(2024, 1, 1), end=date(2024, 12, 31), breakdown_by_type=False
        ),
    )
    assert sum(r.arrests for r in result.rows) == 1
    assert sum(r.domestic for r in result.rows) == 2
    # id=4 has no coordinates.
    assert sum(r.geocoded for r in result.rows) == 3


def test_geography_becomes_a_dimension_and_a_filter(conn):
    """Selecting a geography groups by it; supplying values also narrows it."""
    span = dict(start=date(2024, 1, 1), end=date(2025, 12, 31), geography="community_area")
    everywhere = queries.aggregate(conn, queries.AggregateQuery(**span))
    assert {r.geography_value for r in everywhere.rows} == {1, 2}

    narrowed = queries.aggregate(
        conn, queries.AggregateQuery(geography_values=(2,), **span)
    )
    assert {r.geography_value for r in narrowed.rows} == {2}
    assert _totals(narrowed) == 3


def test_citywide_has_no_geography_column(conn):
    """Citywide is the absence of a dimension, not a filter over one."""
    result = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2024, 1, 1), end=date(2024, 12, 31))
    )
    assert all(r.geography_value is None for r in result.rows)


def test_unpadded_district_still_matches(conn):
    """A caller passing district 10 must not get a confident empty answer.

    The source stores districts zero-padded (`010`), so the unnormalized value
    would match nothing. This is the failure mode normalization exists to stop.
    """
    result = queries.aggregate(
        conn,
        queries.AggregateQuery(
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            geography="district",
            geography_values=(10,),
        ),
    )
    assert result.query.geography_values == ("010",)
    assert _totals(result) == 4


def test_lowercase_type_filter_still_matches(conn):
    """Offense categories are stored upper-case; a filter is upper-cased to match."""
    result = queries.aggregate(
        conn,
        queries.AggregateQuery(
            start=date(2025, 1, 1), end=date(2025, 12, 31), types=("theft",)
        ),
    )
    assert result.query.types == ("THEFT",)
    assert _totals(result) == 2


def test_non_numeric_ward_names_the_field(conn):
    """A bad geography value raises with the field in the message.

    The server layer turns this into the structured error the model retries from,
    so the message has to identify what was wrong.
    """
    with pytest.raises(ValueError, match="ward"):
        queries.aggregate(
            conn,
            queries.AggregateQuery(
                start=date(2024, 1, 1),
                end=date(2024, 12, 31),
                geography="ward",
                geography_values=("Logan Square",),
            ),
        )


def test_inverted_range_and_bad_limit_are_rejected():
    """Unanswerable queries fail at construction, before any SQL is built."""
    with pytest.raises(ValueError, match="before start"):
        queries.AggregateQuery(start=date(2025, 1, 1), end=date(2024, 1, 1))
    with pytest.raises(ValueError, match="limit"):
        queries.AggregateQuery(start=date(2024, 1, 1), end=date(2024, 12, 31), limit=0)


def test_limit_truncates_and_says_so(conn):
    """Bounded results: the cap is enforced and the flag is set."""
    result = queries.aggregate(
        conn,
        queries.AggregateQuery(start=date(2024, 1, 1), end=date(2025, 12, 31), limit=2),
    )
    assert len(result.rows) == 2
    assert result.truncated is True


def test_untruncated_result_is_not_flagged(conn):
    """The extra fetched row must not itself trip the truncation flag."""
    result = queries.aggregate(
        conn,
        queries.AggregateQuery(
            start=date(2024, 1, 1), end=date(2024, 12, 31), breakdown_by_type=False
        ),
    )
    assert result.truncated is False
    assert len(result.rows) == 3


def test_partial_edge_buckets_are_flagged(conn):
    """A span opening or closing mid-bucket marks the edge rows as short."""
    partial = queries.aggregate(
        conn,
        queries.AggregateQuery(
            start=date(2024, 2, 15), end=date(2024, 11, 10), grain="quarter"
        ),
    )
    assert partial.partial_first_period is True
    assert partial.partial_last_period is True

    whole = queries.aggregate(
        conn,
        queries.AggregateQuery(
            start=date(2024, 1, 1), end=date(2024, 12, 31), grain="quarter"
        ),
    )
    assert whole.partial_first_period is False
    assert whole.partial_last_period is False


def test_span_past_the_data_is_provisional(conn):
    """A bucket extending past the newest incident is still filling."""
    past_the_end = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2025, 1, 1), end=date(2025, 12, 31))
    )
    assert past_the_end.provisional is True

    settled = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2024, 1, 1), end=date(2024, 12, 31))
    )
    assert settled.provisional is False


def test_coverage_flags_a_code_that_enters_mid_span(conn):
    """IUCR 0810 appears only in 2025, so a 2024-2025 span is partly built on it."""
    result = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2024, 1, 1), end=date(2025, 12, 31))
    )
    flagged = {c.iucr: c for c in result.coverage.codes}
    assert "0810" in flagged
    assert flagged["0810"].enters is True
    assert flagged["0810"].incidents == 2
    # 2 of the span's 7 rows come from a code that does not cover the span.
    assert result.coverage.total_incidents == 7
    assert result.coverage.affected_incidents == 2
    assert result.coverage.affected_share == pytest.approx(2 / 7)


def test_coverage_share_counts_rows_in_the_span_not_lifetime(conn):
    """The denominator and numerator are both scoped to the requested span.

    A span opening in the month `0810` first appears contains that code's whole
    life, so it is no longer an onset and nothing is flagged -- and the
    denominator is the span's 2 rows, not the dataset's 8.
    """
    result = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2025, 2, 1), end=date(2025, 12, 31))
    )
    assert result.coverage.total_incidents == 2
    assert result.coverage.affected_incidents == 0
    assert result.coverage.affected_share == 0.0


def test_coverage_respects_the_type_filter(conn):
    """Filtering to BATTERY must not weigh a THEFT code's drift against it."""
    result = queries.aggregate(
        conn,
        queries.AggregateQuery(
            start=date(2024, 1, 1), end=date(2025, 12, 31), types=("BATTERY",)
        ),
    )
    assert result.coverage.total_incidents == 5
    assert result.coverage.codes == ()


def test_a_code_silent_at_the_edge_is_not_called_retired(conn):
    """The last observation is right-censored, so absence at the edge proves nothing.

    BATTERY code `0486` has no rows in the dataset's final month. Without the
    retirement buffer it would be flagged as exiting on every span that reaches
    the end of the data -- as would most low-frequency codes, which is noise
    that would make the whole warning ignorable.
    """
    result = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2024, 1, 1), end=date(2025, 12, 31))
    )
    assert "0486" not in {c.iucr for c in result.coverage.codes}


def test_coverage_flags_a_code_that_goes_silent_well_before_the_end(conn):
    """A code absent for longer than the buffer is a real retirement.

    `1310` occurs once in mid-2023 and never again, which is more than
    MIN_ABSENCE_MONTHS before the dataset ends -- unlike `0486`, this absence is
    sustained rather than a gap at the edge.

    The span deliberately opens before the data does, which also pins the lower
    clamp: `1310` appears in the dataset's very first month, so it is a
    retirement and not also an onset.
    """
    result = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2023, 1, 1), end=date(2025, 12, 31))
    )
    flagged = {c.iucr: c for c in result.coverage.codes}
    assert flagged["1310"].exits is True
    assert flagged["1310"].enters is False
    # The onset case is still caught in the same span.
    assert flagged["0810"].enters is True


def test_coverage_fires_under_the_comparable_taxonomy_too(conn):
    """`comparable` corrects only curated drift, so it never suppresses the warning."""
    result = queries.aggregate(
        conn,
        queries.AggregateQuery(
            start=date(2024, 1, 1), end=date(2025, 12, 31), taxonomy="comparable"
        ),
    )
    assert {c.iucr for c in result.coverage.codes} == {"0810"}


def test_span_entirely_past_the_data_reports_nothing(conn):
    """A future span yields an empty report rather than flagging every code as retired."""
    result = queries.aggregate(
        conn, queries.AggregateQuery(start=date(2030, 1, 1), end=date(2030, 12, 31))
    )
    assert result.rows == ()
    assert result.coverage.codes == ()
    assert result.coverage.total_incidents == 0
    assert result.coverage.affected_share == 0.0


def test_dataset_meta_describes_the_build(conn):
    """rollup_meta gives describe_schema its date range and row count."""
    meta = queries.dataset_meta(conn)
    assert meta.source_rows == 8
    assert meta.partitions == 3
    assert meta.min_date == datetime(2023, 6, 15)
    assert meta.max_date == datetime(2025, 2, 27)


def test_categories_lists_valid_values_per_taxonomy(conn):
    """The enum values describe_schema hands the model instead of letting it guess."""
    assert queries.categories(conn, "source") == ("BATTERY", "CRIMINAL DAMAGE", "THEFT")
    assert queries.categories(conn, "comparable") == ("BATTERY", "CRIMINAL DAMAGE", "THEFT")
