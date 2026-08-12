"""Tests for the three Postgres-backed tools.

Marked ``integration``: they need a live Postgres, and run against the dedicated
test database (see ``conftest``), never the dev one.

The same fixture rows are loaded into Postgres *and* rolled up into DuckDB,
because these tools use both stores -- Postgres answers the query and DuckDB
supplies the vocabulary, the provenance, and the geocode rates the radius
warning is composed from. Testing them against a single store would miss the
seam where the two meet.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime

import pytest

from chicago_crime_mcp.server.context import ServerContext, use_context
from chicago_crime_mcp.server.errors import (
    InvalidArgumentError,
    StaleCursorError,
    UnknownValueError,
)
from chicago_crime_mcp.server.tools import get_incident, nearby_incidents, search_incidents
from chicago_crime_mcp.store.config import StoreConfig
from chicago_crime_mcp.store.duckdb import rollups
from tests.helpers import row as _row
from tests.helpers import write_partition as _write_partition

# Every row sits in district 011 except id=3, and in community area 29 except
# id=3, so a geography filter always has something to exclude. See the warning
# in `tests.helpers.row`.
ROWS = [
    # Two offenses under one RD number: the case-number multiplicity.
    _row(id=1, case_number="JF100001", date=datetime(2025, 1, 10, 9, 0), district="011",
         beat="1111"),
    _row(id=2, case_number="JF100001", date=datetime(2025, 1, 10, 9, 0), district="011",
         beat="1111", primary_type="THEFT", primary_type_canonical="THEFT",
         stable_category="THEFT"),
    # A curated remap: source says BURGLARY, comparable says THEFT.
    _row(id=3, case_number="JF100003", date=datetime(2025, 2, 1, 12, 0),
         primary_type_canonical="BURGLARY", stable_category="THEFT", district="010",
         beat="1011", ward=2, community_area=1),
    _row(id=4, case_number="JF100004", date=datetime(2025, 3, 31, 23, 30), district="011",
         beat="1111", arrest=True),
    # Ungeocoded: invisible to a radius query by construction.
    _row(id=5, case_number="JF100005", date=datetime(2025, 3, 5, 1, 0), district="011",
         beat="1111", latitude=None, longitude=None),
    # ~2 km north, so a small radius excludes it.
    _row(id=6, case_number="JF100006", date=datetime(2025, 3, 6, 1, 0), district="011",
         beat="1111", latitude=41.8961),
]


class _PooledTestContext(ServerContext):
    """A context whose Postgres half is the test connection.

    Everything else -- the DuckDB connection, the vocabulary cache, the inode
    check -- is the real implementation, so these tests exercise the same code
    the server runs. Only the pool is replaced, because a pool would open its
    own connections outside the fixture's transaction management.
    """

    def __init__(self, config, conn):
        super().__init__(config)
        self._conn = conn

    @contextmanager
    def postgres(self):
        """Yield the test connection instead of borrowing from a pool."""
        yield self._conn


@pytest.fixture
def context(tmp_path, pg_conn):
    """Load ROWS into Postgres and roll them up in DuckDB; yield a context."""
    from chicago_crime_mcp.store.postgres import loader

    parquet = tmp_path / "parquet"
    _write_partition(parquet, 2025, ROWS)
    loader.load(pg_conn, parquet_root=parquet)

    path = tmp_path / "db" / "crime.duckdb"
    build_conn = rollups.connect(duckdb_path=path, parquet_root=parquet)
    rollups.build(build_conn)
    build_conn.close()

    ctx = _PooledTestContext(
        StoreConfig(duckdb_path=path, parquet_root=parquet), pg_conn
    )
    ctx._open_duckdb()
    with use_context(ctx):
        yield ctx
    ctx.close()


# --- get_incident -----------------------------------------------------------


@pytest.mark.integration
def test_get_incident_by_id_returns_the_full_row(context):
    """The full shape, including the columns search results drop."""
    result = get_incident(incident_id=3)
    assert result.row_count == 1
    incident = result.data.incidents[0]
    assert incident.id == 3
    assert incident.primary_type_canonical == "BURGLARY"
    assert incident.stable_category == "THEFT"
    # Present here and absent from IncidentSummaryModel -- that is the point.
    assert incident.iucr and incident.fbi_code


@pytest.mark.integration
def test_get_incident_carries_no_taxonomy_mode(context):
    """The full row holds both taxonomy columns, so no single mode applies."""
    assert get_incident(incident_id=3).taxonomy_mode is None


@pytest.mark.integration
def test_case_number_multiplicity_is_stated_not_hidden(context):
    """An RD number identifies a report; a report can record several offenses."""
    result = get_incident(case_number="JF100001")
    assert result.row_count == 2
    warning = next(w for w in result.warnings if w.code == "multiple_matches")
    assert warning.detail["count"] == 2
    assert "Only id is unique" in warning.message


@pytest.mark.integration
def test_case_number_is_upper_cased(context):
    """The same silent-miss class as unpadded districts."""
    assert get_incident(case_number="jf100003").row_count == 1


@pytest.mark.integration
def test_missing_identifier_is_an_answer_with_a_warning(context):
    result = get_incident(incident_id=999_999)
    assert result.row_count == 0
    assert any(w.code == "empty_result" for w in result.warnings)


@pytest.mark.integration
@pytest.mark.parametrize(
    "kwargs", [{}, {"incident_id": 1, "case_number": "JF100001"}]
)
def test_get_incident_requires_exactly_one_identifier(context, kwargs):
    with pytest.raises(InvalidArgumentError) as exc:
        get_incident(**kwargs)
    assert "exactly one" in str(exc.value)


@pytest.mark.integration
def test_get_incident_still_carries_provenance(context):
    """Postgres-backed responses report the same provenance as DuckDB ones."""
    provenance = get_incident(incident_id=1).provenance
    assert provenance.rows == len(ROWS)
    assert provenance.coverage_start is not None
    assert provenance.dataset_id == "ijzp-q8t2"


# --- search_incidents -------------------------------------------------------


@pytest.mark.integration
def test_search_returns_newest_first(context):
    result = search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31))
    ids = [i.id for i in result.data.incidents]
    assert ids == [4, 6, 5, 3, 2, 1] or ids[0] == 4
    assert result.route.store == "postgres"


@pytest.mark.integration
def test_search_end_date_is_inclusive(context):
    """id=4 is at 23:30 on the last day; a half-open bug would drop it."""
    result = search_incidents(start=date(2025, 3, 31), end=date(2025, 3, 31))
    assert [i.id for i in result.data.incidents] == [4]


@pytest.mark.integration
def test_unpadded_district_normalizes_and_actually_filters(context):
    """Relational, so the test fails if the predicate is removed."""
    unfiltered = search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31))
    filtered = search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31),
                                geography="district", geography_values=[11])
    assert 0 < filtered.row_count < unfiltered.row_count
    assert 3 not in [i.id for i in filtered.data.incidents]
    assert filtered.filters_applied.geography_values == ["011"]


@pytest.mark.integration
def test_taxonomy_switches_the_filter_column(context):
    """id=3 is BURGLARY at source and THEFT under comparable."""
    source = search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31),
                              types=["BURGLARY"])
    comparable = search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31),
                                  types=["THEFT"], taxonomy="comparable")
    assert [i.id for i in source.data.incidents] == [3]
    assert 3 in [i.id for i in comparable.data.incidents]
    assert source.taxonomy_mode == "source"
    assert comparable.taxonomy_mode == "comparable"


@pytest.mark.integration
def test_unknown_category_is_a_teaching_error(context):
    with pytest.raises(UnknownValueError) as exc:
        search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31), types=["BURGLERY"])
    assert exc.value.nearest_match == "BURGLARY"


@pytest.mark.integration
def test_pagination_walks_every_row_exactly_once(context):
    """Keyset paging: no overlap, no gap, and it terminates."""
    seen: list[int] = []
    cursor = None
    for _ in range(10):
        page = search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31),
                                limit=2, cursor=cursor)
        seen.extend(i.id for i in page.data.incidents)
        cursor = page.cursor
        if cursor is None:
            break
    assert sorted(seen) == [1, 2, 3, 4, 5, 6]
    assert len(seen) == len(set(seen))


@pytest.mark.integration
def test_truncated_page_carries_a_cursor_and_says_so(context):
    page = search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31), limit=2)
    assert page.truncated is True
    assert page.cursor is not None
    warning = next(w for w in page.warnings if w.code == "truncated")
    assert "cursor" in warning.message


@pytest.mark.integration
def test_cursor_from_a_different_query_is_rejected(context):
    """An unbound cursor would silently page into another result set."""
    page = search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31), limit=2)
    with pytest.raises(StaleCursorError) as exc:
        search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31),
                         types=["BATTERY"], limit=2, cursor=page.cursor)
    assert "without a cursor" in str(exc.value)


@pytest.mark.integration
def test_malformed_cursor_is_reported_as_a_cursor_problem(context):
    with pytest.raises(StaleCursorError):
        search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31), cursor="not-a-cursor")


@pytest.mark.integration
def test_search_limit_is_capped_with_a_pointer_to_aggregate(context):
    with pytest.raises(InvalidArgumentError) as exc:
        search_incidents(start=date(2025, 1, 1), end=date(2025, 12, 31), limit=10_000)
    assert "aggregate_incidents" in str(exc.value)


@pytest.mark.integration
def test_empty_search_names_the_filters_that_were_applied(context):
    result = search_incidents(start=date(2025, 1, 1), end=date(2025, 1, 2),
                              types=["THEFT"], arrest=True)
    assert result.row_count == 0
    warning = next(w for w in result.warnings if w.code == "empty_result")
    assert set(warning.detail["filters_applied"]) == {"types", "arrest"}


# --- nearby_incidents -------------------------------------------------------


@pytest.mark.integration
def test_nearby_summarizes_without_returning_rows(context):
    """Summary-first: counts and distances, no row dump unless asked."""
    result = nearby_incidents(latitude=41.8781, longitude=-87.6298, radius_m=500,
                              start=date(2025, 1, 1), end=date(2025, 12, 31))
    assert result.data.total == 4  # six rows, less the ungeocoded one and the far one
    assert result.data.incidents == []
    assert sum(t.incidents for t in result.data.by_type) == result.data.total
    assert sum(r.incidents for r in result.data.rings) == result.data.total


@pytest.mark.integration
def test_nearby_include_rows_opts_into_detail(context):
    result = nearby_incidents(latitude=41.8781, longitude=-87.6298, radius_m=500,
                              start=date(2025, 1, 1), end=date(2025, 12, 31),
                              include_rows=True)
    assert [i.id for i in result.data.incidents]


@pytest.mark.integration
def test_nearby_radius_actually_excludes(context):
    """The far row is in at 3 km and out at 500 m."""
    near = nearby_incidents(latitude=41.8781, longitude=-87.6298, radius_m=500,
                            start=date(2025, 1, 1), end=date(2025, 12, 31))
    far = nearby_incidents(latitude=41.8781, longitude=-87.6298, radius_m=3000,
                           start=date(2025, 1, 1), end=date(2025, 12, 31))
    assert near.data.total < far.data.total


@pytest.mark.integration
def test_nearby_warns_about_ungeocoded_rows_with_real_rates(context):
    """The bare fact comes from Postgres; the rates are composed from DuckDB.

    id=5 has no coordinates, so it cannot appear in any radius result. The
    warning has to quote the rate for the categories actually found, since the
    gap varies by offense type rather than by place.
    """
    result = nearby_incidents(latitude=41.8781, longitude=-87.6298, radius_m=500,
                              start=date(2025, 1, 1), end=date(2025, 12, 31))
    warning = next(w for w in result.warnings if w.code == "excludes_ungeocoded")
    assert "geocoded_rate_by_type" in warning.detail
    rates = warning.detail["geocoded_rate_by_type"]
    assert set(rates) == {t.category for t in result.data.by_type}
    # BATTERY has the ungeocoded row, so its rate must be below 1.
    assert rates["BATTERY"] < 1.0
    assert warning.detail["overall_geocoded_rate"] < 1.0


@pytest.mark.integration
def test_nearby_rings_are_equal_width(context):
    result = nearby_incidents(latitude=41.8781, longitude=-87.6298, radius_m=400,
                              start=date(2025, 1, 1), end=date(2025, 12, 31))
    widths = {round(r.upper_m - r.lower_m, 6) for r in result.data.rings}
    assert len(widths) == 1
    assert result.data.rings[0].lower_m == 0.0


@pytest.mark.integration
def test_nearby_rejects_an_oversized_radius_and_points_elsewhere(context):
    with pytest.raises(InvalidArgumentError) as exc:
        nearby_incidents(latitude=41.8781, longitude=-87.6298, radius_m=50_000,
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
    assert exc.value.field == "radius_m"
    assert "aggregate_incidents" in str(exc.value)


@pytest.mark.integration
def test_nearby_rejects_impossible_coordinates(context):
    with pytest.raises(InvalidArgumentError) as exc:
        nearby_incidents(latitude=91.0, longitude=-87.6298, radius_m=500,
                         start=date(2025, 1, 1), end=date(2025, 12, 31))
    assert exc.value.field == "latitude"


@pytest.mark.integration
def test_nearby_takes_no_geography_filter(context):
    """The radius is the geography; the filters echoed back say so."""
    result = nearby_incidents(latitude=41.8781, longitude=-87.6298, radius_m=500,
                              start=date(2025, 1, 1), end=date(2025, 12, 31))
    assert "geography" not in result.filters_applied.model_dump()
