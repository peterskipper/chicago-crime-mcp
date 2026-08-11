"""Tests for the Postgres read API.

Validation, cursor handling, predicate building and the summary folding are pure
and always run. The round-trips need a live Postgres and are marked
``integration`` (skipped by default; they use the dedicated test database, see
conftest).
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime

import pytest
from psycopg import sql

from chicago_crime_mcp.store.postgres import queries
from tests.helpers import row as _row
from tests.helpers import write_partition as _write_partition

# --- unit tests: no database ------------------------------------------------


def _rendered(composed) -> str:
    return composed.as_string(None)


def test_summary_columns_are_a_subset_of_the_full_row():
    assert set(queries.SUMMARY_COLUMNS) < set(queries.INCIDENT_COLUMNS)


@pytest.mark.parametrize(
    ("columns", "model"),
    [
        (queries.INCIDENT_COLUMNS, queries.Incident),
        (queries.SUMMARY_COLUMNS, queries.IncidentSummary),
    ],
)
def test_column_order_matches_dataclass_field_order(columns, model):
    # Rows are mapped positionally (`Incident(*record)`), so a column tuple that
    # drifts from its dataclass would not raise -- it would quietly shift every
    # value one field over. This is the assertion that stops that.
    assert columns == tuple(f.name for f in dataclasses.fields(model))


def test_lookup_requires_exactly_one_identifier():
    with pytest.raises(ValueError, match="exactly one"):
        queries.LookupQuery()
    with pytest.raises(ValueError, match="exactly one"):
        queries.LookupQuery(incident_id=1, case_number="JF100001")
    assert queries.LookupQuery(incident_id=1).case_number is None


def test_search_rejects_inverted_range_and_bad_limit():
    with pytest.raises(ValueError, match="before start"):
        queries.SearchQuery(start=date(2025, 2, 1), end=date(2025, 1, 1))
    with pytest.raises(ValueError, match="limit must be"):
        queries.SearchQuery(start=date(2025, 1, 1), end=date(2025, 2, 1), limit=0)
    with pytest.raises(ValueError, match="limit must be"):
        queries.SearchQuery(
            start=date(2025, 1, 1), end=date(2025, 2, 1), limit=queries.MAX_LIMIT + 1
        )


def test_nearby_rejects_impossible_geometry():
    span = {"start": date(2025, 1, 1), "end": date(2025, 12, 31)}
    with pytest.raises(ValueError, match="radius_m must be"):
        queries.NearbyQuery(latitude=41.9, longitude=-87.6, radius_m=0, **span)
    with pytest.raises(ValueError, match="radius_m must be"):
        queries.NearbyQuery(
            latitude=41.9, longitude=-87.6, radius_m=queries.MAX_RADIUS_M + 1, **span
        )
    with pytest.raises(ValueError, match="latitude must be"):
        queries.NearbyQuery(latitude=91, longitude=-87.6, radius_m=100, **span)
    with pytest.raises(ValueError, match="longitude must be"):
        queries.NearbyQuery(latitude=41.9, longitude=-181, radius_m=100, **span)


def test_filters_end_date_is_inclusive_via_half_open_range():
    # Must match the DuckDB module exactly: "through March 31st" includes the
    # whole of the 31st, whatever time of day the timestamps carry.
    _conditions, params = queries._filters(
        date(2025, 1, 1), date(2025, 3, 31), (), "source", "citywide", (), None, None
    )
    assert params == [date(2025, 1, 1), date(2025, 4, 1)]


def test_filters_selects_the_taxonomy_column():
    cases = (("source", "primary_type_canonical"), ("comparable", "stable_category"))
    for taxonomy, column in cases:
        conditions, params = queries._filters(
            date(2025, 1, 1), date(2025, 1, 31), ("THEFT",), taxonomy, "citywide", (), None, None
        )
        assert column in _rendered(sql.SQL(" AND ").join(conditions))
        assert params[-1] == ["THEFT"]


def test_filters_omits_absent_predicates():
    conditions, params = queries._filters(
        date(2025, 1, 1), date(2025, 1, 31), (), "source", "citywide", (), None, None
    )
    assert len(conditions) == 2  # the date bounds only
    assert len(params) == 2


def test_filters_adds_geography_and_flags():
    conditions, params = queries._filters(
        date(2025, 1, 1), date(2025, 1, 31), (), "source", "ward", (2, 43), True, False
    )
    rendered = _rendered(sql.SQL(" AND ").join(conditions))
    assert "ward" in rendered
    assert params[2:] == [[2, 43], True, False]


def test_search_normalizes_filters_onto_the_returned_query():
    # Not a database test: normalization happens before any SQL runs, and the
    # normalized form is what the envelope echoes back to the model.
    q = queries._normalized(
        queries.SearchQuery(
            start=date(2025, 1, 1),
            end=date(2025, 1, 31),
            types=(" burglary ",),
            geography="district",
            geography_values=(10,),
        )
    )
    assert q.types == ("BURGLARY",)
    assert q.geography_values == ("010",)


# --- cursors ----------------------------------------------------------------


def test_cursor_round_trips():
    when = datetime(2025, 3, 21, 18, 0)
    cursor = queries.encode_cursor(when, 13782728, "abc123")
    assert queries.decode_cursor(cursor, "abc123") == (when, 13782728)


def test_cursor_is_opaque():
    cursor = queries.encode_cursor(datetime(2025, 3, 21, 18, 0), 1, "abc123")
    assert "2025" not in cursor


@pytest.mark.parametrize("bad", ["", "not-base64!!", "YWJj", "e30="])
def test_malformed_cursor_raises(bad):
    with pytest.raises(ValueError, match="not a valid cursor"):
        queries.decode_cursor(bad, "abc123")


def test_cursor_from_a_different_query_is_rejected():
    # The failure this prevents is silent: replayed against other filters an
    # unbound cursor returns a plausible, non-empty page from the wrong place.
    cursor = queries.encode_cursor(datetime(2025, 3, 21, 18, 0), 1, "abc123")
    with pytest.raises(ValueError, match="different query"):
        queries.decode_cursor(cursor, "def456")


def test_fingerprint_ignores_paging_but_tracks_filters():
    base = queries.SearchQuery(start=date(2025, 1, 1), end=date(2025, 1, 31), types=("THEFT",))
    # Page size may change mid-scan; it does not move the keyset position.
    assert queries._fingerprint(base) == queries._fingerprint(dataclasses.replace(base, limit=10))
    assert queries._fingerprint(base) != queries._fingerprint(
        dataclasses.replace(base, types=("BURGLARY",))
    )
    assert queries._fingerprint(base) != queries._fingerprint(
        dataclasses.replace(base, taxonomy="comparable")
    )


def test_fingerprint_is_computed_after_normalization():
    # district "10" and 10 are the same filter; a cursor issued for one must not
    # be rejected when the caller spells the other on the next page.
    spelling_a = queries._normalized(
        queries.SearchQuery(
            start=date(2025, 1, 1),
            end=date(2025, 1, 31),
            geography="district",
            geography_values=("10",),
        )
    )
    spelling_b = queries._normalized(
        dataclasses.replace(spelling_a, geography_values=(10,))
    )
    assert queries._fingerprint(spelling_a) == queries._fingerprint(spelling_b)


# --- summary folding --------------------------------------------------------


def test_by_type_sums_across_rings_most_frequent_first():
    fetched = [("THEFT", 1, 5), ("THEFT", 2, 7), ("ARSON", 1, 9)]
    assert queries._fold_by_type(fetched) == (
        queries.TypeCount("THEFT", 12),
        queries.TypeCount("ARSON", 9),
    )


def test_rings_are_equal_width_and_include_empty_ones():
    rings = queries._fold_rings([("THEFT", 1, 5), ("THEFT", 4, 2)], radius_m=400)
    assert [(r.lower_m, r.upper_m, r.incidents) for r in rings] == [
        (0.0, 100.0, 5),
        (100.0, 200.0, 0),
        (200.0, 300.0, 0),
        (300.0, 400.0, 2),
    ]


def test_ring_and_type_folds_agree_on_the_total():
    fetched = [("THEFT", 1, 5), ("THEFT", 2, 7), ("ARSON", 4, 9)]
    assert sum(t.incidents for t in queries._fold_by_type(fetched)) == sum(
        r.incidents for r in queries._fold_rings(fetched, 400)
    )


# --- integration tests: need Postgres ---------------------------------------


@pytest.fixture
def seeded(tmp_path, pg_conn):
    """Load a small, hand-shaped dataset into the test database."""
    from chicago_crime_mcp.store.postgres import loader

    # Everything sits in district 011 except id=3, so a district filter has
    # something to exclude. (`helpers.row` defaults to 010, which would otherwise
    # put every row in the same district and make that test vacuous.)
    rows = [
        # Two offenses sharing one case number -- the RD-number multiplicity.
        _row(id=1, case_number="JF100001", date=datetime(2025, 1, 10, 9, 0), district="011"),
        _row(
            id=2,
            case_number="JF100001",
            date=datetime(2025, 1, 10, 9, 0),
            district="011",
            primary_type="THEFT",
            primary_type_canonical="THEFT",
        ),
        # A curated remap: source says BURGLARY, comparable says THEFT.
        _row(
            id=3,
            case_number="JF100003",
            date=datetime(2025, 2, 1, 12, 0),
            primary_type_canonical="BURGLARY",
            stable_category="THEFT",
            district="010",
            ward=2,
        ),
        # Last day of the range, late in the day -- the inclusive-end case.
        _row(
            id=4,
            case_number="JF100004",
            date=datetime(2025, 3, 31, 23, 30),
            district="011",
            arrest=True,
        ),
        # Ungeocoded: invisible to a radius query by construction.
        _row(id=5, case_number="JF100005", date=datetime(2025, 3, 5, 1, 0),
             district="011", latitude=None, longitude=None),
        # Far away (~2 km north), for the radius boundary.
        _row(id=6, case_number="JF100006", date=datetime(2025, 3, 6, 1, 0),
             district="011", latitude=41.8961),
    ]
    _write_partition(tmp_path, 2025, rows)
    loader.load(pg_conn, parquet_root=tmp_path)
    return pg_conn


@pytest.mark.integration
def test_lookup_by_id_returns_one(seeded):
    result = queries.lookup(seeded, queries.LookupQuery(incident_id=3))
    assert len(result.incidents) == 1
    assert result.incidents[0].primary_type_canonical == "BURGLARY"
    assert result.incidents[0].stable_category == "THEFT"
    assert result.timing.store == "postgres"


@pytest.mark.integration
def test_lookup_by_case_number_returns_every_offense(seeded):
    # The RD number is not unique; returning a scalar here would silently hide
    # the second offense on the same report.
    result = queries.lookup(seeded, queries.LookupQuery(case_number="jf100001"))
    assert [i.id for i in result.incidents] == [1, 2]
    assert result.query.case_number == "JF100001"  # normalized form echoed back


@pytest.mark.integration
def test_lookup_missing_returns_empty_not_error(seeded):
    assert queries.lookup(seeded, queries.LookupQuery(incident_id=99999)).incidents == ()


@pytest.mark.integration
def test_search_end_date_is_inclusive(seeded):
    # id=4 sits at 23:30 on the last requested day.
    result = queries.search(
        seeded, queries.SearchQuery(start=date(2025, 1, 1), end=date(2025, 3, 31))
    )
    assert 4 in {r.id for r in result.rows}


@pytest.mark.integration
def test_search_orders_newest_first(seeded):
    result = queries.search(
        seeded, queries.SearchQuery(start=date(2025, 1, 1), end=date(2025, 12, 31))
    )
    dates = [r.date for r in result.rows]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.integration
def test_search_taxonomy_switches_the_filter_column(seeded):
    span = {"start": date(2025, 1, 1), "end": date(2025, 12, 31)}
    source = queries.search(seeded, queries.SearchQuery(types=("BURGLARY",), **span))
    comparable = queries.search(
        seeded, queries.SearchQuery(types=("BURGLARY",), taxonomy="comparable", **span)
    )
    # id=3 is BURGLARY at source and THEFT under the comparable taxonomy.
    assert {r.id for r in source.rows} == {3}
    assert comparable.rows == ()


@pytest.mark.integration
def test_search_normalizes_geography_before_filtering(seeded):
    # The silent-empty-answer case: district 10 is stored as "010".
    result = queries.search(
        seeded,
        queries.SearchQuery(
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            geography="district",
            geography_values=(10,),
        ),
    )
    assert {r.id for r in result.rows} == {3}
    assert result.query.geography_values == ("010",)


@pytest.mark.integration
def test_search_flags_filter(seeded):
    result = queries.search(
        seeded,
        queries.SearchQuery(start=date(2025, 1, 1), end=date(2025, 12, 31), arrest=True),
    )
    assert {r.id for r in result.rows} == {4}


@pytest.mark.integration
def test_search_pagination_walks_every_row_exactly_once(seeded):
    span = {"start": date(2025, 1, 1), "end": date(2025, 12, 31)}
    seen: list[int] = []
    cursor = None
    for _ in range(10):  # bounded, so a cursor bug cannot spin forever
        page = queries.search(seeded, queries.SearchQuery(limit=2, cursor=cursor, **span))
        seen.extend(r.id for r in page.rows)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert sorted(seen) == [1, 2, 3, 4, 5, 6]
    assert len(seen) == len(set(seen))  # no row served twice


@pytest.mark.integration
def test_search_cursor_is_rejected_against_different_filters(seeded):
    span = {"start": date(2025, 1, 1), "end": date(2025, 12, 31)}
    page = queries.search(seeded, queries.SearchQuery(limit=1, **span))
    with pytest.raises(ValueError, match="different query"):
        queries.search(
            seeded,
            queries.SearchQuery(limit=1, cursor=page.next_cursor, types=("THEFT",), **span),
        )


@pytest.mark.integration
def test_search_last_page_has_no_cursor(seeded):
    result = queries.search(
        seeded, queries.SearchQuery(start=date(2025, 1, 1), end=date(2025, 12, 31), limit=100)
    )
    assert result.truncated is False
    assert result.next_cursor is None


@pytest.mark.integration
def test_nearby_summarizes_without_returning_rows(seeded):
    result = queries.nearby(
        seeded,
        queries.NearbyQuery(
            latitude=41.8781,
            longitude=-87.6298,
            radius_m=500,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
        ),
    )
    # ids 1-4 are at the centre point; 5 is ungeocoded and 6 is ~2 km away.
    assert result.total == 4
    assert result.rows == ()
    assert result.excludes_ungeocoded is True
    assert sum(t.incidents for t in result.by_type) == result.total
    assert sum(r.incidents for r in result.rings) == result.total
    assert len(result.rings) == queries.DISTANCE_RINGS


@pytest.mark.integration
def test_nearby_radius_excludes_distant_rows(seeded):
    span = {"start": date(2025, 1, 1), "end": date(2025, 12, 31)}
    point = {"latitude": 41.8781, "longitude": -87.6298}
    assert queries.nearby(seeded, queries.NearbyQuery(radius_m=500, **point, **span)).total == 4
    assert queries.nearby(seeded, queries.NearbyQuery(radius_m=3000, **point, **span)).total == 5


@pytest.mark.integration
def test_nearby_include_rows_opts_into_detail(seeded):
    result = queries.nearby(
        seeded,
        queries.NearbyQuery(
            latitude=41.8781,
            longitude=-87.6298,
            radius_m=500,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            include_rows=True,
            limit=2,
        ),
    )
    assert len(result.rows) == 2
    assert result.truncated is True


@pytest.mark.integration
def test_nearby_reports_distances_in_metres_ascending(seeded):
    result = queries.nearby(
        seeded,
        queries.NearbyQuery(
            latitude=41.8781,
            longitude=-87.6298,
            radius_m=3000,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            nearest=5,
        ),
    )
    distances = [n.distance_m for n in result.nearest]
    assert distances == sorted(distances)
    assert distances[0] < 1  # the co-located rows sit on the query point
    assert 1000 < distances[-1] < 3000  # the ~2 km row, in metres not degrees
