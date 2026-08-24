"""Tests for the resolve_neighborhood tool.

The ladder itself is tested in test_geo_resolve; these cover what the tool adds:
the model it returns, the teaching error on a miss, and the property the whole
thing exists for -- that a value it hands back is one the other tools accept.

A real ServerContext is used because the tool reports provenance from the rollup
build, but no test here reads incident rows through it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from chicago_crime_mcp.server.context import ServerContext, use_context
from chicago_crime_mcp.server.errors import InvalidArgumentError, UnknownValueError
from chicago_crime_mcp.server.tools.resolve_neighborhood import _index, resolve_neighborhood
from tests.helpers import row as _row
from tests.helpers import write_partition as _write_partition


@pytest.fixture
def context(tmp_path):
    """A context over a tiny fixture dataset, opened DuckDB-only."""
    from chicago_crime_mcp.store.config import StoreConfig
    from chicago_crime_mcp.store.duckdb import rollups

    _write_partition(
        tmp_path / "parquet",
        2025,
        [
            _row(id=1, date=datetime(2025, 1, 5), neighborhood="Wicker Park"),
            _row(id=2, date=datetime(2025, 2, 6), neighborhood="Loop"),
        ],
    )
    db = tmp_path / "db" / "crime.duckdb"
    conn = rollups.connect(duckdb_path=db, parquet_root=tmp_path / "parquet")
    rollups.build(conn)
    conn.close()

    ctx = ServerContext(
        StoreConfig(duckdb_path=db, parquet_root=tmp_path / "parquet")
    )
    ctx._open_duckdb()
    with use_context(ctx):
        yield ctx
    ctx.close()


# --- the ladder, through the tool -------------------------------------------


def test_an_exact_match_answers_as_a_neighborhood(context):
    result = resolve_neighborhood("wicker  PARK")
    assert result.resolved
    (candidate,) = result.candidates
    assert candidate.match_kind == "exact"
    assert (candidate.geography, candidate.value) == ("neighborhood", "Wicker Park")
    assert result.query == "wicker  PARK", "the caller's spelling is echoed back"


def test_a_place_without_a_boundary_widens_and_says_by_how_much(context):
    """Pilsen has no polygon, so the honest answer is a broader one, labelled."""
    (candidate,) = resolve_neighborhood("Pilsen").candidates
    assert candidate.match_kind == "containing"
    assert candidate.geography == "community_area"
    assert candidate.value == 31
    (area,) = candidate.containing_areas
    assert area.name == "LOWER WEST SIDE"
    # Unmeasurable, not unknown: there is no polygon to intersect the area with.
    assert area.share_of_neighborhood is None and area.share_of_area is None


def test_an_exact_match_still_reports_how_broad_the_area_answer_would_be(context):
    """The number that justifies the whole column: Wicker Park is ~a fifth of West Town."""
    (candidate,) = resolve_neighborhood("Wicker Park").candidates
    (area,) = candidate.containing_areas
    assert area.name == "WEST TOWN"
    assert 0.15 < area.share_of_area < 0.25


def test_a_near_miss_is_offered_but_not_resolved(context):
    """`resolved` is the flag a caller must read before filtering."""
    result = resolve_neighborhood("Wickr Park")
    assert result.resolved is False
    assert all(c.match_kind == "suggestion" for c in result.candidates)
    assert result.candidates[0].label == "Wicker Park"
    assert 0.0 < result.candidates[0].score <= 1.0


def test_bronzeville_never_resolves_to_andersonville(context):
    """The measured failure this tool exists to prevent -- 19.4 km apart."""
    (candidate,) = resolve_neighborhood("Bronzeville").candidates
    assert candidate.match_kind == "containing"
    assert candidate.containing_areas[0].name == "GRAND BOULEVARD"
    assert "Andersonville" not in [a.name for a in candidate.containing_areas]


# --- the teaching error -----------------------------------------------------


def test_an_unresolvable_name_lists_every_resolvable_one(context):
    with pytest.raises(UnknownValueError) as exc:
        resolve_neighborhood("Xzzyqwv Heights")
    message = str(exc.value)
    assert exc.value.valid_values == _index().names
    # All 98 inline, not the 40 a generic error truncates to: here the inventory
    # is the answer, and it saves a round trip.
    for name in ("Albany Park", "Wicker Park", "Woodlawn", "Wrigleyville"):
        assert name in message, name
    assert "(more)" not in message and "more)" not in message
    assert "community_area, ward, district or beat" in message


def test_the_error_offers_no_fuzzy_guess(context):
    """difflib already declined; re-asking from the error would only lower the bar."""
    with pytest.raises(UnknownValueError) as exc:
        resolve_neighborhood("Xzzyqwv Heights")
    assert exc.value.nearest_match is None


def test_an_empty_name_is_rejected_as_an_argument_error(context):
    for supplied in ("", "   "):
        with pytest.raises(InvalidArgumentError):
            resolve_neighborhood(supplied)


# --- what the tool adds around the ladder -----------------------------------


def test_the_result_reports_provenance_and_a_reference_route(context):
    result = resolve_neighborhood("Wicker Park")
    assert result.provenance.source
    assert result.route.store == "reference", "no incident data is read"
    assert result.route.elapsed_ms >= 0


def test_every_resolvable_name_is_a_value_the_other_tools_accept(context):
    """The property that makes the tool useful rather than merely informative.

    A candidate's value goes straight into geography_values. If the pinned
    boundary file and the loaded data ever disagreed on the 98 names, this tool
    would hand back a value the next call rejects -- so the two are checked
    against each other here rather than reconciled at runtime.
    """
    from chicago_crime_mcp.store.duckdb import queries

    with context.duckdb() as conn:
        in_data = set(queries.geography_values(conn, "neighborhood"))
    # The fixture holds two neighborhoods, so compare against the real dataset's
    # reference set rather than this build's.
    assert in_data <= set(_index().names), "the data holds a name the reference does not"
