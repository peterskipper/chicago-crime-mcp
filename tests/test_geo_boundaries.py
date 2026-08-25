"""Tests for point-in-polygon neighborhood tagging.

Marked ``spatial``: every one of these needs DuckDB's spatial extension, which
downloads on first use. They are excluded from `make test` so the default run
stays offline -- see the marker definitions in pyproject.

The logic tests build their own two-square world rather than using the vendored
polygons, so a failure points at the join and not at Chicago. Two tests do use
the real file, because "the 98 names load" and "these coordinates are in Wicker
Park" are claims about the shipped data that nothing else checks.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from chicago_crime_mcp.geo.boundaries import NeighborhoodBoundaries

pytestmark = pytest.mark.spatial

#: Two unit squares sharing the edge x=1, in a coordinate space where reasoning
#: about what is inside what takes no map. `locate` takes (lat, lon) = (y, x).
WEST_SQUARE = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"
EAST_SQUARE = "POLYGON((1 0, 2 0, 2 1, 1 1, 1 0))"


def _boundaries(
    conn: duckdb.DuckDBPyConnection, polygons: dict[str, str]
) -> NeighborhoodBoundaries:
    """Build a NeighborhoodBoundaries over hand-written WKT polygons."""
    values = ", ".join(
        f"('{name}', ST_GeomFromText('{wkt}'))" for name, wkt in polygons.items()
    )
    conn.execute(
        f"CREATE TABLE neighborhoods AS "
        f"SELECT * FROM (VALUES {values}) t(neighborhood, geom)"
    )
    return NeighborhoodBoundaries(conn)


@pytest.fixture
def squares(spatial_conn):
    return _boundaries(spatial_conn, {"West": WEST_SQUARE, "East": EAST_SQUARE})


# --- the join ---------------------------------------------------------------


def test_points_land_in_the_polygon_that_contains_them(squares):
    """Vary the coordinate, not just the row: a constant would pass either way."""
    lat = pd.Series([0.5, 0.5, 0.5])
    lon = pd.Series([0.5, 1.5, 9.9])
    located = squares.locate(lat, lon)
    assert located[:2].tolist() == ["West", "East"]
    assert located.isna().tolist() == [False, False, True]


def test_moving_a_point_across_the_edge_changes_its_neighborhood(squares):
    """The mutation check, inlined: the answer must track the input."""
    lat = pd.Series([0.5])
    assert squares.locate(lat, pd.Series([0.25])).tolist() == ["West"]
    assert squares.locate(lat, pd.Series([1.75])).tolist() == ["East"]


def test_a_point_outside_every_polygon_is_null_not_an_error(squares):
    """~0.3% of geocoded incidents land in the lake or on the airport."""
    located = squares.locate(pd.Series([50.0]), pd.Series([50.0]))
    assert located.isna().all()


def test_missing_coordinates_yield_null(squares):
    """1.55% of rows have no coordinates at all. That is data, not a failure."""
    located = squares.locate(pd.Series([None, 0.5]), pd.Series([None, 0.5]))
    assert located.isna().tolist() == [True, False]


def test_a_shared_edge_belongs_to_neither(squares):
    """ST_Contains excludes the boundary, which is what keeps the column single-valued.

    The alternative, ST_Intersects, would put a point on x=1 in both squares and
    make the join return more rows than it was given.
    """
    assert squares.locate(pd.Series([0.5]), pd.Series([1.0])).isna().all()


# --- shape of the result ----------------------------------------------------


def test_the_callers_index_is_preserved(squares):
    """A partition is not indexed 0..n, and a misaligned assignment is silent."""
    index = [17, 4, 99]
    located = squares.locate(
        pd.Series([0.5, 0.5, 0.5], index=index), pd.Series([1.5, 0.5, 1.5], index=index)
    )
    assert located.index.tolist() == index
    assert located.to_dict() == {17: "East", 4: "West", 99: "East"}


def test_the_result_is_a_nullable_string_series(squares):
    """Parquet partitions must share a schema; object dtype would not survive."""
    assert squares.locate(pd.Series([0.5]), pd.Series([0.5])).dtype == "string"


def test_an_empty_frame_is_fine(squares):
    assert squares.locate(pd.Series([], dtype=float), pd.Series([], dtype=float)).empty


# --- the disjointness invariant --------------------------------------------


def test_overlapping_polygons_raise_rather_than_pick_a_winner(spatial_conn):
    """If the polygons ever overlap the column stops being single-valued.

    Silently keeping one row would produce a plausible, arbitrary answer. The
    join is expected to be row-preserving, so a row count that grew is the
    cheapest possible detection of that.
    """
    overlapping = _boundaries(
        spatial_conn,
        {"West": WEST_SQUARE, "Wider": "POLYGON((0 0, 2 0, 2 1, 0 1, 0 0))"},
    )
    with pytest.raises(ValueError, match="no longer disjoint"):
        overlapping.locate(pd.Series([0.5]), pd.Series([0.5]))


# --- the vendored polygons --------------------------------------------------


def test_the_real_boundaries_load_all_98_names():
    with NeighborhoodBoundaries.load() as boundaries:
        assert len(boundaries.names) == 98
        assert boundaries.names == tuple(sorted(boundaries.names))
        assert "Wicker Park" in boundaries.names


@pytest.mark.parametrize(
    ("latitude", "longitude", "expected"),
    [
        (41.9088, -87.6796, "Wicker Park"),  # Milwaukee/North/Damen
        (41.8781, -87.6298, "Loop"),  # State and Madison
        (41.9803, -87.9090, "O'Hare"),  # a name whose punctuation survives normalization
        # Pilsen has no polygon of its own; the point falls in the community
        # area that contains it, which is exactly why tier 3 of the ladder
        # points at Lower West Side.
        (41.8574, -87.6668, "Lower West Side"),
        (41.8800, -87.5500, None),  # five km out into Lake Michigan
        (42.0451, -87.6877, None),  # Evanston: outside the city entirely
    ],
)
def test_known_chicago_coordinates_land_where_expected(latitude, longitude, expected):
    with NeighborhoodBoundaries.load() as boundaries:
        located = boundaries.locate(pd.Series([latitude]), pd.Series([longitude]))
    assert (located.iloc[0] if located.notna().iloc[0] else None) == expected
