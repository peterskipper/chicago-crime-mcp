"""Integrity tests for the three neighborhood reference files.

These assert on the *data files*, not on a loader, and they read them with
``json`` and ``pandas`` directly on purpose: a loader that coerces types would
hide exactly the malformation these are here to catch.

No network and no database. ``neighborhood_areas.csv`` happens to carry all 77
community areas -- every one of them intersects some neighborhood -- so the
curated ``containing`` targets can be checked against it offline.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from chicago_crime_mcp.reference import (
    NEIGHBORHOOD_ALIASES_PATH,
    NEIGHBORHOOD_AREAS_PATH,
    NEIGHBORHOOD_BOUNDARIES_PATH,
)

EXPECTED_NEIGHBORHOODS = 98
EXPECTED_COMMUNITY_AREAS = 77


@pytest.fixture(scope="module")
def boundaries() -> dict:
    return json.loads(NEIGHBORHOOD_BOUNDARIES_PATH.read_text())


@pytest.fixture(scope="module")
def areas() -> pd.DataFrame:
    return pd.read_csv(NEIGHBORHOOD_AREAS_PATH)


@pytest.fixture(scope="module")
def aliases() -> pd.DataFrame:
    return pd.read_csv(NEIGHBORHOOD_ALIASES_PATH, dtype={"target_value": str})


# --- the mirrored boundary file ---------------------------------------------


def test_boundaries_hold_98_distinctly_named_polygons(boundaries):
    names = [f["properties"]["pri_neigh"] for f in boundaries["features"]]
    assert len(names) == EXPECTED_NEIGHBORHOODS
    assert len(set(names)) == EXPECTED_NEIGHBORHOODS
    assert all(
        f["geometry"]["type"] in {"Polygon", "MultiPolygon"} for f in boundaries["features"]
    )


def test_boundaries_are_not_simplified(boundaries):
    """Guard the full-precision vendoring decision.

    Simplifying at ~1 m shrinks the file 6.4x but reassigns 2.4% of incidents,
    because each polygon is simplified independently and the shared edges come
    apart. The vertex count is what makes that regression visible: a
    well-meaning `ST_Simplify` would drop it by an order of magnitude.
    """

    def count(geometry: dict) -> int:
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        return sum(len(ring) for polygon in polygons for ring in polygon)

    vertices = sum(count(f["geometry"]) for f in boundaries["features"])
    assert vertices > 50_000, f"only {vertices} vertices - were the polygons simplified?"


# --- the derived containment table ------------------------------------------


def test_every_neighborhood_has_a_containing_community_area(boundaries, areas):
    names = {f["properties"]["pri_neigh"] for f in boundaries["features"]}
    assert set(areas["neighborhood"]) == names
    assert areas["community_area_number"].nunique() == EXPECTED_COMMUNITY_AREAS


def test_shares_are_fractions(areas):
    for column in ("share_of_neighborhood_in_ca", "share_of_ca_covered_by_neighborhood"):
        assert areas[column].between(0.0, 1.0).all(), f"{column} outside [0, 1]"


def test_each_neighborhood_is_fully_accounted_for(areas):
    """A neighborhood's shares should sum to 1: the community areas partition it.

    They fall a hair short because `MIN_SHARE` drops the edge slivers that come
    from two independently drawn boundary sets disagreeing along a shared line.
    A real gap -- a neighborhood whose polygon extends somewhere no community
    area covers -- would show up here as a much larger shortfall.
    """
    covered = areas.groupby("neighborhood")["share_of_neighborhood_in_ca"].sum()
    assert covered.between(0.99, 1.001).all(), covered[~covered.between(0.99, 1.001)]


def test_straddlers_are_the_only_repeated_neighborhoods(areas):
    repeated = areas["neighborhood"].value_counts().loc[lambda s: s > 1]
    assert sorted(repeated.index) == [
        "Englewood",
        "Garfield Park",
        "Humboldt Park",
        "Jackson Park",
        "Old Town",
        "Streeterville",
    ]
    assert (repeated == 2).all(), "a neighborhood spanning 3+ community areas is new"


# --- the curated alias table ------------------------------------------------


def test_alias_rows_are_well_formed(aliases):
    assert set(aliases["match_kind"]) == {"alias", "containing"}
    expected = aliases["match_kind"].map(
        {"alias": "neighborhood", "containing": "community_area"}
    )
    mismatched = aliases[aliases["target_geography"] != expected]
    assert mismatched.empty, mismatched[["alias", "match_kind", "target_geography"]]
    assert aliases["note"].str.strip().str.len().gt(0).all(), "every alias states its reason"


def test_every_alias_target_resolves(aliases, areas):
    known_neighborhoods = set(areas["neighborhood"])
    known_areas = set(
        zip(
            areas["community_area_number"].astype(str),
            areas["community_area"],
            strict=False,
        )
    )
    for row in aliases.itertuples():
        if row.match_kind == "alias":
            assert row.target_value in known_neighborhoods, row.alias
            assert row.target_value == row.target_label, row.alias
        else:
            assert (row.target_value, row.target_label) in known_areas, row.alias


def test_no_alias_shadows_a_real_neighborhood_name(aliases, areas):
    """An alias for a name tier 1 already resolves exactly would be dead weight.

    Worse, if the two ever disagreed the ladder would answer from the curated
    row instead of the polygon that actually exists.
    """
    normalized = {n.casefold() for n in areas["neighborhood"]}
    shadowing = [a for a in aliases["alias"] if a.casefold() in normalized]
    assert not shadowing, f"tier 1 already resolves these: {shadowing}"


def test_alias_keys_are_unique_after_normalization(aliases):
    keys = [" ".join(a.casefold().split()) for a in aliases["alias"]]
    assert len(set(keys)) == len(keys), "two rows normalize to the same lookup key"
