"""Unit tests for the neighborhood resolution ladder.

Pure table lookup -- no geometry, no network, no database -- which is the point
of keeping :mod:`geo.resolve` free of the spatial extension. These run against
the real vendored reference files, because the curated alias rows are half the
behaviour under test and a fixture with invented aliases would test nothing that
ships.
"""

from __future__ import annotations

import pytest

from chicago_crime_mcp.geo.resolve import (
    MAX_SUGGESTIONS,
    NeighborhoodIndex,
    match_key,
)


@pytest.fixture(scope="module")
def index() -> NeighborhoodIndex:
    return NeighborhoodIndex.load()


# --- normalization ----------------------------------------------------------


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("Wicker Park", "wicker park"),
        ("WICKER PARK", "wicker park"),
        ("  Wicker   Park  ", "wicker park"),
        ("The Loop", "loop"),
        ("the  loop", "loop"),
        # Only a *leading* article goes. "Back of the Yards" keeps its interior
        # one, which is why that name needs an alias row rather than a rule.
        ("Back of the Yards", "back of the yards"),
        # Punctuation survives on purpose: these are three different places.
        ("Little Italy, UIC", "little italy, uic"),
        ("O'Hare", "o'hare"),
        ("Rush & Division", "rush & division"),
    ],
)
def test_match_key_normalizes(supplied, expected):
    assert match_key(supplied) == expected


def test_the_98_keys_do_not_collide(index):
    """Normalization must not merge two real neighborhoods into one key."""
    keys = [match_key(name) for name in index.names]
    assert len(set(keys)) == len(index.names)


# --- the ladder, rung by rung -----------------------------------------------


def test_tier_1_matches_a_stored_name_exactly(index):
    resolution = index.resolve("wicker  PARK")
    assert resolution.resolved
    (candidate,) = resolution.candidates
    assert candidate.match_kind == "exact"
    assert candidate.geography == "neighborhood"
    assert candidate.value == "Wicker Park", "must return the stored spelling, not the caller's"
    assert candidate.score is None


def test_tier_1_absorbs_the_leading_article(index):
    """`the loop` is the single case normalization alone was designed to fix."""
    resolution = index.resolve("the loop")
    (candidate,) = resolution.candidates
    assert candidate.match_kind == "exact"
    assert candidate.value == "Loop", "difflib would answer West Loop, 2.4 km away"


def test_tier_2_alias_still_yields_a_polygon(index):
    resolution = index.resolve("Boys Town")
    (candidate,) = resolution.candidates
    assert candidate.match_kind == "alias"
    assert candidate.geography == "neighborhood"
    assert candidate.value == "Boystown"
    assert candidate.label == "Boystown", "the label is the place, not what was typed"


def test_tier_3_widens_to_a_community_area_and_says_so(index):
    resolution = index.resolve("Pilsen")
    (candidate,) = resolution.candidates
    assert candidate.match_kind == "containing"
    assert candidate.geography == "community_area"
    assert candidate.value == 31
    assert candidate.label == "Pilsen"
    (area,) = candidate.containing_areas
    assert area.name == "LOWER WEST SIDE"
    # Not "unknown": there is no polygon to intersect, so the overlap is
    # unmeasurable, and the tool layer has to say that rather than guess.
    assert area.share_of_neighborhood is None
    assert area.share_of_area is None


def test_tier_4_suggests_without_resolving(index):
    resolution = index.resolve("Bronzevile")
    assert not resolution.resolved, "a near miss must never be applied automatically"
    assert all(c.match_kind == "suggestion" for c in resolution.candidates)
    assert len(resolution.candidates) <= MAX_SUGGESTIONS
    best = resolution.candidates[0]
    assert best.label == "Bronzeville"
    assert 0.0 < best.score <= 1.0


def test_tier_5_misses_rather_than_guessing(index):
    resolution = index.resolve("Xzzyqwv Heights")
    assert resolution.candidates == ()
    assert not resolution.resolved
    assert resolution.query == "Xzzyqwv Heights", "echoed back for the teaching error"


# --- the safety property the alias table exists for -------------------------


@pytest.mark.parametrize(
    ("supplied", "expected_area", "wrong_answer"),
    [
        ("Bronzeville", "GRAND BOULEVARD", "Andersonville"),
        ("Roscoe Village", "NORTH CENTER", "East Village"),
        ("South Loop", "NEAR SOUTH SIDE", "West Loop"),
    ],
)
def test_curated_names_never_fall_through_to_fuzzy(
    index, supplied, expected_area, wrong_answer
):
    """These are the measured failures: difflib answers each one confidently wrong.

    Bronzeville to Andersonville is 19.4 km, opposite ends of the city, 417
    incidents against 3,306. An alias row is what stands between the two.
    """
    resolution = index.resolve(supplied)
    assert resolution.resolved
    (candidate,) = resolution.candidates
    assert candidate.match_kind == "containing"
    assert candidate.containing_areas[0].name == expected_area
    assert candidate.label != wrong_answer


# --- containment facts travel with the answer -------------------------------


def test_an_exact_match_reports_how_diluted_a_community_area_answer_would_be(index):
    (candidate,) = index.resolve("Wicker Park").candidates
    (area,) = candidate.containing_areas
    assert area.name == "WEST TOWN"
    # ~21%: answering from the community area covers roughly five times the
    # ground that was asked about.
    assert 0.15 < area.share_of_area < 0.25
    assert area.share_of_neighborhood == pytest.approx(1.0, abs=0.01)


def test_a_straddler_reports_both_areas_widest_first(index):
    (candidate,) = index.resolve("Old Town").candidates
    assert [a.name for a in candidate.containing_areas] == [
        "LINCOLN PARK",
        "NEAR NORTH SIDE",
    ]
    shares = [a.share_of_neighborhood for a in candidate.containing_areas]
    assert shares == sorted(shares, reverse=True)
    assert sum(shares) == pytest.approx(1.0, abs=0.01)


def test_neighborhoods_differ_in_dilution(index):
    """Guard against the shares being constant, which would make the above vacuous."""
    dilution = {
        name: index.resolve(name).candidates[0].containing_areas[0].share_of_area
        for name in ("Greektown", "Wicker Park", "Chinatown", "Little Village")
    }
    assert len(set(dilution.values())) == 4
    assert dilution["Greektown"] < dilution["Wicker Park"] < dilution["Chinatown"]
    assert dilution["Little Village"] == pytest.approx(1.0, abs=0.01)


def test_every_stored_name_resolves_to_itself(index):
    """The 98 are the set a teaching error advertises; each must actually work."""
    for name in index.names:
        resolution = index.resolve(name)
        assert resolution.resolved, name
        assert resolution.candidates[0].value == name


def test_every_alias_resolves(index):
    """No curated row may be dead -- a typo'd alias key would silently do nothing."""
    import pandas as pd

    from chicago_crime_mcp.reference import NEIGHBORHOOD_ALIASES_PATH

    for alias in pd.read_csv(NEIGHBORHOOD_ALIASES_PATH)["alias"]:
        resolution = index.resolve(alias)
        assert resolution.resolved, alias
        assert resolution.candidates[0].match_kind in {"alias", "containing"}, alias
