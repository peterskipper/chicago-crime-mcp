"""Tests for the filter coercion shared by both stores.

The point of these is the silent failure described in the module docstring: an
uncoerced filter does not error, it returns zero rows. So each test asserts the
coerced value matches how the column actually stores it, and the mapping tests
assert both stores can key off the same vocabulary.
"""

from __future__ import annotations

import pytest

from chicago_crime_mcp.store import normalize


def test_types_are_upper_cased_and_stripped():
    assert normalize.normalize_types((" burglary ", "Theft")) == ("BURGLARY", "THEFT")


def test_types_preserve_order_and_empty():
    assert normalize.normalize_types(()) == ()
    assert normalize.normalize_types(("theft", "arson")) == ("THEFT", "ARSON")


def test_beat_is_zero_padded_to_four():
    assert normalize.normalize_geography_values("beat", (111, "0234", " 1011 ")) == (
        "0111",
        "0234",
        "1011",
    )


def test_district_is_zero_padded_to_three():
    # The headline case: a model passing the integer 10 for district `010`
    # would otherwise match nothing and get a confident empty answer.
    assert normalize.normalize_geography_values("district", (10, "010", "7")) == (
        "010",
        "010",
        "007",
    )


def test_ward_and_community_area_are_coerced_to_int():
    assert normalize.normalize_geography_values("ward", ("2", 43, " 7 ")) == (2, 43, 7)
    assert normalize.normalize_geography_values("community_area", ("8",)) == (8,)


def test_citywide_drops_geography_values():
    # There is no geography column on a citywide query, so a filter on one is
    # not a narrower query -- it is a meaningless one.
    assert normalize.normalize_geography_values("citywide", ("0111", 4)) == ()


def test_non_numeric_ward_raises_naming_the_field():
    # The message has to name the geography: the server turns it into the
    # teaching error that tells the model which field it got wrong.
    with pytest.raises(ValueError, match="ward must be an integer"):
        normalize.normalize_geography_values("ward", ("Logan Square",))


def test_type_column_covers_both_taxonomies():
    assert normalize.TYPE_COLUMN["source"] == "primary_type_canonical"
    assert normalize.TYPE_COLUMN["comparable"] == "stable_category"


def test_geo_column_is_none_only_for_citywide():
    assert normalize.GEO_COLUMN["citywide"] is None
    assert all(
        column == geography
        for geography, column in normalize.GEO_COLUMN.items()
        if geography != "citywide"
    )


def test_padded_geographies_are_a_subset_of_the_geography_columns():
    # A width for a geography that no longer exists would silently never apply.
    assert set(normalize.PADDED_GEOGRAPHIES) <= set(normalize.GEO_COLUMN)


# --- neighborhood: the only free-text geography ------------------------------


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("Wicker Park", "Wicker Park"),
        ("wicker park", "Wicker Park"),
        ("WICKER  PARK", "Wicker Park"),
        ("  Wicker Park  ", "Wicker Park"),
        # Punctuation is part of the stored spelling and must come back intact.
        ("little italy, uic", "Little Italy, UIC"),
        ("o'hare", "O'Hare"),
        ("rush & division", "Rush & Division"),
        # The one rewriting rule normalization applies on its own.
        ("the loop", "Loop"),
    ],
)
def test_neighborhood_values_come_back_in_their_stored_spelling(supplied, expected):
    """The predicate must never carry the caller's spelling into SQL.

    Every other geography is a code, so coercing it is arithmetic. This one is a
    name, and a name that reaches the column in the wrong case matches nothing --
    a confident empty page rather than an error, which is the same failure class
    the zero-padding rules exist to prevent.
    """
    assert normalize.normalize_geography_values("neighborhood", (supplied,)) == (expected,)


def test_an_unrecognized_neighborhood_passes_through_untouched():
    """So the vocabulary check answers, with the list of names that do exist.

    Rejecting here would produce a worse error: this function knows the 98 names
    but not what the data currently holds, and it has nowhere to put a hint.
    """
    assert normalize.normalize_geography_values(
        "neighborhood", ("Bronzeville", "Nowhere At All")
    ) == ("Bronzeville", "Nowhere At All")


def test_aliases_are_not_applied_as_a_value_coercion():
    """Deliberate: `Pilsen` is a community area answer, not a neighborhood one.

    Substituting it here would silently change which geography the caller is
    filtering on. Turning a colloquial name into the right argument is
    resolve_neighborhood's job, and it has a return type that can say so.
    """
    assert normalize.normalize_geography_values("neighborhood", ("Pilsen",)) == ("Pilsen",)
    assert normalize.normalize_geography_values("neighborhood", ("Boys Town",)) == ("Boys Town",)


def test_neighborhood_values_keep_their_order_and_count():
    supplied = ("the loop", "Bronzeville", "wicker park")
    assert normalize.normalize_geography_values("neighborhood", supplied) == (
        "Loop", "Bronzeville", "Wicker Park",
    )


def test_every_stored_neighborhood_survives_a_round_trip():
    """A name already in its stored form must be returned identically.

    The punctuated names are the risk: if the match key were built one way and
    the map another, `Little Italy, UIC` would normalize to something that is not
    itself and match nothing.
    """
    from chicago_crime_mcp.geo.resolve import NeighborhoodIndex

    for name in NeighborhoodIndex.load().names:
        assert normalize.normalize_geography_values("neighborhood", (name,)) == (name,)


# --- every geography must be wired up everywhere -----------------------------


def test_every_geography_is_present_in_every_map_keyed_by_one():
    """Adding a `Geography` member is not one edit, and the type checker cannot say so.

    A `Literal` gives no runtime guarantee that the dicts keyed by it are
    exhaustive, so a new member reaches production as a KeyError on the first
    call that uses it -- or, worse, as a dimension `describe_schema` never
    mentions, which a model then never asks for.

    Adding `neighborhood` needed four edits, not the one the design predicted:
    the column map here, the rollup-table map in the DuckDB layer, the prose in
    `describe_schema`, and the tool schema's enum. This asserts the first three;
    the enum has its own check in test_app.
    """
    from typing import get_args

    from chicago_crime_mcp.server.schema import GEOGRAPHY_NOTES
    from chicago_crime_mcp.store.duckdb.queries import _ROLLUP_TABLE

    declared = set(get_args(normalize.Geography))
    assert set(normalize.GEO_COLUMN) == declared, "GEO_COLUMN"
    assert set(_ROLLUP_TABLE) == declared, "_ROLLUP_TABLE"
    assert set(GEOGRAPHY_NOTES) == declared, "GEOGRAPHY_NOTES"


def test_only_the_free_text_geography_forgoes_a_suggestion():
    """The suppression is keyed off TEXT_GEOGRAPHIES, so the sets stay in step.

    Beat and district are codes drawn from a complete set: a near miss there
    really is a typo, and withholding the suggestion would make those errors
    worse for no reason.
    """
    assert normalize.TEXT_GEOGRAPHIES == {"neighborhood"}
    assert not normalize.TEXT_GEOGRAPHIES & set(normalize.PADDED_GEOGRAPHIES)


def test_text_geographies_have_a_column_and_are_not_also_padded():
    """The three coercion rules are mutually exclusive; overlapping them is a bug."""
    for geography in normalize.TEXT_GEOGRAPHIES:
        assert normalize.GEO_COLUMN[geography] is not None
        assert geography not in normalize.PADDED_GEOGRAPHIES
