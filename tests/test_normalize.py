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
