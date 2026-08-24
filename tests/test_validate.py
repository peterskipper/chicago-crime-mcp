"""Tests for the shared argument checks the tools run before touching a store.

`validated_geography_values` is the seam where a caller's spelling becomes a
predicate, so it is where a wrong answer would be silent rather than loud. These
use a stub vocabulary: the function only ever asks it for `values_for`, and a
real one needs a built rollup database to exist.
"""

from __future__ import annotations

import pytest

from chicago_crime_mcp.server.errors import UnknownValueError
from chicago_crime_mcp.server.tools._validate import validated_geography_values

NEIGHBORHOODS = ("Andersonville", "Grand Boulevard", "Little Italy, UIC", "Wicker Park")
COMMUNITY_AREAS = (24, 38)


class _Vocabulary:
    """The one method `validated_geography_values` uses."""

    def values_for(self, geography):
        return {"neighborhood": NEIGHBORHOODS, "community_area": COMMUNITY_AREAS}[geography]


@pytest.fixture
def vocabulary():
    return _Vocabulary()


def test_a_neighborhood_is_coerced_to_its_stored_spelling(vocabulary):
    assert validated_geography_values(vocabulary, "neighborhood", ["wicker  PARK"]) == (
        "Wicker Park",
    )
    assert validated_geography_values(
        vocabulary, "neighborhood", ["little italy, uic"]
    ) == ("Little Italy, UIC",)


def test_an_unknown_neighborhood_gets_no_fuzzy_suggestion(vocabulary):
    """The whole point of the free-text carve-out, asserted through the real path.

    Bronzeville is a real Chicago neighborhood with no boundary of its own.
    difflib's nearest match among these values is Andersonville, 19.4 km away --
    a fluent, confident, wrong answer that a model would act on.
    """
    with pytest.raises(UnknownValueError) as exc:
        validated_geography_values(vocabulary, "neighborhood", ["Bronzeville"])
    assert exc.value.nearest_match is None
    assert "Andersonville" not in (exc.value.hint or "")
    assert "resolve_neighborhood" in exc.value.hint


def test_the_error_still_lists_what_does_exist(vocabulary):
    """Withholding the guess must not mean withholding the vocabulary."""
    with pytest.raises(UnknownValueError) as exc:
        validated_geography_values(vocabulary, "neighborhood", ["Bronzeville"])
    assert exc.value.valid_values == NEIGHBORHOODS


def test_a_coded_geography_keeps_its_suggestion(vocabulary):
    """The carve-out is for free text only.

    Beat, district, ward and community area come from complete sets -- every
    value the data holds is listed -- so a near miss really is a typo and the
    suggestion is the most useful thing the error can offer.
    """
    with pytest.raises(UnknownValueError) as exc:
        validated_geography_values(vocabulary, "community_area", [999])
    assert "resolve_neighborhood" not in (exc.value.hint or "")


def test_citywide_has_no_values_to_validate(vocabulary):
    assert validated_geography_values(vocabulary, "citywide", ["anything"]) == ()
    assert validated_geography_values(vocabulary, "neighborhood", None) == ()
