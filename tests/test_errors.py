"""Tests for the structured tool errors.

The rendered string is the contract: MCP transmits a failure as text, so
anything these errors know that does not reach ``__str__`` never reaches the
model. These assert on that string, not only on the attributes.
"""

from __future__ import annotations

import pytest

from chicago_crime_mcp.server.errors import (
    MAX_LISTED_VALUES,
    DataUnavailableError,
    InvalidArgumentError,
    StaleCursorError,
    ToolError,
    UnknownValueError,
    suggest,
)

CATEGORIES = ("ARSON", "ASSAULT", "BATTERY", "BURGLARY", "HOMICIDE", "THEFT")


def test_suggest_finds_the_intended_category_from_a_typo():
    assert suggest("BURGLERY", CATEGORIES) == "BURGLARY"


def test_suggest_is_case_insensitive():
    assert suggest("burglary", CATEGORIES) == "BURGLARY"


def test_suggest_declines_when_nothing_is_close():
    # A confidently wrong suggestion is worse than none: the model will take it.
    assert suggest("ZZZZZZZZ", CATEGORIES) is None


def test_unknown_value_infers_the_nearest_match():
    err = UnknownValueError("Unknown category.", field="types", received="THEFF",
                            valid_values=CATEGORIES)
    assert err.nearest_match == "THEFT"


def test_an_explicit_nearest_match_is_not_overridden():
    err = UnknownValueError("Unknown category.", field="types", received="THEFF",
                            valid_values=CATEGORIES, nearest_match="ARSON")
    assert err.nearest_match == "ARSON"


def test_nearest_match_is_not_inferred_for_a_non_string_value():
    err = UnknownValueError("Unknown ward.", field="geography_values", received=999,
                            valid_values=("1", "2"))
    assert err.nearest_match is None


def test_the_rendered_message_carries_field_value_suggestion_and_options():
    err = UnknownValueError("Unknown offense category.", field="types", received="BURGLERY",
                            valid_values=CATEGORIES)
    rendered = str(err)
    assert "Unknown offense category." in rendered
    assert "types" in rendered and "BURGLERY" in rendered
    assert "Did you mean 'BURGLARY'?" in rendered
    assert "'ASSAULT'" in rendered


def test_the_suggestion_precedes_the_inventory():
    # A model that stops reading early should already have the answer.
    rendered = str(
        UnknownValueError("Unknown.", field="types", received="BURGLERY", valid_values=CATEGORIES)
    )
    assert rendered.index("Did you mean") < rendered.index("Valid values")


def test_long_value_lists_truncate_in_the_message_but_not_in_the_details():
    values = tuple(f"AREA_{i}" for i in range(MAX_LISTED_VALUES + 10))
    err = UnknownValueError("Unknown area.", field="geography_values", received="AREA_X",
                            valid_values=values)
    rendered = str(err)
    assert "(10 more)" in rendered
    assert values[-1] not in rendered
    assert len(err.details()["valid_values"]) == len(values)


def test_details_omits_what_was_never_set():
    err = ToolError("Something went wrong.")
    assert err.details() == {"code": "invalid_argument", "message": "Something went wrong."}


def test_details_carries_the_code_for_telemetry():
    assert UnknownValueError("x").details()["code"] == "unknown_value"
    assert StaleCursorError("x").details()["code"] == "stale_cursor"
    assert DataUnavailableError("x").details()["code"] == "unavailable"
    assert InvalidArgumentError("x").details()["code"] == "invalid_argument"


def test_a_stale_cursor_states_the_remedy_by_default():
    assert "without a cursor" in str(StaleCursorError("Cursor belongs to another query."))


def test_an_explicit_hint_overrides_the_stale_cursor_default():
    err = StaleCursorError("Cursor expired.", hint="Start over.")
    assert "Start over." in str(err) and "without a cursor" not in str(err)


def test_a_store_value_error_converts_without_losing_its_message():
    exc = ValueError("end (2024-01-01) is before start (2025-01-01)")
    err = InvalidArgumentError.from_value_error(exc, field="end")
    assert err.field == "end"
    assert "is before start" in str(err)


def test_errors_are_raisable_and_catchable_as_one_family():
    with pytest.raises(ToolError):
        raise StaleCursorError("nope")
