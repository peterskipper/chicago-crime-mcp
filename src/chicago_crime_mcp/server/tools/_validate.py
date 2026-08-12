"""Argument checking shared by the tools -- affordance #2, the teaching errors.

The division of labour here is deliberate and worth stating once:

* **Static closed sets** (``taxonomy``, ``grain``, ``geography``) are declared as
  ``Literal`` in the tool signatures, so FastMCP publishes them in the tool's
  JSON schema and the framework rejects a bad value before a call is ever made.
  Re-checking them here would be a second, worse copy of the same rule.
* **Data-derived closed sets** (offense categories, the geography values that
  actually occur) cannot be a ``Literal``: they are properties of the loaded
  data, and a set frozen at import time is wrong the first time CPD mints a
  code. Those are checked here, against
  :class:`~chicago_crime_mcp.server.vocabulary.Vocabulary`.
* **Cross-argument rules** (an inverted date range, both identifiers at once)
  belong to no single field's type, and are checked here or in the store's
  ``__post_init__``, whose ``ValueError`` is adapted rather than re-raised raw.

Every rejection names the offending argument, echoes what it received, offers
the nearest valid value when there is one, and lists the valid set. That is what
makes the failure a step in the conversation rather than the end of it.

Values are **normalized before they are validated**, never after. A caller
passing district ``10`` means ``"010"``, and validating the raw form would
reject a request the store would have answered perfectly.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from collections.abc import Sequence

from chicago_crime_mcp.server.errors import InvalidArgumentError, UnknownValueError
from chicago_crime_mcp.server.vocabulary import Vocabulary
from chicago_crime_mcp.store.normalize import (
    Geography,
    Taxonomy,
    normalize_geography_values,
    normalize_types,
)


def validated_types(
    vocabulary: Vocabulary, types: Sequence[str] | None, taxonomy: Taxonomy
) -> tuple[str, ...]:
    """Normalize offense categories and reject any the data does not contain.

    Every value is checked, not just the first bad one's presence: a list
    filter is an ``OR``, so one valid category is enough to return a confident
    non-empty page with the typo silently dropped. That failure is invisible to
    the caller, which is why this is checked up front rather than inferred from
    an empty result.

    Args:
        vocabulary: The valid values for the current build.
        types: Categories as supplied, in any case. None or empty means all.
        taxonomy: Which taxonomy the categories are expressed in, which changes
            what is valid.

    Returns:
        The categories, upper-cased, in the order given.

    Raises:
        UnknownValueError: If any category is not in the vocabulary.
    """
    if not types:
        return ()
    normalized = normalize_types(types)
    valid = vocabulary.categories_for(taxonomy)
    unknown = [t for t in normalized if t not in set(valid)]
    if unknown:
        raise UnknownValueError(
            f"{len(unknown)} offense category value(s) do not exist under the "
            f"'{taxonomy}' taxonomy.",
            field="types",
            received=unknown[0] if len(unknown) == 1 else unknown,
            valid_values=valid,
            hint=(
                None
                if taxonomy == "source"
                else "These are the 'comparable' taxonomy's categories; the 'source' taxonomy "
                "has a different set."
            ),
        )
    return normalized


def validated_geography_values(
    vocabulary: Vocabulary, geography: Geography, values: Sequence[str | int] | None
) -> tuple[str | int, ...]:
    """Coerce geography values to the column's type and reject unknown ones.

    Coercion comes first and is the more common fix: a district passed as ``10``
    matches nothing against a zero-padded column, and that is a silent empty
    answer rather than an error. Only once a value is in the column's own form
    is it meaningful to ask whether the data contains it.

    Args:
        vocabulary: The valid values for the current build.
        geography: Which dimension the values name.
        values: The values as supplied. None or empty means no filter.

    Returns:
        The coerced values, in the order given. Empty for ``citywide``, which
        has no column to filter on.

    Raises:
        InvalidArgumentError: If a value cannot be coerced to the column's type.
        UnknownValueError: If a coerced value does not occur in the data.
    """
    if not values or geography == "citywide":
        return ()
    try:
        normalized = normalize_geography_values(geography, values)
    except ValueError as exc:
        raise InvalidArgumentError.from_value_error(
            exc,
            field="geography_values",
            hint=f"Values for geography='{geography}' must be integers.",
        ) from exc

    valid = vocabulary.values_for(geography)
    unknown = [v for v in normalized if v not in set(valid)]
    if unknown:
        raise UnknownValueError(
            f"{len(unknown)} {geography} value(s) do not occur in the data.",
            field="geography_values",
            received=unknown[0] if len(unknown) == 1 else unknown,
            valid_values=tuple(str(v) for v in valid),
            hint=(
                f"Values for geography='{geography}' are matched in their stored form; "
                "describe_schema lists every one that occurs."
            ),
        )
    return normalized


def validated_range(start: object, end: object) -> None:
    """Reject an inverted date range before the store has to.

    The store checks this too, and its message is perfectly good. Checking here
    as well buys the argument names as the tool declares them, which is what the
    caller has to change.

    Args:
        start: First day requested.
        end: Last day requested, inclusive.

    Raises:
        InvalidArgumentError: If ``end`` precedes ``start``.
    """
    if end < start:  # type: ignore[operator]
        raise InvalidArgumentError(
            f"end ({end}) is before start ({start}).",
            field="end",
            received=str(end),
            hint="Date ranges are inclusive of both ends; pass end on or after start.",
        )


def applied_filter_names(**filters: object) -> tuple[str, ...]:
    """Name the filters that were actually in force, for the empty-result warning.

    An empty page is indistinguishable from a filter that quietly matched
    nothing, so the warning names what was applied and lets the caller judge.

    Args:
        **filters: Candidate filters by argument name. Ones that are None,
            empty, or otherwise falsy are treated as not applied -- except
            ``False``, which is a real filter value for ``arrest`` and
            ``domestic``.

    Returns:
        The names of the filters that were applied, in the order given.
    """
    return tuple(
        name for name, value in filters.items() if value is not None and (value or value is False)
    )
