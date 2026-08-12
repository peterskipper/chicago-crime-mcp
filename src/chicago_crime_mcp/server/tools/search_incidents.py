"""The ``search_incidents`` tool: filtered, paginated rows from Postgres.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import Field

from chicago_crime_mcp.server.context import get_context
from chicago_crime_mcp.server.envelope import (
    Provenance,
    ResultWarning,
    RouteInfo,
    boundary_warning,
    empty_result_warning,
    truncated_warning,
)
from chicago_crime_mcp.server.errors import (
    InvalidArgumentError,
    StaleCursorError,
)
from chicago_crime_mcp.server.models import SearchFilters, SearchPayload, SearchResponse
from chicago_crime_mcp.server.tools._validate import (
    applied_filter_names,
    validated_geography_values,
    validated_range,
    validated_types,
)
from chicago_crime_mcp.store.normalize import Geography, Taxonomy
from chicago_crime_mcp.store.postgres import queries


def search_incidents(
    start: Annotated[date, Field(description="First day to include, inclusive.")],
    end: Annotated[date, Field(description="Last day to include, inclusive.")],
    types: Annotated[
        list[str] | None,
        Field(
            description="Offense categories to include, interpreted under `taxonomy`. Omit for "
            "all. Values must come from describe_schema."
        ),
    ] = None,
    taxonomy: Annotated[
        Taxonomy,
        Field(description="Which offense taxonomy `types` names and results are labelled in."),
    ] = "source",
    geography: Annotated[
        Geography,
        Field(description="Which geography dimension `geography_values` names."),
    ] = "citywide",
    geography_values: Annotated[
        list[str | int] | None,
        Field(
            description="Restrict to these values of `geography`. Omit for the whole city. "
            "Zero-padding is applied for you: district 10 is matched as '010'."
        ),
    ] = None,
    arrest: Annotated[
        bool | None, Field(description="Only offenses with/without an arrest. Omit for either.")
    ] = None,
    domestic: Annotated[
        bool | None, Field(description="Only domestic-flagged offenses. Omit for either.")
    ] = None,
    limit: Annotated[
        int, Field(description=f"Rows per page, at most {queries.MAX_LIMIT}.")
    ] = queries.DEFAULT_LIMIT,
    cursor: Annotated[
        str | None,
        Field(
            description="Continue a previous search. Pass the `cursor` from its response "
            "verbatim, with every other argument unchanged."
        ),
    ] = None,
) -> SearchResponse:
    """List individual offenses matching a set of filters, newest first.

    Use this when the question is about *which* offenses -- their locations,
    descriptions, dates. For "how many", use ``aggregate_incidents`` instead: it
    answers from pre-summed months, and asking this tool to list thousands of
    rows in order to count them is both slower and capped.

    Results are paginated by keyset, not by offset, so pages stay fast and do
    not shift when the nightly load lands a row into a page already read. When
    the response carries a ``cursor``, more matched: send it back with **every
    other argument unchanged** to continue. A cursor is bound to the filters it
    was issued for and will be rejected against different ones rather than
    silently returning a page from the wrong result set.

    Args:
        start: First day to include, inclusive.
        end: Last day to include, inclusive.
        types: Offense categories, interpreted under ``taxonomy``.
        taxonomy: Which offense taxonomy to filter and label by.
        geography: Which geography dimension to filter on.
        geography_values: Values of that dimension to restrict to.
        arrest: Restrict to offenses with or without an arrest.
        domestic: Restrict to domestic-flagged offenses.
        limit: Rows per page.
        cursor: Cursor from a previous response.

    Returns:
        A page of compacted offense rows in the standard result envelope. Pass
        any ``id`` to ``get_incident`` for the full record.

    Raises:
        InvalidArgumentError: If the range is inverted or the limit is out of
            range.
        UnknownValueError: If a category or geography value does not exist.
        StaleCursorError: If the cursor was issued for different filters.
    """
    context = get_context()
    vocabulary = context.vocabulary()

    validated_range(start, end)
    if not 1 <= limit <= queries.MAX_LIMIT:
        raise InvalidArgumentError(
            f"limit must be between 1 and {queries.MAX_LIMIT}.",
            field="limit",
            received=limit,
            hint="Broad questions are better answered by aggregate_incidents than by a long list.",
        )
    resolved_types = validated_types(vocabulary, types, taxonomy)
    resolved_values = validated_geography_values(vocabulary, geography, geography_values)

    query = queries.SearchQuery(
        start=start,
        end=end,
        types=resolved_types,
        taxonomy=taxonomy,
        geography=geography,
        geography_values=resolved_values,
        arrest=arrest,
        domestic=domestic,
        limit=limit,
        cursor=cursor,
    )

    with context.postgres() as conn:
        try:
            result = queries.search(conn, query)
        except ValueError as exc:
            # Everything else this call validates -- the range, the limit, the
            # category and geography values -- was checked above, so a ValueError
            # from here is the cursor. Reported as its own error kind because the
            # remedy is unique: drop the cursor, do not edit the filters.
            if cursor is not None:
                raise StaleCursorError(str(exc), field="cursor", received=cursor) from exc
            raise InvalidArgumentError.from_value_error(exc) from exc

    warnings: list[ResultWarning] = []
    if result.truncated:
        warnings.append(
            truncated_warning(returned=len(result.rows), has_cursor=result.next_cursor is not None)
        )
    if not result.rows:
        warnings.append(
            empty_result_warning(
                filters=applied_filter_names(
                    types=resolved_types,
                    geography_values=resolved_values,
                    arrest=arrest,
                    domestic=domestic,
                )
            )
        )
    boundary = boundary_warning(geography=geography, start=start, end=end)
    if boundary is not None and resolved_values:
        # Only when the geography is a *filter*. Search does not group, so an
        # unfiltered search is not comparing anything across a redrawn boundary.
        warnings.append(boundary)

    return SearchResponse(
        data=SearchPayload.from_store(result),
        filters_applied=SearchFilters.from_query(result.query),
        row_count=len(result.rows),
        truncated=result.truncated,
        cursor=result.next_cursor,
        route=RouteInfo.from_store(result.timing, store="postgres"),
        taxonomy_mode=taxonomy,
        warnings=warnings,
        provenance=Provenance.from_dataset_meta(vocabulary.dataset),
    )


__all__ = ["search_incidents"]
