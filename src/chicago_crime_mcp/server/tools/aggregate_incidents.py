"""The ``aggregate_incidents`` tool: counts over time, from the DuckDB rollups.

This is where the taxonomy obligations land, because this is the tool a trend
question reaches:

1. ``taxonomy`` is an explicit parameter defaulting to ``source``. It is never
   inferred from the question and never auto-switched on a long date range --
   choosing how to group offense codes is an analytic decision, not one a server
   makes on the caller's behalf.
2. The envelope names the mode that was applied, in **both** modes including the
   default. That is structural: ``taxonomy_mode`` is a top-level field of
   :class:`~chicago_crime_mcp.server.envelope.ToolResult`.
3. The code-coverage warning fires in **both** modes. ``comparable`` corrects
   only the drift someone curated, so it is never a reason to suppress the
   warning.

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
    coverage_warning,
    empty_result_warning,
    partial_period_warning,
    provisional_warning,
    truncated_warning,
)
from chicago_crime_mcp.server.errors import InvalidArgumentError
from chicago_crime_mcp.server.models import (
    AggregateFilters,
    AggregatePayload,
    AggregateResponse,
)
from chicago_crime_mcp.server.tools._validate import (
    applied_filter_names,
    validated_geography_values,
    validated_range,
    validated_types,
)
from chicago_crime_mcp.store.duckdb import queries
from chicago_crime_mcp.store.normalize import Geography, Taxonomy

#: Alias for the store's cap, under the name the schema block publishes it as.
MAX_BUCKETS = queries.MAX_LIMIT


def aggregate_incidents(
    start: Annotated[date, Field(description="First day to include, inclusive.")],
    end: Annotated[date, Field(description="Last day to include, inclusive.")],
    grain: Annotated[
        queries.Grain, Field(description="Time bucket size.")
    ] = "month",
    types: Annotated[
        list[str] | None,
        Field(
            description="Offense categories to include, interpreted under `taxonomy`. Omit for "
            "all. Values must come from describe_schema."
        ),
    ] = None,
    taxonomy: Annotated[
        Taxonomy,
        Field(
            description="How offense codes are grouped into categories. 'source' is faithful to "
            "CPD's labels; 'comparable' re-groups codes that moved between categories mid-window "
            "and is the right choice for a multi-year trend."
        ),
    ] = "source",
    geography: Annotated[
        Geography,
        Field(
            description="Geography to break down by. With no `geography_values` this is a "
            "returned dimension (one bucket per value); with them it is also a filter."
        ),
    ] = "citywide",
    geography_values: Annotated[
        list[str | int] | None,
        Field(description="Restrict to these values of `geography`. Omit for all of them."),
    ] = None,
    breakdown_by_type: Annotated[
        bool,
        Field(
            description="Return one bucket per offense category. False collapses them into a "
            "single total per period."
        ),
    ] = True,
    limit: Annotated[
        int, Field(description=f"Maximum buckets, at most {MAX_BUCKETS}.")
    ] = queries.DEFAULT_LIMIT,
) -> AggregateResponse:
    """Count offenses over time, optionally broken down by category and geography.

    This is the tool for "how many", "which is most common", "has it gone up".
    It answers from pre-summed monthly rollups whenever the requested span is
    whole months, and falls back to a live scan when it is not; both produce
    identical numbers and the response says which ran.

    Measures are **counts, never rates**: ``incidents``, ``arrests``,
    ``domestic`` and ``geocoded`` per bucket. Derive an arrest rate by dividing
    if you need one -- summing stored rates across buckets of different sizes
    would be wrong.

    Read the warnings before drawing a trend. Three of them change the answer:

    * ``provisional`` -- the last bucket is still filling and will grow. A
      decline at the right-hand edge is usually this.
    * ``code_coverage_drift`` -- offense codes were introduced or retired inside
      the requested span, so part of any change is administrative rather than
      criminal.
    * ``boundary_instability`` -- wards and police districts have been redrawn
      inside this dataset's window. ``community_area`` is the only geography
      safe for a long series.

    Args:
        start: First day to include, inclusive.
        end: Last day to include, inclusive.
        grain: Time bucket size.
        types: Offense categories, interpreted under ``taxonomy``.
        taxonomy: How offense codes are grouped into categories.
        geography: Geography dimension to break down by.
        geography_values: Values of that dimension to restrict to.
        breakdown_by_type: Whether to return a bucket per offense category.
        limit: Maximum buckets.

    Returns:
        The buckets in the standard result envelope, with the applied taxonomy
        named and any qualifications attached.

    Raises:
        InvalidArgumentError: If the range is inverted or the limit is out of
            range.
        UnknownValueError: If a category or geography value does not exist.
    """
    context = get_context()
    vocabulary = context.vocabulary()

    validated_range(start, end)
    if not 1 <= limit <= MAX_BUCKETS:
        raise InvalidArgumentError(
            f"limit must be between 1 and {MAX_BUCKETS}.",
            field="limit",
            received=limit,
            hint="Use a coarser grain, or fewer geographies, to fit a long span into fewer "
            "buckets.",
        )
    resolved_types = validated_types(vocabulary, types, taxonomy)
    resolved_values = validated_geography_values(vocabulary, geography, geography_values)

    query = queries.AggregateQuery(
        start=start,
        end=end,
        grain=grain,
        geography=geography,
        geography_values=resolved_values,
        types=resolved_types,
        taxonomy=taxonomy,
        breakdown_by_type=breakdown_by_type,
        limit=limit,
    )

    with context.duckdb() as conn:
        try:
            result = queries.aggregate(conn, query)
        except ValueError as exc:  # pragma: no cover - inputs are validated above
            raise InvalidArgumentError.from_value_error(exc) from exc

    warnings: list[ResultWarning] = []
    if result.provisional and result.rows:
        warnings.append(
            provisional_warning(
                last_period=result.rows[-1].period,
                coverage_end=result.dataset.max_date.date(),
            )
        )
    if result.partial_first_period or result.partial_last_period:
        warnings.append(
            partial_period_warning(
                first=result.partial_first_period,
                last=result.partial_last_period,
                grain=grain,
            )
        )
    if result.truncated:
        warnings.append(truncated_warning(returned=len(result.rows), has_cursor=False))
    # Obligation 3: fires in both taxonomy modes. `coverage_warning` decides
    # whether the share clears the noise threshold and words the remedy
    # differently depending on the mode; it is never skipped because the caller
    # already asked for 'comparable'.
    coverage = coverage_warning(result.coverage, taxonomy=taxonomy)
    if coverage is not None:
        warnings.append(coverage)
    boundary = boundary_warning(geography=geography, start=start, end=end)
    if boundary is not None:
        warnings.append(boundary)
    if not result.rows:
        warnings.append(
            empty_result_warning(
                filters=applied_filter_names(
                    types=resolved_types, geography_values=resolved_values
                )
            )
        )

    return AggregateResponse(
        data=AggregatePayload.from_store(result),
        filters_applied=AggregateFilters.from_query(result.query),
        row_count=len(result.rows),
        truncated=result.truncated,
        route=RouteInfo.from_store(result.route, store="duckdb"),
        # Obligation 2: always named, in both modes including the default.
        taxonomy_mode=taxonomy,
        warnings=warnings,
        provenance=Provenance.from_dataset_meta(result.dataset),
    )


__all__ = ["MAX_BUCKETS", "aggregate_incidents"]
