"""The ``nearby_incidents`` tool: what happens around a point.

Two shapes come together here. Postgres answers the radius, because that is
where the incident points and the spatial index live. DuckDB supplies the
geocode rates the answer has to be qualified with, because ``geocoded`` is
already a rollup measure there and the gap varies by **offense type, not by
place** -- measured, roughly 91% to 100% across categories but under a point
between community areas. Computing it locally in Postgres was tried and
rejected: it doubled the latency of the call to sharpen the axis that does not
vary.

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
    empty_result_warning,
    truncated_warning,
    ungeocoded_warning,
)
from chicago_crime_mcp.server.errors import InvalidArgumentError
from chicago_crime_mcp.server.models import NearbyFilters, NearbyPayload, NearbyResponse
from chicago_crime_mcp.server.tools._validate import (
    applied_filter_names,
    validated_range,
    validated_types,
)
from chicago_crime_mcp.store.duckdb import queries as duckdb_queries
from chicago_crime_mcp.store.normalize import Taxonomy
from chicago_crime_mcp.store.postgres import queries

#: How many closest offenses to itemize by default.
DEFAULT_NEAREST = 10

#: Ceiling on the itemized nearest list. Past this the answer is a listing, which
#: is what ``include_rows`` and ``search_incidents`` are for.
MAX_NEAREST = 50


def nearby_incidents(
    latitude: Annotated[float, Field(description="Centre latitude, WGS84 degrees.")],
    longitude: Annotated[float, Field(description="Centre longitude, WGS84 degrees.")],
    radius_m: Annotated[
        float, Field(description=f"Search radius in metres, at most {queries.MAX_RADIUS_M}.")
    ],
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
    arrest: Annotated[
        bool | None, Field(description="Only offenses with/without an arrest. Omit for either.")
    ] = None,
    domestic: Annotated[
        bool | None, Field(description="Only domestic-flagged offenses. Omit for either.")
    ] = None,
    nearest: Annotated[
        int, Field(description=f"How many closest offenses to itemize, at most {MAX_NEAREST}.")
    ] = DEFAULT_NEAREST,
    include_rows: Annotated[
        bool,
        Field(
            description="Also return the full matching rows. Off by default: a few hundred "
            "metres downtown over a year is thousands of offenses."
        ),
    ] = False,
    limit: Annotated[
        int, Field(description=f"Rows returned when `include_rows`, at most {queries.MAX_LIMIT}.")
    ] = queries.DEFAULT_LIMIT,
) -> NearbyResponse:
    """Summarize offenses within a radius of a point.

    Answers "what happens around here": a total, counts per offense category, a
    distance histogram, and the closest few offenses with their distances. Set
    ``include_rows`` to also get the matching rows, but prefer the summary --
    the counts answer most questions about a location without thousands of rows.

    Two things about the numbers:

    * A radius can only see offenses that **have coordinates**, and a small
      share of the feed does not. The response reports the geocode rate for the
      categories it found, because the gap is systematic by offense type rather
      than by place. Do not compare a radius count directly against an
      ``aggregate_incidents`` total for the same area.
    * The distance rings are **equal in width, not in area**. An outer ring
      covers far more ground, so counts rising outward is the expected shape and
      is not a density gradient.

    Addresses in this dataset are block-level by design, so the points are block
    centroids rather than exact locations.

    Args:
        latitude: Centre latitude, WGS84 degrees.
        longitude: Centre longitude, WGS84 degrees.
        radius_m: Search radius in metres.
        start: First day to include, inclusive.
        end: Last day to include, inclusive.
        types: Offense categories, interpreted under ``taxonomy``.
        taxonomy: Which offense taxonomy to filter and label by.
        arrest: Restrict to offenses with or without an arrest.
        domestic: Restrict to domestic-flagged offenses.
        nearest: How many closest offenses to itemize.
        include_rows: Whether to also return the full matching rows.
        limit: Rows returned when ``include_rows``.

    Returns:
        The radius summary in the standard result envelope.

    Raises:
        InvalidArgumentError: If the range is inverted, the radius or limit is
            out of range, or the point is not on Earth.
        UnknownValueError: If a category does not exist.
    """
    context = get_context()
    vocabulary = context.vocabulary()

    validated_range(start, end)
    if not 0 < radius_m <= queries.MAX_RADIUS_M:
        raise InvalidArgumentError(
            f"radius_m must be between 1 and {queries.MAX_RADIUS_M}.",
            field="radius_m",
            received=radius_m,
            hint="For a wider area, use aggregate_incidents with a geography such as "
            "community_area, which answers from pre-summed months.",
        )
    if not -90 <= latitude <= 90:
        raise InvalidArgumentError(
            "latitude must be between -90 and 90.", field="latitude", received=latitude
        )
    if not -180 <= longitude <= 180:
        raise InvalidArgumentError(
            "longitude must be between -180 and 180.", field="longitude", received=longitude
        )
    if not 1 <= nearest <= MAX_NEAREST:
        raise InvalidArgumentError(
            f"nearest must be between 1 and {MAX_NEAREST}.", field="nearest", received=nearest
        )
    if not 1 <= limit <= queries.MAX_LIMIT:
        raise InvalidArgumentError(
            f"limit must be between 1 and {queries.MAX_LIMIT}.", field="limit", received=limit
        )
    resolved_types = validated_types(vocabulary, types, taxonomy)

    query = queries.NearbyQuery(
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        start=start,
        end=end,
        types=resolved_types,
        taxonomy=taxonomy,
        arrest=arrest,
        domestic=domestic,
        nearest=nearest,
        include_rows=include_rows,
        limit=limit,
    )

    with context.postgres() as conn:
        try:
            result = queries.nearby(conn, query)
        except ValueError as exc:  # pragma: no cover - inputs are validated above
            raise InvalidArgumentError.from_value_error(exc) from exc

    warnings: list[ResultWarning] = [_geocode_warning(context, result, taxonomy, start, end)]
    if result.truncated:
        warnings.append(truncated_warning(returned=len(result.rows), has_cursor=False))
    if not result.total:
        warnings.append(
            empty_result_warning(
                filters=applied_filter_names(
                    radius_m=radius_m, types=resolved_types, arrest=arrest, domestic=domestic
                )
            )
        )

    return NearbyResponse(
        data=NearbyPayload.from_store(result),
        filters_applied=NearbyFilters.from_query(result.query),
        row_count=len(result.rows) if include_rows else len(result.nearest),
        truncated=result.truncated,
        route=RouteInfo.from_store(result.timing, store="postgres"),
        taxonomy_mode=taxonomy,
        warnings=warnings,
        provenance=Provenance.from_dataset_meta(vocabulary.dataset),
    )


def _geocode_warning(
    context: object, result: object, taxonomy: Taxonomy, start: date, end: date
) -> ResultWarning:
    """Compose the un-geocoded warning from the categories the radius found.

    The Postgres result carries only the bare fact that a radius excludes
    un-geocoded offenses. The rates come from the DuckDB rollups, scoped to the
    categories actually present and the span requested, so the warning quotes
    numbers about *this* answer rather than a dataset-wide headline. They remain
    citywide rates -- the rollup that supplies them carries no geography -- which
    is fine, because that is the axis along which the gap does not vary.

    Args:
        context: The open server context.
        result: The ``store.postgres.queries.NearbyResult``.
        taxonomy: The applied taxonomy, which decides how categories are named.
        start: First day of the span.
        end: Last day of the span.

    Returns:
        The warning, with per-category rates when the radius matched anything.
    """
    present = tuple(t.category for t in result.by_type)  # type: ignore[attr-defined]
    if not present:
        return ungeocoded_warning()
    with context.duckdb() as conn:  # type: ignore[attr-defined]
        coverage = duckdb_queries.geocode_coverage(
            conn, taxonomy=taxonomy, types=present, start=start, end=end
        )
    return ungeocoded_warning(
        rates_by_type={t.category: t.rate for t in coverage.by_type},
        overall_rate=coverage.rate,
    )


__all__ = ["DEFAULT_NEAREST", "MAX_NEAREST", "nearby_incidents"]
