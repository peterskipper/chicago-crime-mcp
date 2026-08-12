"""The shape of what ``describe_schema`` returns.

This is affordance #1 -- schema discovery -- and its whole job is to remove the
need to guess. Every closed set a tool will reject a bad value from is published
here, read from the data rather than from a constant: the offense categories
that actually occur under both taxonomies, the beats and wards and community
areas that actually appear, the window the data covers, and the caps a request
will be held to. A model that reads this first cannot invent a category, and one
that does not read it gets told the valid values by the error instead.

**Not a query result, so not the query envelope.** The four data tools return
:class:`~chicago_crime_mcp.server.envelope.ToolResult`, whose
``filters_applied``, ``row_count``, ``truncated`` and ``cursor`` are the point of
the thing. None of the four mean anything for a description of the dataset, and
filling them with zeroes and empty objects would be uniformity in the shape
rather than in the meaning. What *is* shared is kept shared: the same
:class:`~chicago_crime_mcp.server.envelope.Provenance` and
:class:`~chicago_crime_mcp.server.envelope.RouteInfo` models, so provenance
reads identically wherever a caller meets it.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chicago_crime_mcp.server.envelope import Provenance, RouteInfo
from chicago_crime_mcp.store.normalize import Geography, Taxonomy

#: Prose for each geography dimension: what it is, and whether its outline has
#: moved inside the loaded window. The stability flag is the same fact the
#: ``boundary_instability`` warning fires on, published up front so a model
#: planning a multi-year comparison can pick the right dimension the first time
#: instead of being corrected after the fact.
GEOGRAPHY_NOTES: dict[Geography, tuple[str, str, bool]] = {
    "citywide": ("No geographic breakdown; the whole city as one bucket.", "n/a", True),
    "beat": (
        "CPD patrol beat, the finest geography in the feed.",
        "string, zero-padded to 4 characters (e.g. '0111')",
        False,
    ),
    "district": (
        "CPD police district; each contains several beats.",
        "string, zero-padded to 3 characters (e.g. '010')",
        False,
    ),
    "ward": (
        "City council ward.",
        "integer",
        False,
    ),
    "community_area": (
        "One of Chicago's 77 official community areas.",
        "integer, 1-77",
        True,
    ),
}

#: What each taxonomy means, in the terms the choice actually turns on.
TAXONOMY_NOTES: dict[Taxonomy, str] = {
    "source": (
        "Offense categories as CPD publishes them, with label drift resolved (e.g. "
        "'CRIM SEXUAL ASSAULT' and 'CRIMINAL SEXUAL ASSAULT' unified). Faithful to the "
        "source. The default, and the right choice for a question about one period."
    ),
    "comparable": (
        "Offense categories re-grouped so a series holds its meaning across years, by "
        "re-assigning offense codes that CPD moved between categories mid-window. Use for "
        "trend questions spanning several years. It corrects only curated drift, so the "
        "coverage warning still applies."
    ),
}


class CategoryVocabulary(BaseModel):
    """The offense categories available under one taxonomy."""

    taxonomy: Taxonomy = Field(description="The value to pass as the tools' taxonomy argument.")
    column: str = Field(description="The stored column this taxonomy selects.")
    description: str = Field(description="What this taxonomy means and when to choose it.")
    categories: list[str] = Field(
        description="Every category that occurs in the loaded window. These are the only values "
        "the types argument accepts; anything else is rejected with a suggestion."
    )


class GeographyDimension(BaseModel):
    """One geography a tool can filter or group by."""

    name: Geography = Field(description="The value to pass as the tools' geography argument.")
    description: str = Field(description="What the dimension is.")
    storage: str = Field(
        description="The type and format values are coerced to. A district passed as 10 is "
        "filtered as '010'; the applied form comes back in filters_applied."
    )
    stable_over_time: bool = Field(
        description="Whether the boundaries have held still across the loaded window. False means "
        "a multi-year change may reflect a redrawn outline rather than a change in offenses; "
        "community_area is the only long-series-safe choice."
    )
    value_count: int = Field(description="How many distinct values occur.")
    values: list[str | int] = Field(description="Every value that occurs, sorted.")


class TypeGeocodeRate(BaseModel):
    """Geocoding completeness for one offense category."""

    category: str = Field(description="Offense category, 'source' taxonomy.")
    rate: float = Field(description="Share of this category's offenses that have coordinates.")
    incidents: int = Field(description="Offenses in the loaded window.")
    geocoded: int = Field(description="How many of them have coordinates.")


class GeocodeCoverageModel(BaseModel):
    """How much of the data a radius query can see.

    Published rather than left to be discovered because it decides whether two
    answers are comparable: ``nearby_incidents`` can only match offenses that
    have coordinates, while ``aggregate_incidents`` counts every row. The gap is
    small overall and **systematically uneven by offense type** -- privacy
    suppression on offenses involving children and sex offenses, none at all on
    homicide -- so the per-category rate, not the headline, is the one that
    matters when comparing a radius count against an aggregate.
    """

    overall_rate: float = Field(description="Share of all offenses that have coordinates.")
    incidents: int = Field(description="Offenses in the loaded window.")
    geocoded: int = Field(description="How many of them have coordinates.")
    note: str = Field(
        default="Radius queries see only geocoded offenses; aggregates see every row. The gap "
        "varies by offense type rather than by place, so compare using the per-category rate.",
        description="How to use these numbers.",
    )
    by_type: list[TypeGeocodeRate] = Field(
        description="Per-category rates, most frequent category first."
    )


class Limits(BaseModel):
    """The caps a request will be held to, whatever it asks for.

    Stated up front so a model plans within them: a request for 5,000 rows is
    not refused, it is silently capped and flagged ``truncated``, and knowing
    that in advance turns a broad question into a pagination plan or an
    aggregate rather than a surprise.
    """

    max_rows_per_call: int = Field(description="Hard ceiling on rows from search or nearby.")
    default_rows_per_call: int = Field(description="Rows returned when limit is not given.")
    max_aggregate_buckets: int = Field(description="Hard ceiling on aggregate buckets.")
    max_radius_m: float = Field(
        description="Widest radius nearby_incidents accepts. Past this the question is an "
        "aggregate one; use aggregate_incidents with a geography instead."
    )
    distance_rings: int = Field(
        description="Rings in the nearby distance histogram. Equal-width in metres, so unequal "
        "in area: counts rising outward is the expected shape, not a density gradient."
    )


class SchemaDescription(BaseModel):
    """Everything a caller needs to build a valid query without guessing."""

    provenance: Provenance = Field(
        description="Source, measure, caveats, and the window the data covers."
    )
    taxonomies: list[CategoryVocabulary] = Field(
        description="The two offense taxonomies and their valid categories."
    )
    geographies: list[GeographyDimension] = Field(
        description="The geography dimensions and their valid values."
    )
    grains: list[str] = Field(
        description="Time bucket sizes aggregate_incidents accepts. All are whole numbers of "
        "months, which is what lets the pre-summed month rollups answer every one of them."
    )
    geocoding: GeocodeCoverageModel = Field(description="How much of the data has coordinates.")
    limits: Limits = Field(description="Caps every request is held to.")
    time_zone: str = Field(
        default="America/Chicago",
        description="Time zone of every timestamp in the data and in these arguments.",
    )
    date_semantics: str = Field(
        default="Date ranges are inclusive of both start and end: end='2025-03-31' includes the "
        "whole of 31 March.",
        description="How the tools read start and end.",
    )
    route: RouteInfo = Field(description="Which store answered, and why.")


def _rate_models(coverage: Any) -> list[TypeGeocodeRate]:
    """Map store geocode counts onto their model, deriving the rates.

    The store returns counts and never a stored rate, because a stored rate is
    wrong the moment two rows are summed. The division belongs at the point of
    presentation, which is here.

    Args:
        coverage: A ``store.duckdb.queries.GeocodeCoverage``.

    Returns:
        One entry per category, in the order the store returned them.
    """
    return [
        TypeGeocodeRate(
            category=t.category, rate=round(t.rate, 4), incidents=t.incidents, geocoded=t.geocoded
        )
        for t in coverage.by_type
    ]


def geocode_model(coverage: Any) -> GeocodeCoverageModel:
    """Build the published geocoding block from the store's counts.

    Args:
        coverage: A ``store.duckdb.queries.GeocodeCoverage``.

    Returns:
        The model, with rates derived.
    """
    return GeocodeCoverageModel(
        overall_rate=round(coverage.rate, 4),
        incidents=coverage.incidents,
        geocoded=coverage.geocoded,
        by_type=_rate_models(coverage),
    )


__all__ = [
    "GEOGRAPHY_NOTES",
    "TAXONOMY_NOTES",
    "CategoryVocabulary",
    "GeocodeCoverageModel",
    "GeographyDimension",
    "Limits",
    "SchemaDescription",
    "TypeGeocodeRate",
    "geocode_model",
]
