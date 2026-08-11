"""Pydantic views of the store layer's frozen dataclasses.

The stores return dataclasses on purpose: ``chicago_crime_mcp.store`` has to
import with only the ``[store]`` extras, and a loader running in a nightly cron
should not drag in a web framework's validation stack. This module is the seam
where those facts become the schema FastMCP publishes to the model.

**Two shapes, and the difference is the point.** :class:`IncidentModel` carries
every column and is returned only by ``get_incident``, where the caller has
already narrowed to one report and the provenance fields are what they came
for. :class:`IncidentSummaryModel` drops six columns that no search result needs
-- the State Plane coordinate pair, a portal bookkeeping timestamp, the raw
pre-canonical type, the IUCR code, the FBI code -- and is what list-returning
tools use. Nothing is lost; it is a ``get_incident`` away. That is response
compaction as a type rather than as a caller's good intentions.

**Filters are typed, not a bag.** Each tool has a ``*Filters`` model echoing the
filters it actually applied, built by an explicit ``from_query`` rather than a
blanket field copy: ``limit`` and ``cursor`` are mechanism, not filters, and the
echo is for a model checking whether it was understood.

Every field carries a description, because those descriptions are the schema the
model reads before it ever calls anything -- the cheapest place to prevent a
malformed call is the argument list.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chicago_crime_mcp.server.envelope import ToolResult
from chicago_crime_mcp.store.normalize import Geography, Taxonomy


class _StoreModel(BaseModel):
    """Base for models mapped straight off a store dataclass.

    ``from_attributes`` lets ``model_validate`` read a frozen dataclass
    directly, so the mapping stays a field-name correspondence rather than a
    hand-written constructor call that drifts silently when a column is added.
    """

    model_config = ConfigDict(from_attributes=True)


class IncidentModel(_StoreModel):
    """One offense, in full. Returned by ``get_incident``."""

    id: int = Field(description="Unique offense identifier. The only truly unique key.")
    case_number: str | None = Field(
        description="CPD RD number. Identifies the report, not the offense, and is shared by "
        "every offense recorded under it."
    )
    date: datetime = Field(description="When the offense occurred, America/Chicago.")
    updated_on: datetime | None = Field(description="When the source record was last revised.")
    iucr: str = Field(description="Illinois Uniform Crime Reporting code.")
    primary_type: str = Field(description="Offense category exactly as the source spells it.")
    primary_type_canonical: str = Field(
        description="Offense category with the source's label drift resolved. The 'source' "
        "taxonomy."
    )
    stable_category: str = Field(
        description="Offense category re-grouped for cross-year comparability. The 'comparable' "
        "taxonomy."
    )
    description: str | None = Field(description="Secondary description of the offense.")
    fbi_code: str | None = Field(description="FBI NIBRS classification code.")
    block: str | None = Field(
        description="Block-level address. Deliberately imprecise to protect victim privacy; "
        "deriving a specific address from it is prohibited."
    )
    location_description: str | None = Field(description="Where it happened, e.g. 'STREET'.")
    beat: str | None = Field(description="Police beat, zero-padded to 4 characters.")
    district: str | None = Field(description="Police district, zero-padded to 3 characters.")
    ward: int | None = Field(description="City council ward.")
    community_area: int | None = Field(description="Community area number, 1-77.")
    arrest: bool | None = Field(description="Whether an arrest was made.")
    domestic: bool | None = Field(description="Whether it was domestic-related.")
    latitude: float | None = Field(description="Block-centroid latitude, WGS84.")
    longitude: float | None = Field(description="Block-centroid longitude, WGS84.")
    x_coordinate: float | None = Field(description="State Plane easting.")
    y_coordinate: float | None = Field(description="State Plane northing.")


class IncidentSummaryModel(_StoreModel):
    """One offense, compacted. Returned by every tool that returns a list."""

    id: int = Field(description="Unique offense identifier; pass to get_incident for full detail.")
    case_number: str | None = Field(description="CPD RD number; not unique.")
    date: datetime = Field(description="When the offense occurred, America/Chicago.")
    primary_type_canonical: str = Field(description="Offense category, 'source' taxonomy.")
    stable_category: str = Field(description="Offense category, 'comparable' taxonomy.")
    description: str | None = Field(description="Secondary description of the offense.")
    block: str | None = Field(description="Block-level address.")
    location_description: str | None = Field(description="Where it happened.")
    beat: str | None = Field(description="Police beat.")
    district: str | None = Field(description="Police district.")
    ward: int | None = Field(description="City council ward.")
    community_area: int | None = Field(description="Community area number.")
    arrest: bool | None = Field(description="Whether an arrest was made.")
    domestic: bool | None = Field(description="Whether it was domestic-related.")
    latitude: float | None = Field(description="Block-centroid latitude.")
    longitude: float | None = Field(description="Block-centroid longitude.")


def _summaries(rows: Any) -> list[IncidentSummaryModel]:
    """Map store summary rows onto their model, preserving order.

    Shared by the two payloads that carry compacted rows, so ``search`` and
    ``nearby`` cannot come to disagree about the shape of a row.

    Args:
        rows: An iterable of ``store.postgres.queries.IncidentSummary``.

    Returns:
        The mapped rows, in order.
    """
    return [IncidentSummaryModel.model_validate(r) for r in rows]


class AggregateBucketModel(_StoreModel):
    """One bucket of an aggregate result.

    Measures are counts, never rates. A rate stored per bucket is wrong the
    moment two buckets are summed, because averaging rates over unequal buckets
    is not the combined rate -- so ``arrest_rate`` is derived by whoever presents
    it, from numbers that survive addition.
    """

    period: date = Field(description="First day of the bucket.")
    category: str | None = Field(
        description="Offense category under the applied taxonomy; null when the breakdown was "
        "turned off."
    )
    geography_value: str | int | None = Field(
        description="The geography this bucket covers; null for a citywide query."
    )
    incidents: int = Field(description="Offenses in the bucket.")
    arrests: int = Field(description="Offenses in the bucket that resulted in an arrest.")
    domestic: int = Field(description="Offenses in the bucket flagged domestic.")
    geocoded: int = Field(
        description="Offenses in the bucket that have coordinates. Carried per bucket because "
        "the geocode rate varies systematically by offense type, so this is the right "
        "denominator when comparing against a radius query."
    )


class TypeCountModel(_StoreModel):
    """Offenses of one category within a radius."""

    category: str = Field(description="Offense category under the applied taxonomy.")
    incidents: int = Field(description="Offenses of that category inside the radius.")


class DistanceRingModel(_StoreModel):
    """One ring of a distance histogram.

    Rings are equal-width in metres, which makes them **unequal in area**: an
    outer ring covers far more ground than an inner one, so rising counts
    outward are the expected shape and say nothing about density.
    """

    lower_m: float = Field(description="Inner edge in metres, inclusive.")
    upper_m: float = Field(description="Outer edge in metres.")
    incidents: int = Field(description="Offenses in the ring. A count, not a density.")


class NearbyIncidentModel(_StoreModel):
    """An offense with its distance from the query point."""

    incident: IncidentSummaryModel = Field(description="The compacted offense row.")
    distance_m: float = Field(description="Great-circle distance from the query point, metres.")


class LookupPayload(BaseModel):
    """The offenses an identifier resolved to.

    A list rather than a single row **on purpose**: a case number is the CPD RD
    number, which identifies a report, and a report covering several offenses
    records one row per offense. The multiplicity is a fact the envelope states
    rather than a surprise the caller discovers by indexing.
    """

    incidents: list[IncidentModel] = Field(description="The matching offenses, in full.")

    @classmethod
    def from_store(cls, result: Any) -> LookupPayload:
        """Build from a ``store.postgres.queries.LookupResult``."""
        return cls(incidents=[IncidentModel.model_validate(r) for r in result.incidents])


class SearchPayload(BaseModel):
    """A page of offenses matching the filters, newest first."""

    incidents: list[IncidentSummaryModel] = Field(
        description="The page, newest first. Pass an id to get_incident for the full row."
    )

    @classmethod
    def from_store(cls, result: Any) -> SearchPayload:
        """Build from a ``store.postgres.queries.SearchResult``."""
        return cls(incidents=_summaries(result.rows))


class AggregatePayload(BaseModel):
    """Counts bucketed by time, and optionally by offense category and geography."""

    buckets: list[AggregateBucketModel] = Field(
        description="The buckets, ordered by period, then category, then geography."
    )

    @classmethod
    def from_store(cls, result: Any) -> AggregatePayload:
        """Build from a ``store.duckdb.queries.AggregateResult``."""
        return cls(buckets=[AggregateBucketModel.model_validate(r) for r in result.rows])


class NearbyPayload(BaseModel):
    """The radius answer: a summary, and rows only if they were asked for.

    Summary-first is not a default to be overridden lightly -- a 500m radius
    downtown matches thousands of offenses, and the counts-by-type plus the
    distance histogram answer most questions about a location without any of
    them.
    """

    total: int = Field(description="Offenses inside the radius matching the filters.")
    by_type: list[TypeCountModel] = Field(description="Counts per category, most frequent first.")
    rings: list[DistanceRingModel] = Field(description="Distance histogram, innermost first.")
    nearest: list[NearbyIncidentModel] = Field(description="The closest offenses, with distances.")
    incidents: list[IncidentSummaryModel] = Field(
        default_factory=list,
        description="Full rows, only when include_rows was set. Empty otherwise.",
    )

    @classmethod
    def from_store(cls, result: Any) -> NearbyPayload:
        """Build from a ``store.postgres.queries.NearbyResult``."""
        return cls(
            total=result.total,
            by_type=[TypeCountModel.model_validate(t) for t in result.by_type],
            rings=[DistanceRingModel.model_validate(r) for r in result.rings],
            nearest=[NearbyIncidentModel.model_validate(n) for n in result.nearest],
            incidents=_summaries(result.rows),
        )


class LookupFilters(BaseModel):
    """The identifier a lookup resolved on."""

    incident_id: int | None = Field(default=None, description="The id looked up.")
    case_number: str | None = Field(
        default=None, description="The case number looked up, upper-cased as stored."
    )

    @classmethod
    def from_query(cls, query: Any) -> LookupFilters:
        """Build from a ``store.postgres.queries.LookupQuery``."""
        return cls(incident_id=query.incident_id, case_number=query.case_number)


class SearchFilters(BaseModel):
    """Filters a search actually applied, after normalization.

    ``limit`` and ``cursor`` are absent on purpose: they shape the page, not the
    matching set, and echoing them here would invite a model to treat the page
    size as part of what it asked about.
    """

    start: date = Field(description="First day included.")
    end: date = Field(description="Last day included, inclusive.")
    types: list[str] = Field(description="Offense categories, upper-cased. Empty means all.")
    geography: Geography = Field(description="Geography dimension filtered on.")
    geography_values: list[str | int] = Field(
        description="Values filtered on, coerced to the column's storage type -- districts and "
        "beats zero-padded, wards and community areas as integers."
    )
    arrest: bool | None = Field(description="Arrest filter; null means either.")
    domestic: bool | None = Field(description="Domestic filter; null means either.")

    @classmethod
    def from_query(cls, query: Any) -> SearchFilters:
        """Build from a normalized ``store.postgres.queries.SearchQuery``."""
        return cls(
            start=query.start,
            end=query.end,
            types=list(query.types),
            geography=query.geography,
            geography_values=list(query.geography_values),
            arrest=query.arrest,
            domestic=query.domestic,
        )


class AggregateFilters(BaseModel):
    """Filters and dimensions an aggregate actually applied, after normalization."""

    start: date = Field(description="First day included.")
    end: date = Field(description="Last day included, inclusive.")
    grain: str = Field(description="Time bucket size.")
    geography: Geography = Field(
        description="Geography dimension. With no values supplied this is a returned dimension "
        "rather than a filter."
    )
    geography_values: list[str | int] = Field(
        description="Values filtered on, coerced to the column's storage type. Empty means all."
    )
    types: list[str] = Field(description="Offense categories, upper-cased. Empty means all.")
    breakdown_by_type: bool = Field(description="Whether category is a returned dimension.")

    @classmethod
    def from_query(cls, query: Any) -> AggregateFilters:
        """Build from a normalized ``store.duckdb.queries.AggregateQuery``."""
        return cls(
            start=query.start,
            end=query.end,
            grain=query.grain,
            geography=query.geography,
            geography_values=list(query.geography_values),
            types=list(query.types),
            breakdown_by_type=query.breakdown_by_type,
        )


class NearbyFilters(BaseModel):
    """Filters a radius query actually applied.

    There is no geography here because there is none in the query: the radius
    *is* the geography, and combining the two invites an answer whose two halves
    disagree over a beat the circle only clips.
    """

    latitude: float = Field(description="Centre latitude, WGS84.")
    longitude: float = Field(description="Centre longitude, WGS84.")
    radius_m: float = Field(description="Radius in metres.")
    start: date = Field(description="First day included.")
    end: date = Field(description="Last day included, inclusive.")
    types: list[str] = Field(description="Offense categories, upper-cased. Empty means all.")
    arrest: bool | None = Field(description="Arrest filter; null means either.")
    domestic: bool | None = Field(description="Domestic filter; null means either.")

    @classmethod
    def from_query(cls, query: Any) -> NearbyFilters:
        """Build from a normalized ``store.postgres.queries.NearbyQuery``."""
        return cls(
            latitude=query.latitude,
            longitude=query.longitude,
            radius_m=query.radius_m,
            start=query.start,
            end=query.end,
            types=list(query.types),
            arrest=query.arrest,
            domestic=query.domestic,
        )


#: The four tool response types. Every one of them wraps its payload in a named
#: object, including the three that carry a single list. The extra level buys
#: uniformity a model can rely on -- ``data`` is always an object with named
#: fields, never sometimes an array -- and it leaves room to add a second part
#: to a payload later without changing the shape of ``data`` under a caller.
LookupResponse = ToolResult[LookupPayload, LookupFilters]
SearchResponse = ToolResult[SearchPayload, SearchFilters]
AggregateResponse = ToolResult[AggregatePayload, AggregateFilters]
NearbyResponse = ToolResult[NearbyPayload, NearbyFilters]

__all__ = [
    "AggregateBucketModel",
    "AggregateFilters",
    "AggregatePayload",
    "AggregateResponse",
    "DistanceRingModel",
    "IncidentModel",
    "IncidentSummaryModel",
    "LookupFilters",
    "LookupPayload",
    "LookupResponse",
    "NearbyFilters",
    "NearbyIncidentModel",
    "NearbyPayload",
    "NearbyResponse",
    "SearchFilters",
    "SearchPayload",
    "SearchResponse",
    "Taxonomy",
    "TypeCountModel",
]
