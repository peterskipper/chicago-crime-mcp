"""The result envelope: everything a model needs to judge an answer it cannot see.

A tool that returns bare rows asks the model to trust them. This one returns
rows *plus the conditions under which they are true* -- which filters were
actually applied after normalization, whether the list is complete, which store
answered and why, which offense taxonomy the categories are expressed in, and
what about the underlying data would make a naive reading wrong.

The store layer deliberately returns facts and phrases none of them: ``provisional``
is a boolean there and a sentence here. This module is where that translation
happens, and keeping it in one place is what stops two tools from describing the
same caveat two different ways.

**Warnings are structured, not prose.** Each carries a ``code`` from a closed
vocabulary alongside its message. A model can branch on ``provisional``; it can
only pattern-match on "this period is still filling". The message is what gets
read, the code is what gets *acted on* and what telemetry counts.

**Generic over both payload and filters.** ``ToolResult[T, F]`` keeps the echoed
filters a typed model rather than an untyped bag, so the schema FastMCP derives
tells the model what an echoed filter set looks like before it ever makes a
call. That is the same argument as the tool surface itself: describe the shape,
do not make the model infer it.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

from chicago_crime_mcp.store.normalize import Geography, Taxonomy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

#: The closed warning vocabulary.
#:
#: * ``provisional`` -- the last bucket extends past the newest incident, so it
#:   is still filling. The failure this exists to prevent: an agent reporting a
#:   40% drop because the month has not finished loading.
#: * ``partial_period`` -- an edge bucket covers less time than a full one
#:   because the *requested* span opened or closed mid-bucket. Distinct from
#:   ``provisional``, which is about the available data rather than the request.
#: * ``truncated`` -- more matched than were returned.
#: * ``code_coverage_drift`` -- offense codes that do not span the whole period
#:   move the totals for reasons that are administrative, not criminal.
#: * ``boundary_instability`` -- the geography itself moved during the span.
#: * ``excludes_ungeocoded`` -- a radius query can only see incidents that have
#:   coordinates, and the ones that do not are not a random sample.
#: * ``multiple_matches`` -- a case number resolved to several offense rows.
#: * ``empty_result`` -- nothing matched, with the applied filters restated so
#:   the model can tell "no crime there" from "wrong filter value".
WarningCode = Literal[
    "provisional",
    "partial_period",
    "truncated",
    "code_coverage_drift",
    "boundary_instability",
    "excludes_ungeocoded",
    "multiple_matches",
    "empty_result",
]

#: Geographies whose boundaries have been redrawn inside the dataset's window.
#: Community areas are the only stable long series -- see the README's "On
#: comparing crime over time".
_UNSTABLE_GEOGRAPHIES: tuple[Geography, ...] = ("ward", "district", "beat")

#: Share of rows from drifting codes below which the coverage warning is not
#: worth the model's attention. A warning that fires on every call trains its
#: reader to ignore it.
COVERAGE_WARNING_THRESHOLD = 0.005


class _RouteLike(Protocol):
    """Structural type for the store layers' routing facts.

    Declared structurally rather than imported so this module stays importable
    with only Pydantic -- the store's query modules pull in ``duckdb`` and
    ``psycopg``, and the envelope has no business requiring either.
    """

    reason: str
    elapsed_ms: float


class RouteInfo(BaseModel):
    """Which store answered, and why -- the routing decision, made legible.

    The two stores report this in slightly different shapes (DuckDB names a
    tier and a relation; Postgres names neither, because there is only one) and
    the envelope reports one. The narrow fields are optional rather than
    invented for the store that has no equivalent.

    Attributes:
        store: ``postgres`` or ``duckdb``.
        tier: The DuckDB tier -- ``rollup`` for pre-summed months, ``scan`` for
            a live read that answers a span the month grain cannot express.
            None for Postgres.
        table: The relation actually read, when the store distinguishes.
        reason: Why this route, in one clause, written by the store.
        elapsed_ms: Wall time of the query itself, excluding normalization,
            mapping and envelope construction.
    """

    store: Literal["postgres", "duckdb"] = Field(description="Which store answered.")
    tier: str | None = Field(default=None, description="Rollup tier, for the DuckDB path.")
    table: str | None = Field(default=None, description="The relation read.")
    reason: str = Field(description="Why the query routed here.")
    elapsed_ms: float = Field(description="Query wall time in milliseconds.")

    @classmethod
    def from_store(cls, route: _RouteLike, *, store: Literal["postgres", "duckdb"]) -> RouteInfo:
        """Build from either store's routing dataclass.

        Args:
            route: A ``Route`` (DuckDB) or ``Timing`` (Postgres).
            store: Which store it came from. Passed explicitly because
                ``Route`` does not carry it -- within the DuckDB module there is
                only one store to be.

        Returns:
            The unified routing record.
        """
        return cls(
            store=store,
            tier=getattr(route, "tier", None),
            table=getattr(route, "table", None),
            reason=route.reason,
            elapsed_ms=round(route.elapsed_ms, 3),
        )


class ResultWarning(BaseModel):
    """One qualification on the result.

    Attributes:
        code: The machine-readable kind, from a closed vocabulary.
        message: The human- and model-readable statement.
        detail: The numbers behind the message, so a model can weigh it rather
            than only heed it -- the share of rows affected, the codes involved,
            the date past which the data is still settling.
    """

    code: WarningCode = Field(description="Machine-readable warning kind.")
    message: str = Field(description="What the caller should know.")
    detail: dict[str, Any] = Field(default_factory=dict, description="Supporting figures.")


class Provenance(BaseModel):
    """Where the answer came from and what the source says about itself.

    Carried on every response rather than documented once, because the model
    reading the response is not the entity that read the README.

    Attributes:
        source: The originating system, in words.
        dataset_id: The Socrata identifier, so a claim can be traced.
        measure: What is actually being counted. *Offenses recorded in CLEAR*,
            not *crime* -- one row is one offense, and unreported crime is not
            in here at all.
        caveats: The city's own conditions on the data.
        coverage_start: Earliest incident in the loaded window.
        coverage_end: Latest incident. Trails today by about a week by design:
            the feed withholds the most recent seven days.
        rows: Incidents in the loaded window.
        built_at: When the rollups this answer derives from were built.
    """

    source: str = Field(
        default="Chicago Police Department CLEAR, via the Chicago Data Portal",
        description="Originating system.",
    )
    dataset_id: str = Field(default="ijzp-q8t2", description="Socrata dataset identifier.")
    measure: str = Field(
        default="offenses recorded in CLEAR (one row per offense, not per incident)",
        description="What the counts count.",
    )
    caveats: tuple[str, ...] = Field(
        default=(
            "Data is preliminary; classifications can change on further investigation.",
            "Addresses are block-level to protect victim privacy; deriving a specific "
            "address is prohibited.",
            "The feed excludes the most recent 7 days, so today is never in the data.",
        ),
        description="The source's stated conditions.",
    )
    coverage_start: date | None = Field(default=None, description="Earliest incident.")
    coverage_end: date | None = Field(default=None, description="Latest incident.")
    rows: int | None = Field(default=None, description="Incidents in the loaded window.")
    built_at: datetime | None = Field(default=None, description="When the rollups were built.")

    @classmethod
    def from_dataset_meta(cls, meta: Any) -> Provenance:
        """Fill the coverage fields from the DuckDB build's ``rollup_meta`` row.

        Every tool populates these, including the Postgres-backed ones: the read
        is a single row and the alternative is a response whose provenance is
        thinner depending on which store happened to answer, which is exactly
        the kind of inconsistency the envelope exists to remove. It is also the
        first obvious thing to put behind the cache -- it changes once a night.

        Args:
            meta: A ``store.duckdb.queries.DatasetMeta``.

        Returns:
            Provenance with the coverage window filled in.
        """
        return cls(
            coverage_start=meta.min_date.date(),
            coverage_end=meta.max_date.date(),
            rows=meta.source_rows,
            built_at=meta.built_at,
        )


class ToolResult[PayloadT, FiltersT: BaseModel](BaseModel):
    """The uniform envelope every tool returns.

    Attributes:
        data: The payload, whose shape is the tool's own.
        filters_applied: The filters **as actually applied**, after
            normalization -- the zero-padded district and the upper-cased
            category the data was really filtered on, not what the caller typed.
            A model that asked for district ``10`` and reads back ``"010"`` can
            see that its input was interpreted, not ignored.
        row_count: Rows in ``data``. Not the number that matched; see
            ``truncated``.
        truncated: More matched than were returned.
        cursor: Opaque cursor for the next page, or None when there is no more.
        route: Which store answered, and why.
        taxonomy_mode: Which offense taxonomy the categories are expressed in.
            Always populated, in **both** modes including the default: silently
            returning re-grouped counts is the same class of failure as silently
            truncating. Lifted out of ``filters_applied`` because it is not a
            predicate -- it narrows nothing, it changes what the category column
            *means*.
        warnings: Qualifications on the result. Empty is the common case.
        provenance: Where the data came from and what the source says about it.
    """

    data: PayloadT = Field(description="The tool's payload.")
    filters_applied: FiltersT = Field(description="Filters as applied, after normalization.")
    row_count: int = Field(description="Rows returned in this response.")
    truncated: bool = Field(default=False, description="More rows matched than were returned.")
    cursor: str | None = Field(default=None, description="Opaque cursor for the next page.")
    route: RouteInfo = Field(description="Which store answered, and why.")
    taxonomy_mode: Taxonomy | None = Field(
        default=None, description="Offense taxonomy the categories are expressed in."
    )
    warnings: list[ResultWarning] = Field(
        default_factory=list, description="Qualifications on this result."
    )
    provenance: Provenance = Field(
        default_factory=Provenance, description="Source and coverage of the data."
    )


def provisional_warning(*, last_period: date, coverage_end: date) -> ResultWarning:
    """Say that the final bucket is still filling.

    Args:
        last_period: First day of the bucket at risk.
        coverage_end: The newest incident in the dataset.

    Returns:
        The warning.
    """
    return ResultWarning(
        code="provisional",
        message=(
            f"The final period (starting {last_period}) extends past the newest incident in "
            f"the dataset ({coverage_end}), so its counts are incomplete and will grow. Do not "
            "read a decline from it. The source feed also withholds the most recent 7 days."
        ),
        detail={"last_period": last_period.isoformat(), "coverage_end": coverage_end.isoformat()},
    )


def partial_period_warning(*, first: bool, last: bool, grain: str) -> ResultWarning:
    """Say that an edge bucket covers less time than the rest.

    Args:
        first: The span opened mid-bucket.
        last: The span closed mid-bucket.
        grain: The bucket size, for the message.

    Returns:
        The warning.
    """
    edge, buckets = {
        (True, True): (f"starts and ends mid-{grain}", "first and last buckets cover"),
        (True, False): (f"starts mid-{grain}", "first bucket covers"),
        (False, True): (f"ends mid-{grain}", "last bucket covers"),
    }[(first, last)]
    return ResultWarning(
        code="partial_period",
        message=(
            f"The requested range {edge}, so the {buckets} less time than a full {grain} and "
            "must not be compared with the others."
        ),
        detail={"partial_first": first, "partial_last": last, "grain": grain},
    )


def truncated_warning(*, returned: int, has_cursor: bool) -> ResultWarning:
    """Say that the result was capped.

    Args:
        returned: Rows in this response.
        has_cursor: Whether a next page is available.

    Returns:
        The warning.
    """
    remedy = (
        "Pass the returned cursor to continue from where this page ended."
        if has_cursor
        else "Narrow the filters or aggregate instead of listing rows."
    )
    return ResultWarning(
        code="truncated",
        message=f"More rows matched than the {returned} returned. {remedy}",
        detail={"returned": returned, "cursor_available": has_cursor},
    )


def coverage_warning(coverage: Any, *, taxonomy: Taxonomy) -> ResultWarning | None:
    """Report offense codes that do not span the requested period.

    Fires in **both** taxonomy modes. ``comparable`` corrects only the drift
    someone curated -- one code today -- so it is never a reason to suppress
    this. The share is citywide even when the query was filtered to a
    geography, because the underlying table carries no geography dimension:
    drift is a property of how the city codes offenses, not of where they
    happen. That is stated in the message rather than left as a footnote.

    Args:
        coverage: A ``store.duckdb.queries.CoverageReport``.
        taxonomy: The mode the answer was computed under, which changes what the
            reader should do about it.

    Returns:
        The warning, or None when the affected share is below
        :data:`COVERAGE_WARNING_THRESHOLD`.
    """
    share = coverage.affected_share
    if share < COVERAGE_WARNING_THRESHOLD or not coverage.code_count:
        return None
    remedy = (
        "Re-running with taxonomy='comparable' re-groups the codes that have a curated stable "
        "category."
        if taxonomy == "source"
        else "This is already the comparable taxonomy; the remaining drift has no curated mapping."
    )
    return ResultWarning(
        code="code_coverage_drift",
        message=(
            f"{coverage.code_count} offense code(s) do not cover the whole requested period -- "
            f"they were introduced or retired inside it -- accounting for {share:.1%} of "
            f"matching offenses citywide. Part of any change over this span is administrative "
            f"rather than criminal. {remedy}"
        ),
        detail={
            "code_count": coverage.code_count,
            "affected_share": round(share, 4),
            "affected_incidents": coverage.affected_incidents,
            "total_incidents": coverage.total_incidents,
            "codes": [
                {
                    "iucr": c.iucr,
                    "description": c.description,
                    "category": c.category,
                    "first_month": c.first_month.isoformat(),
                    "last_month": c.last_month.isoformat(),
                    "incidents": c.incidents,
                    "enters": c.enters,
                    "exits": c.exits,
                }
                for c in coverage.codes
            ],
            "scope": "citywide",
        },
    )


def boundary_warning(*, geography: Geography, start: date, end: date) -> ResultWarning | None:
    """Nudge a multi-year question toward the only stable geography.

    Wards are redistricted every decade and police districts and beats have been
    consolidated inside this dataset's window, so a ward's counts can move
    because its outline moved. Community areas have been fixed since the 1920s.

    Args:
        geography: The geography dimension the query used.
        start: First day of the span.
        end: Last day of the span.

    Returns:
        The warning, or None for a stable geography or a within-year span.
    """
    if geography not in _UNSTABLE_GEOGRAPHIES or start.year == end.year:
        return None
    return ResultWarning(
        code="boundary_instability",
        message=(
            f"{geography} boundaries have been redrawn within this dataset's window, so a change "
            f"across {start.year}-{end.year} may reflect a boundary moving rather than offenses "
            "moving. community_area is the only geography stable over a long series."
        ),
        detail={"geography": geography, "start_year": start.year, "end_year": end.year},
    )


def ungeocoded_warning(
    *, rates_by_type: Mapping[str, float] | None = None, overall_rate: float | None = None
) -> ResultWarning:
    """Say that a radius can only see incidents that have coordinates.

    The bare fact comes from the Postgres result; the *rates* are composed here
    from the DuckDB rollups, where ``geocoded`` is already a measure. They
    belong in the message because the gap is not uniform: measured across the
    loaded window it runs from roughly 91% for offenses involving children to
    100% for homicide. A caller comparing a radius count against an aggregate
    over the same area needs the local rate, and the axis that varies is offense
    type, not place.

    Args:
        rates_by_type: Geocode rate per offense category present in the result.
        overall_rate: Geocode rate across the whole result.

    Returns:
        The warning.
    """
    message = (
        "Radius queries can only see incidents that have coordinates, so this count excludes "
        "un-geocoded offenses and is not directly comparable with an aggregate over the same "
        "area."
    )
    detail: dict[str, Any] = {}
    if overall_rate is not None:
        message += f" About {overall_rate:.1%} of matching offenses are geocoded."
        detail["overall_geocoded_rate"] = round(overall_rate, 4)
    if rates_by_type:
        worst = min(rates_by_type.items(), key=lambda kv: kv[1])
        message += (
            f" The gap is systematic by offense type, not by place: {worst[0]} is "
            f"{worst[1]:.1%} geocoded here."
        )
        detail["geocoded_rate_by_type"] = {k: round(v, 4) for k, v in rates_by_type.items()}
    return ResultWarning(code="excludes_ungeocoded", message=message, detail=detail)


def multiple_matches_warning(*, case_number: str, count: int) -> ResultWarning:
    """Say that a case number resolved to more than one offense row.

    Args:
        case_number: The RD number looked up.
        count: How many rows it matched.

    Returns:
        The warning.
    """
    return ResultWarning(
        code="multiple_matches",
        message=(
            f"Case number {case_number} covers {count} offense rows. The CPD RD number "
            "identifies a report, not an offense, and a report that records several offenses "
            "has one row per offense. Only id is unique."
        ),
        detail={"case_number": case_number, "count": count},
    )


def empty_result_warning(*, filters: Sequence[str] = ()) -> ResultWarning:
    """Say that nothing matched, and point at the likely cause.

    Not an error: the request was valid and the answer is that there are none.
    But "no offenses of that type there" and "that filter value does not exist"
    are indistinguishable from an empty list, and only one of them is worth
    retrying, so the envelope names the filters that could be at fault.

    Args:
        filters: Names of the filters that were actually applied.

    Returns:
        The warning.
    """
    applied = ", ".join(filters) if filters else "none beyond the date range"
    return ResultWarning(
        code="empty_result",
        message=(
            "No offenses matched. The filters were valid, so this is an answer rather than a "
            f"failure -- but check the applied values before concluding there were none "
            f"(filters applied: {applied}). describe_schema lists the valid values."
        ),
        detail={"filters_applied": list(filters)},
    )
