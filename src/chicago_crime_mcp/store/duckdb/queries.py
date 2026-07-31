"""Read API over the DuckDB rollups -- the aggregate arm of the query router.

This is the backend for the ``aggregate_incidents`` MCP tool. It answers one
question shape ("how many incidents, bucketed by time and optionally by offense
type and geography") and it picks between two tiers to do it:

* **rollup** -- the requested span is month-aligned, so it can be answered by
  summing whole months out of the materialized table for the requested
  geography. Sub-millisecond.
* **scan** -- the span starts or ends mid-month, which no month-grain table can
  represent, so the same query runs live over ``incidents_tagged``. A few
  milliseconds, and the numbers are identical.

Both tiers are built by the *same* SQL builder, differing only in which table
they read and whether the measures are ``sum()`` of stored counts or ``count()``
of rows. That is deliberate: it makes "the two tiers agree" a structural
property rather than a coincidence two hand-written queries have to maintain,
and the test suite asserts it directly.

**No SQL escape hatch.** The caller supplies an :class:`AggregateQuery`, not a
predicate. Every identifier interpolated into the SQL below comes from a
closed mapping keyed by a ``Literal`` type -- table names, the geography column,
the type column, the grain -- and every *value* is bound as a parameter. There
is no path from caller input to SQL text.

**Facts, not prose.** The result carries the raw findings the envelope needs
(which tier answered and why, whether the edge buckets are partial, whether the
last one is still filling, which offense codes do not cover the span and what
share of rows they move) but phrases none of them. Turning those into warnings
a model reads is the server layer's job; keeping it out of here means the same
facts can be logged, cached and tested without a wording change rippling
through.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Literal

import duckdb

from chicago_crime_mcp.store.duckdb.rollups import (
    CODE_MONTH_TABLE,
    COVERAGE_TABLE,
    TAGGED_VIEW,
)

#: Time buckets. All three are whole numbers of months, which is what lets the
#: month-grain rollups answer every one of them by summing.
Grain = Literal["month", "quarter", "year"]

#: Geography dimension. ``citywide`` means no geography column at all, not a
#: filter -- it selects the dimensionless rollup table.
Geography = Literal["citywide", "beat", "district", "community_area", "ward"]

#: Which offense taxonomy to group by. ``source`` reports what the city called
#: it; ``comparable`` reports the curated stable category, which is what makes a
#: cross-year comparison valid. See the README's "On comparing crime over time".
#: Defaults to ``source`` everywhere: normalizing counts is an analytic choice
#: the caller has to make explicitly, never one this layer makes for them.
Taxonomy = Literal["source", "comparable"]

Tier = Literal["rollup", "scan"]

# One rollup table per geography. `district` is NOT derived from `beat` -- see
# the header note in rollups.sql for the 248 rows that make that unsafe.
_ROLLUP_TABLE: dict[Geography, str] = {
    "citywide": "rollup_citywide",
    "beat": "rollup_beat",
    "district": "rollup_district",
    "community_area": "rollup_community_area",
    "ward": "rollup_ward",
}

# The geography column carried by each rollup table (and by incidents_tagged,
# which uses the same names). None for citywide: no column, no GROUP BY term.
_GEO_COLUMN: dict[Geography, str | None] = {
    "citywide": None,
    "beat": "beat",
    "district": "district",
    "community_area": "community_area",
    "ward": "ward",
}

_TYPE_COLUMN: dict[Taxonomy, str] = {
    "source": "primary_type_canonical",
    "comparable": "stable_category",
}

# Zero-padded string geographies, with their widths. The source feed stores beat
# `1011` and district `010` as padded text, so a caller (or a model) passing the
# integer 10 for a district would otherwise match nothing and get a confident,
# empty, entirely wrong answer. ward and community_area are integers and are
# coerced as such.
_PADDED_GEOGRAPHIES: dict[str, int] = {"beat": 4, "district": 3}

#: How many drifting offense codes the coverage report names. The share is
#: always computed over *every* affected code; only the itemization is capped,
#: because a long-span query can implicate dozens and the model needs the
#: magnitude plus the worst offenders, not an inventory.
MAX_COVERAGE_CODES = 10

#: How long a code must be absent from one end of the span before that absence
#: counts as an introduction or a retirement rather than a gap.
#:
#: Without a buffer this report is noise. Measured on the real dataset, an
#: unbuffered rule flags 144 codes on a full-span query, the bulk of them
#: low-frequency ones that merely happened not to occur in the opening or
#: closing month: `141B` (unlawful use of a firearm) first appears in month
#: three of a 139-month span, which at its ~17 incidents a month is sampling
#: variation, not an introduction. A warning that fires on 144 codes trains its
#: reader to ignore it, which costs more than not having it.
#:
#: **Symmetric, on purpose.** An earlier draft buffered only the retirement
#: side, reasoning that a first appearance is an observed onset while a last
#: appearance is right-censored by the end of the data. That asymmetry is real
#: but it is not what makes the bounds noisy -- both ends are noisy for the same
#: reason, which is that a rare code's presence in any *particular* month is a
#: coin flip. Only a sustained absence is evidence either way. (The censoring
#: argument does still justify clamping both bounds to the months the dataset
#: covers; see :func:`_coverage`.)
#:
#: Twelve months because IUCR codes are minted and retired administratively
#: against an annually published reference table, so a code silent for a full
#: year either did not exist yet or is not coming back, while sub-year gaps are
#: ordinary for a code with a few dozen incidents a year. At twelve, the real
#: dataset's full-span report keeps `0760` BURGLARY FROM MOTOR VEHICLE -- absent
#: for the first 82 months and the one code the curation actually remaps -- and
#: drops the sampling noise.
MIN_ABSENCE_MONTHS = 12


@dataclass(frozen=True)
class AggregateQuery:
    """A single aggregate request, already shaped like something answerable.

    Attributes:
        start: First day to include.
        end: Last day to include, **inclusive** -- the boundary a person means
            when they say "through March 31st".
        grain: Time bucket size.
        geography: Geography dimension, or ``citywide`` for none.
        geography_values: Restrict to these geographies; empty means all of
            them, which also makes the geography a returned dimension rather
            than a filter. Strings or ints; normalized to the column's type.
        types: Restrict to these offense categories, interpreted under
            ``taxonomy``; empty means all. Case-insensitive.
        taxonomy: Which offense taxonomy to group and filter by.
        breakdown_by_type: Include the offense category as a dimension. False
            collapses it, giving a plain total per period.
        limit: Maximum rows returned. One extra row is fetched internally to
            set :attr:`AggregateResult.truncated` without a second count query.
    """

    start: date
    end: date
    grain: Grain = "month"
    geography: Geography = "citywide"
    geography_values: tuple[str | int, ...] = ()
    types: tuple[str, ...] = ()
    taxonomy: Taxonomy = "source"
    breakdown_by_type: bool = True
    limit: int = 500

    def __post_init__(self) -> None:
        """Reject queries that cannot be answered, with a message naming the field.

        Raises:
            ValueError: If the date range is inverted or the limit is not
                positive. The server layer catches these and re-raises them as
                the structured, self-correcting errors the model retries from.
        """
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) is before start ({self.start})")
        if self.limit < 1:
            raise ValueError(f"limit must be at least 1, got {self.limit}")


@dataclass(frozen=True)
class AggregateRow:
    """One bucket of the result.

    Measures are counts, never rates -- ``arrest_rate`` is derived at read time
    by whoever presents it. Storing or returning a rate would be wrong the
    moment two rows are summed, because averaging rates over buckets of
    different sizes is not the combined rate.

    Attributes:
        period: First day of the bucket.
        category: Offense category under the applied taxonomy, or None when
            ``breakdown_by_type`` was False.
        geography_value: The geography, or None for a citywide query.
        incidents: Rows in the bucket.
        arrests: Rows with an arrest.
        domestic: Rows flagged domestic.
        geocoded: Rows with coordinates. Carried per bucket because the geocode
            rate is *systematically* biased by offense type (measured: 91.5% for
            offenses involving children, 100% for homicide), so a caller
            comparing these counts against a radius query needs the local rate,
            not the dataset-wide one.
    """

    period: date
    category: str | None
    geography_value: str | int | None
    incidents: int
    arrests: int
    domestic: int
    geocoded: int


@dataclass(frozen=True)
class Route:
    """Which store answered, and why -- the routing decision, made legible.

    Attributes:
        tier: ``rollup`` or ``scan``.
        table: The relation actually read.
        reason: Why that tier, in one clause, for the envelope and the logs.
        elapsed_ms: Wall time of the aggregate query alone.
    """

    tier: Tier
    table: str
    reason: str
    elapsed_ms: float


@dataclass(frozen=True)
class DatasetMeta:
    """Provenance of the rollup build, read from ``rollup_meta``.

    Attributes:
        built_at: When the rollups were built (naive UTC -- the one
            non-America/Chicago timestamp in the project, because it is a system
            clock reading rather than source data).
        source_rows: Incidents rolled up.
        partitions: Year partitions covered.
        min_date: Earliest incident.
        max_date: Latest incident. Trails today by about a week *by design*:
            the source feed withholds the most recent seven days. This is the
            bound that decides whether a trailing bucket is still filling.
    """

    built_at: datetime
    source_rows: int
    partitions: int
    min_date: datetime
    max_date: datetime


@dataclass(frozen=True)
class CodeCoverage:
    """An offense code that does not cover the requested span.

    Attributes:
        iucr: The IUCR code.
        description: Its description in the reference snapshot.
        category: Its category under the applied taxonomy.
        first_month: First month the code appears anywhere in the dataset.
        last_month: Last month it appears anywhere in the dataset.
        incidents: Its rows *within the requested span*, citywide.
        enters: The code was absent for at least :data:`MIN_ABSENCE_MONTHS`
            after the span opened, so early buckets lack it and its category
            trends upward for no real-world reason.
        exits: The code went silent at least :data:`MIN_ABSENCE_MONTHS` before
            the span closed. Both flags can be true at once, for a code that
            lived and died inside the span.
    """

    iucr: str
    description: str | None
    category: str | None
    first_month: date
    last_month: date
    incidents: int
    enters: bool
    exits: bool


@dataclass(frozen=True)
class CoverageReport:
    """How much of the answer rests on offense codes that drifted mid-span.

    Fires in **both** taxonomy modes. ``comparable`` corrects only the drift
    someone curated -- one code today -- so it is never a reason to suppress
    this. See the README's "How taxonomy drift is actually handled".

    The share is citywide even when the query was filtered to a geography:
    ``rollup_code_month`` deliberately carries no geography dimension, because
    drift is a property of how the city codes offenses rather than of where
    they happen. Callers presenting the number should say so.

    Attributes:
        codes: The affected codes, largest first, capped at
            :data:`MAX_COVERAGE_CODES`.
        code_count: How many were affected in total, including any not itemized.
        affected_incidents: In-span rows from affected codes, all of them.
        total_incidents: In-span rows from every code, the denominator.
    """

    codes: tuple[CodeCoverage, ...]
    code_count: int
    affected_incidents: int
    total_incidents: int

    @property
    def affected_share(self) -> float:
        """Fraction of in-span rows from codes that do not cover the span.

        Returns:
            A value in ``[0, 1]``; 0.0 when the span holds no rows at all.
        """
        if not self.total_incidents:
            return 0.0
        return self.affected_incidents / self.total_incidents


@dataclass(frozen=True)
class AggregateResult:
    """The buckets plus everything the envelope needs to qualify them.

    Attributes:
        rows: The buckets, ordered by period then category then geography.
        query: The query as actually applied, after normalization -- so the
            envelope echoes the zero-padded district and upper-cased type the
            data was really filtered on, not what the caller typed.
        route: Which tier answered, and why.
        truncated: More buckets matched than ``limit`` returned.
        partial_first_period: The span opens mid-bucket, so the first row covers
            less time than a full one and must not be compared to the rest.
        partial_last_period: The span closes mid-bucket. Same caveat.
        provisional: The last bucket extends past the newest incident in the
            dataset, so it is still filling and will grow. Distinct from
            ``partial_last_period``, which is about the *requested* span; this
            is about the *available* data.
        coverage: Taxonomy-drift findings for the span.
        dataset: Provenance of the rollup build that answered.
    """

    rows: tuple[AggregateRow, ...]
    query: AggregateQuery
    route: Route
    truncated: bool
    partial_first_period: bool
    partial_last_period: bool
    provisional: bool
    coverage: CoverageReport
    dataset: DatasetMeta


def aggregate(conn: duckdb.DuckDBPyConnection, query: AggregateQuery) -> AggregateResult:
    """Answer an aggregate query, routing it to the cheapest tier that is exact.

    Args:
        conn: An open connection to a built rollup database. A read-only
            connection is expected in the server; nothing here writes.
        query: The request. Filter values are normalized before use, and the
            normalized form comes back on the result.

    Returns:
        The buckets plus the routing, completeness and taxonomy-drift facts the
        result envelope reports alongside them.

    Raises:
        ValueError: If a geography value cannot be coerced to the column's type
            (e.g. a non-numeric ward).
    """
    q = _normalized(query)
    dataset = dataset_meta(conn)

    tier, reason = _route(q)
    sql, params = _build_aggregate_sql(q, tier)

    started = time.perf_counter()
    fetched = conn.execute(sql, params).fetchall()
    elapsed_ms = (time.perf_counter() - started) * 1000

    # One row past the limit was requested, so its presence *is* the truncation
    # flag -- no second COUNT(*) over the same predicate.
    truncated = len(fetched) > q.limit
    rows = tuple(_to_row(r, q) for r in fetched[: q.limit])

    return AggregateResult(
        rows=rows,
        query=q,
        route=Route(
            tier=tier,
            table=_ROLLUP_TABLE[q.geography] if tier == "rollup" else TAGGED_VIEW,
            reason=reason,
            elapsed_ms=elapsed_ms,
        ),
        truncated=truncated,
        partial_first_period=q.start != _period_start(q.start, q.grain),
        partial_last_period=q.end + timedelta(days=1) != _next_period(q.end, q.grain),
        provisional=_next_period(q.end, q.grain) > dataset.max_date.date(),
        coverage=_coverage(conn, q, dataset),
        dataset=dataset,
    )


def dataset_meta(conn: duckdb.DuckDBPyConnection) -> DatasetMeta:
    """Read the single ``rollup_meta`` row describing the current build.

    Args:
        conn: An open connection to a built rollup database.

    Returns:
        The build's provenance.

    Raises:
        RuntimeError: If ``rollup_meta`` is empty, which means the rollups were
            never built against this database file.
    """
    row = conn.execute(
        "SELECT built_at, source_rows, partitions, min_date, max_date FROM rollup_meta"
    ).fetchone()
    if row is None:
        raise RuntimeError("rollup_meta is empty -- run `chicago-crime-rollup` first")
    return DatasetMeta(*row)


def categories(conn: duckdb.DuckDBPyConnection, taxonomy: Taxonomy = "source") -> tuple[str, ...]:
    """List the offense categories that actually occur, under one taxonomy.

    Backs ``describe_schema`` (so the model is told the valid values instead of
    guessing them) and the structured errors that offer a nearest match when it
    guesses anyway. Read from ``rollup_citywide``, which is a few thousand rows.

    Args:
        conn: An open connection to a built rollup database.
        taxonomy: Which taxonomy's categories to list.

    Returns:
        The distinct non-null categories, sorted.
    """
    column = _TYPE_COLUMN[taxonomy]
    rows = conn.execute(
        f"SELECT DISTINCT {column} FROM rollup_citywide "
        f"WHERE {column} IS NOT NULL ORDER BY 1"
    ).fetchall()
    return tuple(r[0] for r in rows)


def _normalized(query: AggregateQuery) -> AggregateQuery:
    """Coerce filter values to what the columns actually hold.

    Offense categories are stored upper-case; beats and districts are
    zero-padded text; wards and community areas are integers. A caller that gets
    any of these wrong would otherwise match nothing and receive a confident
    empty answer, which is the worst failure mode available to us.

    Args:
        query: The request as supplied.

    Returns:
        The same query with normalized filter values.

    Raises:
        ValueError: If a numeric geography value is not numeric.
    """
    types = tuple(t.strip().upper() for t in query.types)
    return replace(query, types=types, geography_values=_normalize_geographies(query))


def _normalize_geographies(query: AggregateQuery) -> tuple[str | int, ...]:
    """Coerce geography values to the storage type of the chosen geography.

    Args:
        query: The request as supplied.

    Returns:
        The coerced values; empty for a citywide query, where a geography filter
        is meaningless (there is no column to filter).

    Raises:
        ValueError: If a ward or community area is not an integer. The message
            names the geography so the server can turn it into a teaching error.
    """
    if query.geography == "citywide":
        return ()
    width = _PADDED_GEOGRAPHIES.get(query.geography)
    if width is not None:
        return tuple(str(v).strip().zfill(width) for v in query.geography_values)
    coerced: list[str | int] = []
    for value in query.geography_values:
        try:
            coerced.append(int(str(value).strip()))
        except ValueError as exc:
            raise ValueError(f"{query.geography} must be an integer, got {value!r}") from exc
    return tuple(coerced)


def _route(query: AggregateQuery) -> tuple[Tier, str]:
    """Choose a tier, and say why in words the envelope can repeat.

    The rollups are month buckets, so they can express a span only if it starts
    on the first of a month and ends on the last of one. Anything else -- "the
    last 30 days", "January 5th to February 20th" -- is not representable at
    that grain, and pretending otherwise would silently round the user's
    question. Those fall through to a live scan, which answers the same query a
    few milliseconds slower and to the digit.

    Args:
        query: The normalized request.

    Returns:
        The tier and its one-clause justification.
    """
    if _month_aligned(query.start, query.end):
        table = _ROLLUP_TABLE[query.geography]
        return "rollup", (
            f"span is month-aligned, so it sums whole months out of {table}"
        )
    return "scan", (
        "span starts or ends mid-month, which the month-grain rollups cannot "
        "express exactly, so it is scanned live over the source Parquet"
    )


def _build_aggregate_sql(query: AggregateQuery, tier: Tier) -> tuple[str, list]:
    """Build the aggregate SQL for either tier from one template.

    The tiers differ in exactly two places: the relation they read (a
    pre-aggregated rollup versus the row-level tagged view) and whether the
    measures sum stored counts or count rows. Dimensions, filters, grouping and
    ordering are shared, which is what makes the two tiers agree by construction
    rather than by careful maintenance of two similar queries.

    Every interpolated identifier comes from a closed mapping keyed by a
    ``Literal``; every value is bound.

    Args:
        query: The normalized request.
        tier: The tier chosen by :func:`_route`.

    Returns:
        The SQL and its bound parameters.
    """
    if tier == "rollup":
        table = _ROLLUP_TABLE[query.geography]
        time_column = "month"
        measures = ("sum(incidents)", "sum(arrests)", "sum(domestic)", "sum(geocoded)")
    else:
        table = TAGGED_VIEW
        time_column = "date"
        measures = (
            "count(*)",
            "count(*) FILTER (WHERE arrest)",
            "count(*) FILTER (WHERE domestic)",
            "count(*) FILTER (WHERE latitude IS NOT NULL)",
        )

    type_column = _TYPE_COLUMN[query.taxonomy]
    geo_column = _GEO_COLUMN[query.geography]

    # date_trunc over an already-truncated month column is a no-op for grain
    # 'month' and rolls months up cleanly for 'quarter' and 'year'.
    dimensions = [f"date_trunc('{query.grain}', {time_column}) AS period"]
    if query.breakdown_by_type:
        dimensions.append(f"{type_column} AS category")
    if geo_column is not None:
        dimensions.append(f"{geo_column} AS geography_value")

    # Half-open on the inclusive end date, so the last day is whole regardless of
    # the time of day on a timestamp column.
    conditions = [f"{time_column} >= ?", f"{time_column} < ?"]
    params: list = [query.start, query.end + timedelta(days=1)]
    if query.types:
        conditions.append(f"{type_column} IN ({_placeholders(len(query.types))})")
        params.extend(query.types)
    if geo_column is not None and query.geography_values:
        conditions.append(f"{geo_column} IN ({_placeholders(len(query.geography_values))})")
        params.extend(query.geography_values)

    # Order by ordinal, so the sort follows whichever dimensions are present.
    order = ", ".join(str(i + 1) for i in range(len(dimensions)))
    params.append(query.limit + 1)

    sql = (
        f"SELECT {', '.join(dimensions)}, {', '.join(measures)} "
        f"FROM {table} "
        f"WHERE {' AND '.join(conditions)} "
        f"GROUP BY ALL "
        f"ORDER BY {order} "
        f"LIMIT ?"
    )
    return sql, params


def _to_row(record: tuple, query: AggregateQuery) -> AggregateRow:
    """Map one fetched tuple onto :class:`AggregateRow`.

    The column layout varies with the query -- the category and geography
    columns are only selected when they are dimensions -- so the tuple is
    consumed positionally against the same two flags that built the SELECT.

    Args:
        record: One row as fetched.
        query: The normalized request that shaped the SELECT.

    Returns:
        The parsed row, with the period as a plain date.
    """
    cursor = iter(record)
    period = next(cursor)
    category = next(cursor) if query.breakdown_by_type else None
    geography = next(cursor) if _GEO_COLUMN[query.geography] is not None else None
    incidents, arrests, domestic, geocoded = cursor
    return AggregateRow(
        period=period.date() if isinstance(period, datetime) else period,
        category=category,
        geography_value=geography,
        incidents=incidents,
        arrests=arrests,
        domestic=domestic,
        geocoded=geocoded,
    )


def _coverage(
    conn: duckdb.DuckDBPyConnection,
    query: AggregateQuery,
    dataset: DatasetMeta,
) -> CoverageReport:
    """Quantify how much of the span rests on offense codes that drifted.

    Two sub-millisecond lookups against ``rollup_code_month``: the span's total
    rows, and the rows belonging to codes whose observed lifespan
    (``code_coverage``) does not cover the span. A code that simply never occurs
    in the span is not implicated -- it contributes no rows, so the join drops
    it.

    Both bounds are clamped to the months the dataset actually covers: asking
    for years before the data begins would otherwise mark every code as an
    onset, and asking past where it ends would mark every code as retired, in
    both cases because the code is absent from months that hold no rows at all.
    Clamping cannot change the totals, because there are no rows outside those
    months to include or drop. A code then has to be absent for
    :data:`MIN_ABSENCE_MONTHS` past a clamped bound to be flagged against it.

    Args:
        conn: An open connection to a built rollup database.
        query: The normalized request.
        dataset: Provenance of the build, supplying both clamps.

    Returns:
        The affected codes and the share of in-span rows they move.
    """
    type_column = _TYPE_COLUMN[query.taxonomy]
    span_first_month = max(query.start, dataset.min_date.date()).replace(day=1)
    span_last_month = min(query.end, dataset.max_date.date()).replace(day=1)

    # The whole span sits past the end of the data: nothing to weigh.
    if span_last_month < span_first_month:
        return CoverageReport(codes=(), code_count=0, affected_incidents=0, total_incidents=0)

    introduced_after = _add_months(span_first_month, MIN_ABSENCE_MONTHS)
    retired_before = _add_months(span_last_month, -MIN_ABSENCE_MONTHS)

    conditions = ["month >= ?", "month < ?"]
    params: list = [span_first_month, _add_months(span_last_month, 1)]
    if query.types:
        conditions.append(f"{type_column} IN ({_placeholders(len(query.types))})")
        params.extend(query.types)
    where = " AND ".join(conditions)

    total = conn.execute(
        f"SELECT coalesce(sum(incidents), 0) FROM {CODE_MONTH_TABLE} WHERE {where}",
        params,
    ).fetchone()[0]

    # The null-IUCR bucket is excluded from the *numerator* only: it has no
    # meaningful lifespan to compare, but its rows still belong in the total.
    affected = conn.execute(
        f"""
        WITH in_span AS (
            SELECT iucr, sum(incidents) AS incidents
            FROM {CODE_MONTH_TABLE}
            WHERE {where} AND iucr IS NOT NULL
            GROUP BY iucr
        )
        SELECT s.iucr, c.description, c.{type_column},
               c.first_month, c.last_month, s.incidents,
               c.first_month > ? AS enters, c.last_month < ? AS exits
        FROM in_span s
        JOIN {COVERAGE_TABLE} c USING (iucr)
        WHERE c.first_month > ? OR c.last_month < ?
        ORDER BY s.incidents DESC
        """,
        [*params, introduced_after, retired_before, introduced_after, retired_before],
    ).fetchall()

    codes = tuple(
        CodeCoverage(
            iucr=iucr,
            description=description,
            category=category,
            first_month=first.date() if isinstance(first, datetime) else first,
            last_month=last.date() if isinstance(last, datetime) else last,
            incidents=incidents,
            enters=enters,
            exits=exits,
        )
        for iucr, description, category, first, last, incidents, enters, exits in affected
    )
    return CoverageReport(
        codes=codes[:MAX_COVERAGE_CODES],
        code_count=len(codes),
        affected_incidents=sum(c.incidents for c in codes),
        total_incidents=int(total),
    )


def _placeholders(count: int) -> str:
    """Render ``count`` bind placeholders for an ``IN`` list.

    Args:
        count: How many values will be bound.

    Returns:
        A comma-separated run of ``?``.
    """
    return ", ".join("?" * count)


def _add_months(day: date, months: int) -> date:
    """Shift to the first of the month ``months`` away from ``day``'s month.

    Args:
        day: Any date.
        months: Months to add; may be negative.

    Returns:
        The first day of the resulting month.
    """
    index = day.month - 1 + months
    return date(day.year + index // 12, index % 12 + 1, 1)


def _period_start(day: date, grain: Grain) -> date:
    """First day of the bucket containing ``day``.

    Args:
        day: Any date.
        grain: Bucket size.

    Returns:
        The bucket's first day.
    """
    if grain == "month":
        return day.replace(day=1)
    if grain == "quarter":
        return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)
    return date(day.year, 1, 1)


def _next_period(day: date, grain: Grain) -> date:
    """First day of the bucket *after* the one containing ``day``.

    Args:
        day: Any date.
        grain: Bucket size.

    Returns:
        The next bucket's first day.
    """
    start = _period_start(day, grain)
    if grain == "month":
        return _add_months(start, 1)
    if grain == "quarter":
        return _add_months(start, 3)
    return date(start.year + 1, 1, 1)


def _month_aligned(start: date, end: date) -> bool:
    """Whether ``[start, end]`` is exactly a whole number of months.

    This is the routing predicate: only such a span can be summed out of the
    month-grain rollups without rounding the question.

    Args:
        start: First day of the span.
        end: Last day of the span, inclusive.

    Returns:
        True if the span opens on the first of a month and closes on the last.
    """
    return start.day == 1 and end + timedelta(days=1) == _add_months(end.replace(day=1), 1)
