"""Read API over Postgres -- the row and radius arm of the query router.

This backs the three MCP tools the DuckDB rollups cannot answer:
``get_incident`` (:func:`lookup`), ``search_incidents`` (:func:`search`) and
``nearby_incidents` (:func:`nearby`). Aggregates go the other way, to
``store/duckdb/queries.py``; this module never aggregates over a whole span and
that one never returns rows.

It mirrors the DuckDB module deliberately, because the server should not have to
learn two idioms:

* the connection is **injected**, so tests supply their own;
* results are **frozen dataclasses, not Pydantic**, so ``store`` stays importable
  with only the ``[store]`` extras and the server maps its own models onto these;
* results carry **facts, not prose** -- ``excludes_ungeocoded`` is a boolean here
  and a sentence in the envelope;
* there is **no SQL escape hatch**. Every interpolated identifier comes from a
  closed mapping keyed by a ``Literal`` (see
  :mod:`chicago_crime_mcp.store.normalize`), and every value is bound.

**Filter values are normalized before use** by the same code the aggregate path
uses. That is the point of the shared module: if ``search`` and ``aggregate``
disagreed about whether district ``10`` means ``"010"``, one of them would
silently return nothing for a filter the other honours.

**Inclusive end dates** are implemented as ``date >= start AND date < end + 1
day``, matching the DuckDB module exactly. If the two disagreed about what
"through March 31st" covers, the router could not claim the tiers are
interchangeable.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any

import psycopg
from psycopg import sql

from chicago_crime_mcp.store.normalize import (
    GEO_COLUMN,
    TYPE_COLUMN,
    Geography,
    Taxonomy,
    normalize_geography_values,
    normalize_types,
)

#: Every column of ``incidents`` except the generated ``geom``, in the order
#: :class:`Incident` declares them. Returned only by :func:`lookup`, where the
#: caller has already narrowed to a specific incident and provenance fields are
#: the point.
INCIDENT_COLUMNS: tuple[str, ...] = (
    "id",
    "case_number",
    "date",
    "updated_on",
    "iucr",
    "primary_type",
    "primary_type_canonical",
    "stable_category",
    "description",
    "fbi_code",
    "block",
    "location_description",
    "beat",
    "district",
    "ward",
    "community_area",
    "arrest",
    "domestic",
    "latitude",
    "longitude",
    "x_coordinate",
    "y_coordinate",
)

#: The compact row shape, in the order :class:`IncidentSummary` declares them.
#: Dropped relative to the full row: ``x_coordinate``/``y_coordinate`` (State
#: Plane duplicates of lat/long that no tool consumes), ``updated_on`` (a
#: portal-side bookkeeping timestamp), ``iucr`` and ``primary_type`` (superseded
#: by the two taxonomy columns) and ``fbi_code``. Nothing here is lost -- it is a
#: ``get_incident`` away -- and dropping it is response compaction made
#: structural rather than left to a caller's discipline.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "id",
    "case_number",
    "date",
    "primary_type_canonical",
    "stable_category",
    "description",
    "block",
    "location_description",
    "beat",
    "district",
    "ward",
    "community_area",
    "arrest",
    "domestic",
    "latitude",
    "longitude",
)

#: Hard ceiling on rows returned by a single call, whatever the caller asks for.
#: Broad queries have to degrade into pagination rather than into a huge payload.
MAX_LIMIT = 200

#: Rings in the :func:`nearby` distance histogram. Equal-width in metres, which
#: means they are **not** equal in area -- an outer ring covers far more ground
#: than an inner one, so the counts describe how far away incidents are, not how
#: dense they are. Whoever presents them has to say so.
DISTANCE_RINGS = 4

#: Widest radius accepted by :func:`nearby`. Past this the question is really an
#: aggregate one and belongs on the DuckDB path, which answers it from
#: pre-summed months instead of scanning points.
MAX_RADIUS_M = 5000

#: Truncated SHA-256 of the normalized filters, carried inside the cursor. Long
#: enough that an unrelated query will not collide into a valid-looking resume.
_FINGERPRINT_CHARS = 16


def connect(dsn: str, **kwargs: Any) -> psycopg.Connection:
    """Open a connection configured for this module's query shapes.

    ``prepare_threshold=None`` disables psycopg's automatic server-side
    prepared statements. This is not a micro-optimisation: once a statement is
    prepared, Postgres may switch to a *generic* plan built without knowing the
    parameter values, and these queries are exactly the shape that punishes --
    the same SQL is issued with a very common category one call and a very rare
    one the next, and the two want different plans. When the generic plan wins
    out, a selective filter falls off a cliff, orders of magnitude slower, after
    a handful of identical calls have gone through fine. The composite indexes
    in ``schema.sql`` immunise the category filters, but not filters on beat,
    ward or community area, so the setting stays.

    Args:
        dsn: A libpq connection string, e.g. ``StoreConfig.database_url``.
        **kwargs: Passed through to :func:`psycopg.connect`.

    Returns:
        An open connection.
    """
    return psycopg.connect(dsn, prepare_threshold=None, **kwargs)


@dataclass(frozen=True)
class Incident:
    """One incident, every stored field except the generated geometry."""

    id: int
    case_number: str | None
    date: datetime
    updated_on: datetime | None
    iucr: str
    primary_type: str
    primary_type_canonical: str
    stable_category: str
    description: str | None
    fbi_code: str | None
    block: str | None
    location_description: str | None
    beat: str | None
    district: str | None
    ward: int | None
    community_area: int | None
    arrest: bool | None
    domestic: bool | None
    latitude: float | None
    longitude: float | None
    x_coordinate: float | None
    y_coordinate: float | None


@dataclass(frozen=True)
class IncidentSummary:
    """One incident, compacted for list results. See :data:`SUMMARY_COLUMNS`."""

    id: int
    case_number: str | None
    date: datetime
    primary_type_canonical: str
    stable_category: str
    description: str | None
    block: str | None
    location_description: str | None
    beat: str | None
    district: str | None
    ward: int | None
    community_area: int | None
    arrest: bool | None
    domestic: bool | None
    latitude: float | None
    longitude: float | None


@dataclass(frozen=True)
class NearbyIncident:
    """An incident summary with its distance from the query point.

    Attributes:
        incident: The compact row.
        distance_m: Great-circle distance in metres, from the geography type.
    """

    incident: IncidentSummary
    distance_m: float


@dataclass(frozen=True)
class TypeCount:
    """Incidents of one offense category within the radius."""

    category: str
    incidents: int


@dataclass(frozen=True)
class DistanceRing:
    """One ring of the distance histogram.

    Attributes:
        lower_m: Inner edge, inclusive.
        upper_m: Outer edge.
        incidents: Rows falling in the ring. Not a density -- see
            :data:`DISTANCE_RINGS`.
    """

    lower_m: float
    upper_m: float
    incidents: int


@dataclass(frozen=True)
class Timing:
    """How the query ran, for the envelope and the logs.

    Attributes:
        store: Always ``postgres`` here; the envelope reports which store
            answered, and hardcoding it at the call site would be a second
            source of truth.
        reason: Why this store, in one clause.
        elapsed_ms: Wall time of the SQL, excluding normalization and mapping.
    """

    store: str
    reason: str
    elapsed_ms: float


@dataclass(frozen=True)
class LookupQuery:
    """Identify one incident, by surrogate key or by case number.

    Exactly one of the two must be given. They are not interchangeable: ``id``
    is the verified unique key, while ``case_number`` is the CPD RD number and
    is shared by every offense recorded under the same report.

    Attributes:
        incident_id: The numeric primary key.
        case_number: The CPD RD number. Case-insensitive.
    """

    incident_id: int | None = None
    case_number: str | None = None

    def __post_init__(self) -> None:
        """Reject an ambiguous or empty identifier.

        Raises:
            ValueError: If neither or both fields are supplied. The message
                names both fields so the server can turn it into a teaching
                error.
        """
        if (self.incident_id is None) == (self.case_number is None):
            raise ValueError(
                "provide exactly one of incident_id or case_number, "
                f"got incident_id={self.incident_id!r}, case_number={self.case_number!r}"
            )


@dataclass(frozen=True)
class LookupResult:
    """The incidents matching an identifier.

    Attributes:
        incidents: The matches. A tuple rather than a single row **on purpose**:
            a ``case_number`` lookup legitimately returns several, because a
            report covering multiple offenses records one row per offense. The
            multiplicity is a fact the envelope states rather than a surprise
            the caller discovers by indexing.
        query: The query as applied.
        timing: How it ran.
    """

    incidents: tuple[Incident, ...]
    query: LookupQuery
    timing: Timing


@dataclass(frozen=True)
class SearchQuery:
    """A filtered, ordered, paginated request for individual incidents.

    Attributes:
        start: First day to include.
        end: Last day to include, **inclusive**.
        types: Offense categories, interpreted under ``taxonomy``; empty means
            all. Case-insensitive.
        taxonomy: Which offense taxonomy ``types`` names.
        geography: Which geography ``geography_values`` names.
        geography_values: Restrict to these; empty means no geography filter.
        arrest: Restrict to rows with/without an arrest; None means either.
        domestic: Restrict to domestic-flagged rows; None means either.
        limit: Rows per page, capped at :data:`MAX_LIMIT`.
        cursor: An opaque cursor from a previous result's ``next_cursor``.
    """

    start: date
    end: date
    types: tuple[str, ...] = ()
    taxonomy: Taxonomy = "source"
    geography: Geography = "citywide"
    geography_values: tuple[str | int, ...] = ()
    arrest: bool | None = None
    domestic: bool | None = None
    limit: int = 50
    cursor: str | None = None

    def __post_init__(self) -> None:
        """Reject requests that cannot be answered.

        Raises:
            ValueError: If the date range is inverted or the limit is out of
                range. The message names the field at fault.
        """
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) is before start ({self.start})")
        if not 1 <= self.limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {self.limit}")


@dataclass(frozen=True)
class SearchResult:
    """A page of incidents plus what the envelope needs to qualify it.

    Attributes:
        rows: The page, newest first.
        query: The query as actually applied, after normalization, so the
            envelope echoes the zero-padded district that was really filtered
            on rather than what the caller typed.
        truncated: More rows matched than this page returned.
        next_cursor: Opaque cursor for the following page, or None at the end.
        timing: How it ran.
    """

    rows: tuple[IncidentSummary, ...]
    query: SearchQuery
    truncated: bool
    next_cursor: str | None
    timing: Timing


@dataclass(frozen=True)
class NearbyQuery:
    """Incidents within a radius of a point.

    There is no geography filter here: the radius *is* the geography, and
    combining the two invites a query whose two halves disagree (a beat that the
    circle only clips) for no gain.

    Attributes:
        latitude: Centre latitude, WGS84 degrees.
        longitude: Centre longitude, WGS84 degrees.
        radius_m: Search radius in metres, capped at :data:`MAX_RADIUS_M`.
        start: First day to include.
        end: Last day to include, **inclusive**.
        types: Offense categories under ``taxonomy``; empty means all.
        taxonomy: Which offense taxonomy ``types`` names.
        arrest: Restrict to rows with/without an arrest; None means either.
        domestic: Restrict to domestic-flagged rows; None means either.
        nearest: How many closest incidents to itemize.
        include_rows: Also return the full matching rows. Off by default: a
            wide downtown radius matches thousands, and the summary answers
            most questions without them.
        limit: Rows returned when ``include_rows``, capped at :data:`MAX_LIMIT`.
    """

    latitude: float
    longitude: float
    radius_m: float
    start: date
    end: date
    types: tuple[str, ...] = ()
    taxonomy: Taxonomy = "source"
    arrest: bool | None = None
    domestic: bool | None = None
    nearest: int = 10
    include_rows: bool = False
    limit: int = 50

    def __post_init__(self) -> None:
        """Reject requests that cannot be answered.

        Raises:
            ValueError: If the range is inverted, the radius is out of range,
                or the coordinates are not on Earth. The message names the
                field at fault.
        """
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) is before start ({self.start})")
        if not 0 < self.radius_m <= MAX_RADIUS_M:
            raise ValueError(
                f"radius_m must be between 1 and {MAX_RADIUS_M}, got {self.radius_m}"
            )
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"latitude must be between -90 and 90, got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"longitude must be between -180 and 180, got {self.longitude}")
        if not 1 <= self.limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, got {self.limit}")


@dataclass(frozen=True)
class NearbyResult:
    """A radius summary, with rows only if they were asked for.

    Attributes:
        total: Incidents inside the radius matching the filters.
        by_type: Counts per offense category, most frequent first.
        rings: The distance histogram, innermost first.
        nearest: The closest incidents, with distances.
        rows: The full matching page when ``include_rows``, else empty.
        query: The query as actually applied, after normalization.
        truncated: ``include_rows`` was set and more rows matched than
            returned.
        excludes_ungeocoded: Always True, and stated rather than assumed. A
            radius query can only see incidents that have coordinates, and a
            small share of the feed does not -- so this count is not comparable
            with an aggregate over the same area. **Only the bare fact lives
            here.** How much it matters varies by offense type, not by place,
            and the per-type geocode rate is already a measure on the DuckDB
            rollups; the server composes the two and words the warning. Making
            this module reach for that number would put routing below the
            router and leave it untestable without a built rollup database.
        timing: How it ran.
    """

    total: int
    by_type: tuple[TypeCount, ...]
    rings: tuple[DistanceRing, ...]
    nearest: tuple[NearbyIncident, ...]
    rows: tuple[IncidentSummary, ...]
    query: NearbyQuery
    truncated: bool
    timing: Timing
    excludes_ungeocoded: bool = field(default=True)


def lookup(conn: psycopg.Connection, query: LookupQuery) -> LookupResult:
    """Fetch one incident by id, or every incident sharing a case number.

    Args:
        conn: An open connection.
        query: The identifier.

    Returns:
        The matches, which may be empty and -- for a case number -- may be
        several.
    """
    if query.incident_id is not None:
        condition = sql.SQL("id = %s")
        params: list[Any] = [query.incident_id]
        applied = query
    else:
        # Case numbers are stored upper-case; a lower-cased one would otherwise
        # miss silently, the same failure the geography padding guards against.
        case_number = query.case_number.strip().upper()  # type: ignore[union-attr]
        condition = sql.SQL("upper(case_number) = %s")
        params = [case_number]
        applied = replace(query, case_number=case_number)

    statement = sql.SQL("SELECT {columns} FROM incidents WHERE {condition} ORDER BY id").format(
        columns=_columns(INCIDENT_COLUMNS),
        condition=condition,
    )
    fetched, elapsed_ms = _timed(conn, statement, params)

    return LookupResult(
        incidents=tuple(Incident(*record) for record in fetched),
        query=applied,
        timing=Timing(
            store="postgres",
            reason="point lookup on an indexed identifier",
            elapsed_ms=elapsed_ms,
        ),
    )


def search(conn: psycopg.Connection, query: SearchQuery) -> SearchResult:
    """Return a page of incidents matching the filters, newest first.

    Ordered by ``(date, id)`` descending and paginated by keyset rather than
    ``OFFSET``. Offset pagination re-scans everything it skips, so deep pages get
    steadily slower, and it silently shifts when the nightly upsert inserts a row
    into a page the caller has already read. A keyset cursor names the row to
    resume after, so neither happens.

    Args:
        conn: An open connection.
        query: The request. Filter values are normalized before use and the
            normalized form comes back on the result.

    Returns:
        The page, plus the cursor for the next one.

    Raises:
        ValueError: If a geography value cannot be coerced, or the cursor is
            malformed or belongs to a different set of filters.
    """
    q = _normalized(query)
    conditions, params = _filters(q.start, q.end, q.types, q.taxonomy, q.geography,
                                 q.geography_values, q.arrest, q.domestic)

    if q.cursor is not None:
        cursor_date, cursor_id = decode_cursor(q.cursor, _fingerprint(q))
        # Row comparison, not `date < a AND id < b`. The tuple form is what the
        # planner turns into a single index seek, and it is also the only one
        # that is correct: the scalar form drops every row of the tie group that
        # shares the cursor's timestamp.
        conditions.append(sql.SQL("(date, id) < (%s, %s)"))
        params.extend([cursor_date, cursor_id])

    statement = sql.SQL(
        "SELECT {columns} FROM incidents WHERE {conditions} "
        "ORDER BY date DESC, id DESC LIMIT %s"
    ).format(
        columns=_columns(SUMMARY_COLUMNS),
        conditions=sql.SQL(" AND ").join(conditions),
    )
    # One row past the limit, so its presence *is* the truncation flag -- no
    # second COUNT(*) over the same predicate.
    fetched, elapsed_ms = _timed(conn, statement, [*params, q.limit + 1])

    truncated = len(fetched) > q.limit
    rows = tuple(IncidentSummary(*record) for record in fetched[: q.limit])
    next_cursor = (
        encode_cursor(rows[-1].date, rows[-1].id, _fingerprint(q)) if truncated and rows else None
    )

    return SearchResult(
        rows=rows,
        query=q,
        truncated=truncated,
        next_cursor=next_cursor,
        timing=Timing(
            store="postgres",
            reason="row-level result, so it reads the incident table rather than a rollup",
            elapsed_ms=elapsed_ms,
        ),
    )


def nearby(conn: psycopg.Connection, query: NearbyQuery) -> NearbyResult:
    """Summarize incidents within a radius; return rows only if asked.

    Summary-first because the alternative does not scale into a model's context:
    a few hundred metres downtown over a year is thousands of incidents, and the
    question behind the request is nearly always "what happens around here", not
    "list it all". So the default answer is counts by offense type, a distance
    histogram, and the closest few -- with ``include_rows`` opting into detail.

    Args:
        conn: An open connection.
        query: The request.

    Returns:
        The summary, plus rows when ``include_rows`` was set.

    Raises:
        ValueError: If a filter value cannot be coerced.
    """
    q = _normalized_nearby(query)
    type_column = sql.Identifier(TYPE_COLUMN[q.taxonomy])
    point = _point(q.longitude, q.latitude)

    conditions, params = _filters(q.start, q.end, q.types, q.taxonomy, "citywide", (),
                                  q.arrest, q.domestic)
    # ST_DWithin on a geography measures in metres and is the GiST-indexable
    # form; ST_Distance in the predicate would not be.
    conditions.append(sql.SQL("ST_DWithin(geom, {point}, %s)").format(point=point))
    params.append(q.radius_m)
    where = sql.SQL(" AND ").join(conditions)

    # One pass yields both summaries: counts per (category, ring) fold into the
    # by-type totals and the histogram without scanning twice.
    summary_stmt = sql.SQL(
        "SELECT {type_column} AS category, "
        "least(width_bucket(ST_Distance(geom, {point}), 0, %s, %s), %s) AS ring, "
        "count(*) FROM incidents WHERE {where} GROUP BY 1, 2"
    ).format(type_column=type_column, point=point, where=where)
    summary_params = [q.radius_m, DISTANCE_RINGS, DISTANCE_RINGS, *params]
    fetched, elapsed_ms = _timed(conn, summary_stmt, summary_params)

    total = sum(count for _, _, count in fetched)
    by_type = _fold_by_type(fetched)
    rings = _fold_rings(fetched, q.radius_m)

    # KNN ordering (`<->`) walks the GiST index in distance order, so the top-N
    # stops early instead of measuring every match.
    nearest_stmt = sql.SQL(
        "SELECT {columns}, ST_Distance(geom, {point}) FROM incidents "
        "WHERE {where} ORDER BY geom <-> {point} LIMIT %s"
    ).format(columns=_columns(SUMMARY_COLUMNS), point=point, where=where)
    nearest_rows, nearest_ms = _timed(conn, nearest_stmt, [*params, q.nearest])
    nearest = tuple(
        NearbyIncident(incident=IncidentSummary(*record[:-1]), distance_m=record[-1])
        for record in nearest_rows
    )
    elapsed_ms += nearest_ms

    rows: tuple[IncidentSummary, ...] = ()
    truncated = False
    if q.include_rows:
        rows_stmt = sql.SQL(
            "SELECT {columns} FROM incidents WHERE {where} ORDER BY date DESC, id DESC LIMIT %s"
        ).format(columns=_columns(SUMMARY_COLUMNS), where=where)
        fetched_rows, rows_ms = _timed(conn, rows_stmt, [*params, q.limit + 1])
        truncated = len(fetched_rows) > q.limit
        rows = tuple(IncidentSummary(*record) for record in fetched_rows[: q.limit])
        elapsed_ms += rows_ms

    return NearbyResult(
        total=total,
        by_type=by_type,
        rings=rings,
        nearest=nearest,
        rows=rows,
        query=q,
        truncated=truncated,
        timing=Timing(
            store="postgres",
            reason="radius query, which needs the spatial index on the incident points",
            elapsed_ms=elapsed_ms,
        ),
    )


def encode_cursor(row_date: datetime, row_id: int, fingerprint: str) -> str:
    """Encode a keyset position as an opaque cursor.

    Base64 is not security -- anyone can decode it. It is there so the cursor
    does not *look* editable: a readable ``{"date": ...}`` invites a model to
    "helpfully" adjust it, and an adjusted cursor silently skips rows. It also
    keeps the format free to change without a caller having built a parser for
    it. Telemetry should log the decoded form, so opacity costs no debuggability.

    Args:
        row_date: ``date`` of the last row on the page.
        row_id: ``id`` of the last row on the page.
        fingerprint: Fingerprint of the filters this cursor belongs to.

    Returns:
        A URL-safe base64 string.
    """
    payload = json.dumps(
        {"d": row_date.isoformat(), "i": row_id, "f": fingerprint}, separators=(",", ":")
    )
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str, expected_fingerprint: str) -> tuple[datetime, int]:
    """Decode a cursor and check it belongs to the query being run.

    The fingerprint check is the load-bearing part. An unbound cursor replayed
    against different filters does not fail -- it returns a plausible,
    non-empty page from the wrong position, having silently skipped everything
    before it. That is undetectable by the caller, so it is rejected here.

    Args:
        cursor: The opaque cursor from a previous result.
        expected_fingerprint: Fingerprint of the current query's filters.

    Returns:
        The ``(date, id)`` position to resume after.

    Raises:
        ValueError: If the cursor is malformed, or was issued for a different
            set of filters. The message says which, so the model can retry
            without the cursor rather than guessing.
    """
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        row_date = datetime.fromisoformat(payload["d"])
        row_id = int(payload["i"])
        fingerprint = payload["f"]
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise ValueError(f"cursor is not a valid cursor: {cursor!r}") from exc

    if fingerprint != expected_fingerprint:
        raise ValueError(
            "cursor was issued for a different query; drop the cursor to start "
            "from the first page, or repeat the original filters"
        )
    return row_date, row_id


def _fingerprint(query: SearchQuery) -> str:
    """Fingerprint the filters a cursor is only valid for.

    Computed from the **normalized** query: an un-normalized fingerprint would
    reject a perfectly valid cursor just because the caller wrote ``"10"`` for
    the district this time and ``10`` last time. ``limit`` and ``cursor`` are
    excluded -- changing page size mid-scan is legitimate and does not move the
    keyset position.

    Args:
        query: The normalized query.

    Returns:
        A truncated hex digest.
    """
    material = json.dumps(
        [
            query.start.isoformat(),
            query.end.isoformat(),
            list(query.types),
            query.taxonomy,
            query.geography,
            [str(v) for v in query.geography_values],
            query.arrest,
            query.domestic,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode()).hexdigest()[:_FINGERPRINT_CHARS]


def _normalized(query: SearchQuery) -> SearchQuery:
    """Apply the shared filter coercion to a search request.

    Args:
        query: The request as supplied.

    Returns:
        The same query with normalized filter values.

    Raises:
        ValueError: If a geography value cannot be coerced.
    """
    return replace(
        query,
        types=normalize_types(query.types),
        geography_values=normalize_geography_values(query.geography, query.geography_values),
    )


def _normalized_nearby(query: NearbyQuery) -> NearbyQuery:
    """Apply the shared filter coercion to a radius request.

    Args:
        query: The request as supplied.

    Returns:
        The same query with normalized offense categories.
    """
    return replace(query, types=normalize_types(query.types))


def _filters(
    start: date,
    end: date,
    types: tuple[str, ...],
    taxonomy: Taxonomy,
    geography: Geography,
    geography_values: tuple[str | int, ...],
    arrest: bool | None,
    domestic: bool | None,
) -> tuple[list[sql.Composed | sql.SQL], list[Any]]:
    """Build the predicates every row query shares.

    One builder so ``search`` and ``nearby`` cannot drift on what a filter
    means, in the same spirit as the single aggregate builder on the DuckDB side.

    Args:
        start: First day to include.
        end: Last day to include, inclusive.
        types: Normalized offense categories; empty means all.
        taxonomy: Which taxonomy column ``types`` refers to.
        geography: Which geography ``geography_values`` names.
        geography_values: Normalized geography values; empty means all.
        arrest: Arrest flag to require, or None.
        domestic: Domestic flag to require, or None.

    Returns:
        The conditions and their bound parameters.
    """
    # Half-open on the inclusive end date, so the final day is whole whatever
    # time of day the timestamps carry. Must match the DuckDB module exactly.
    conditions: list[sql.Composed | sql.SQL] = [
        sql.SQL("date >= %s"),
        sql.SQL("date < %s"),
    ]
    params: list[Any] = [start, end + timedelta(days=1)]

    if types:
        conditions.append(
            sql.SQL("{column} = ANY(%s)").format(column=sql.Identifier(TYPE_COLUMN[taxonomy]))
        )
        params.append(list(types))

    geo_column = GEO_COLUMN[geography]
    if geo_column is not None and geography_values:
        conditions.append(
            sql.SQL("{column} = ANY(%s)").format(column=sql.Identifier(geo_column))
        )
        params.append(list(geography_values))

    if arrest is not None:
        conditions.append(sql.SQL("arrest = %s"))
        params.append(arrest)
    if domestic is not None:
        conditions.append(sql.SQL("domestic = %s"))
        params.append(domestic)

    return conditions, params


def _columns(names: tuple[str, ...]) -> sql.Composed:
    """Render a column list as quoted identifiers.

    Args:
        names: Column names, from a module-level constant.

    Returns:
        A comma-separated identifier list.
    """
    return sql.SQL(", ").join(sql.Identifier(name) for name in names)


def _point(longitude: float, latitude: float) -> sql.Composed:
    """Build the query point as a bound geography literal.

    Args:
        longitude: Degrees east.
        latitude: Degrees north.

    Returns:
        A SQL expression with the coordinates inlined as literals. They are
        floats validated by :meth:`NearbyQuery.__post_init__`, and the point
        appears several times in one statement, which a positional placeholder
        would force the caller to repeat in order.
    """
    return sql.SQL("ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)::geography").format(
        lon=sql.Literal(longitude), lat=sql.Literal(latitude)
    )


def _fold_by_type(fetched: list[tuple]) -> tuple[TypeCount, ...]:
    """Sum the per-(category, ring) counts into per-category totals.

    Args:
        fetched: Rows of ``(category, ring, count)``.

    Returns:
        Counts per category, most frequent first.
    """
    totals: dict[str, int] = {}
    for category, _ring, count in fetched:
        totals[category] = totals.get(category, 0) + count
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return tuple(TypeCount(category=category, incidents=count) for category, count in ordered)


def _fold_rings(fetched: list[tuple], radius_m: float) -> tuple[DistanceRing, ...]:
    """Sum the per-(category, ring) counts into the distance histogram.

    Every ring is reported, including empty ones: "nothing within 100 m" is an
    answer, and a histogram with holes in it invites the reader to misjudge the
    shape.

    Args:
        fetched: Rows of ``(category, ring, count)``.
        radius_m: The search radius, which sets the ring width.

    Returns:
        The rings, innermost first.
    """
    totals: dict[int, int] = {}
    for _category, ring, count in fetched:
        totals[ring] = totals.get(ring, 0) + count
    width = radius_m / DISTANCE_RINGS
    return tuple(
        DistanceRing(
            lower_m=width * index,
            upper_m=width * (index + 1),
            incidents=totals.get(index + 1, 0),
        )
        for index in range(DISTANCE_RINGS)
    )


def _timed(
    conn: psycopg.Connection, statement: sql.Composed, params: list[Any]
) -> tuple[list[tuple], float]:
    """Run a statement and report how long the database took.

    Args:
        conn: An open connection.
        statement: The composed statement.
        params: Bound parameters.

    Returns:
        The fetched rows and the elapsed milliseconds.
    """
    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(statement, params)
        fetched = cur.fetchall()
    return fetched, (time.perf_counter() - started) * 1000
