"""The per-call log record: one flat JSON object per tool call.

**Why the envelope is also the telemetry schema.** Every tool already returns
which store answered and why, how many rows came back, whether the page was
capped, and which qualifications applied -- because a model reading a result it
cannot verify needs all of that. Those are the same facts an operator needs, so
this module reads them back out of the serialized envelope rather than asking
the tools to report anything twice. Nothing in
:mod:`chicago_crime_mcp.server.tools` knows telemetry exists.

**Flat, not nested.** ``route`` and the error fields are spread into
``route_*`` and ``error_*`` scalars instead of nested objects. The reason is the
reader: :mod:`chicago_crime_mcp.telemetry.rollup` reads a *glob* of daily files
with one declared schema, and a nested struct whose members vary between files
-- which they do, since a day with no errors writes no error fields -- is
exactly what makes that fragile. ``args`` stays nested because it genuinely
cannot be flattened: its keys are the calling tool's signature.

**A dataclass, not a Pydantic model.** This is a log line, not part of the
schema FastMCP publishes, and keeping it stdlib-only means the rollup job -- a
batch script that should need DuckDB and nothing else -- does not import a web
framework's validation stack to read a file. Same argument the store layer makes
for its query dataclasses.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

#: What became of the call.
#:
#: * ``ok`` -- returned rows, or returned something that is not a page of rows
#:   at all (``describe_schema``, a resolved name).
#: * ``empty`` -- succeeded and matched nothing. Not an error: the request was
#:   valid and the answer is that there are none. Tracked separately because the
#:   *rate* is the signal -- a filter combination that is always empty is a
#:   vocabulary mismatch the tool descriptions should have prevented.
#: * ``error`` -- a teaching error was raised. The model is expected to recover,
#:   so this is not a server fault; the rate and the *field* are what matter.
Outcome = Literal["ok", "empty", "error"]

#: How ``resolve_neighborhood`` answered, or None for every other tool.
#:
#: A tool-specific column on an otherwise generic record, and deliberately so:
#: ``suggestion`` and ``none`` are the two values that build the alias backlog,
#: which is this project's most concrete customer-feedback loop. Reconstructing
#: them at rollup time from a JSON payload would work and would be worse -- the
#: one query anyone actually wants to run should not need to know the shape of
#: a candidate list.
#:
#: * ``exact`` / ``alias`` -- the name has a boundary of its own.
#: * ``containing`` -- real place, no boundary; answered by community area.
#: * ``suggestion`` -- nothing matched; the candidates are guesses and the model
#:   was told not to filter on them. **A soft miss.**
#: * ``none`` -- nothing matched at all and an error was raised. A hard miss.
ResolutionKind = Literal["exact", "alias", "containing", "suggestion", "none"]


@dataclass(frozen=True)
class CallRecord:
    """One tool call, as it will be written to the log.

    Attributes:
        ts: When the call *finished*, ISO 8601 in UTC. End rather than start so
            a record's timestamp and its presence in the file never disagree.
        call_id: Unique per call. The join key if a record is ever split.
        session_id: The MCP session, when the transport has one. ``None`` under
            stdio, which has no session concept beyond the process -- an honest
            gap, not a bug: per-conversation analysis needs the HTTP transport.
        client_id: The client's self-reported identity, when it sends one.
        transport: ``stdio`` or the HTTP transport name.
        tool: The tool called.
        duration_ms: Wall time for the whole call -- validation, query, mapping
            and envelope construction. Compare against ``route_elapsed_ms``,
            which is the query alone: the difference is this server's own
            overhead, and it is the only way to tell a slow database from a slow
            serializer.
        args: The arguments as received, before normalization. Before, because
            the question telemetry has to answer is what the *model* sent --
            which values it invents, which field it gets wrong. The envelope
            already echoes the normalized form to the caller.
        outcome: See :data:`Outcome`.
        route_store: ``postgres``, ``duckdb`` or ``reference``.
        route_tier: The DuckDB tier (``rollup`` / ``scan``), else None.
        route_table: The relation actually read, when the store distinguishes.
        route_reason: Why the query routed there, in the store's own words.
        route_elapsed_ms: The query's own wall time.
        row_count: Rows returned in this response, not rows matched.
        truncated: More matched than were returned.
        cursor_issued: A next page was offered.
        taxonomy_mode: Which offense taxonomy the categories are expressed in.
        warning_codes: The envelope's warning codes, in order. Codes rather than
            messages: the whole reason warnings carry a closed vocabulary is so
            that counting them does not mean grepping prose.
        result_bytes: Size of the serialized response in bytes -- the whole
            envelope, not just the payload, because the envelope is on the wire
            too and its overhead is a real cost worth watching. A proxy for
            tokens, and the measurement that makes "the tools return compact
            summaries, not row dumps" a checkable claim rather than an
            intention.
        resolution_kind: See :data:`ResolutionKind`.
        error_code: What kind of failure it was. Three families, and keeping
            them apart is the point: one of
            :data:`~chicago_crime_mcp.server.errors.ErrorCode` for a teaching
            error (the self-correcting loop working as designed);
            ``schema_validation`` when the arguments did not match the published
            JSON Schema and the tool was never entered; ``unhandled`` for an
            exception nobody planned for, which is a bug. A rate that mixes them
            is uninterpretable -- the first is healthy, the second points at the
            schema or its description, the third at us.
        error_field: The argument at fault, named as the tool declares it.
        error_received: The offending value, stringified. Grouping on this is
            how "which enum values does the model invent" gets answered.
        error_nearest_match: What the error proposed instead, if anything.
        error_message: The failure in words, truncated. For a teaching error
            this is redundant with the fields above; for ``schema_validation``
            and ``unhandled`` it is the only diagnostic there is, which is why
            it exists.
    """

    tool: str
    outcome: Outcome
    duration_ms: float
    args: dict[str, Any] = field(default_factory=dict)
    ts: str = ""
    call_id: str = ""
    session_id: str | None = None
    client_id: str | None = None
    transport: str | None = None
    route_store: str | None = None
    route_tier: str | None = None
    route_table: str | None = None
    route_reason: str | None = None
    route_elapsed_ms: float | None = None
    row_count: int | None = None
    truncated: bool | None = None
    cursor_issued: bool | None = None
    taxonomy_mode: str | None = None
    warning_codes: list[str] = field(default_factory=list)
    result_bytes: int | None = None
    resolution_kind: ResolutionKind | None = None
    error_code: str | None = None
    error_field: str | None = None
    error_received: str | None = None
    error_nearest_match: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        """Fill the identity fields that default to being generated here.

        ``ts`` and ``call_id`` are assigned rather than required so that a
        caller constructing a record never has to remember them, while a test
        can still pin both by passing them.
        """
        if not self.ts:
            object.__setattr__(self, "ts", datetime.now(UTC).isoformat())
        if not self.call_id:
            object.__setattr__(self, "call_id", uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-serializable mapping.

        Every field is present, including the ones that are None. Absent keys
        would make the rollup's declared schema a lie on any day where some
        branch never fired, which is the failure this record's flatness exists
        to avoid -- see the module docstring.

        Returns:
            The full field set, ready for :func:`json.dumps`.
        """
        return asdict(self)

    def summary(self) -> str:
        """Render the one-line human form, for the stderr mirror.

        The JSONL file is what gets aggregated; this is what someone tailing a
        local server reads. It carries the outcome, the size and the route,
        which is enough to notice something is wrong without opening the file.

        Returns:
            A single line, no trailing newline.
        """
        parts = [self.tool, self.outcome]
        if self.outcome == "error":
            parts.append(f"{self.error_code}:{self.error_field or '-'}")
        elif self.row_count is not None:
            parts.append(f"{self.row_count} rows{' (truncated)' if self.truncated else ''}")
        if self.route_store:
            route = self.route_store
            if self.route_tier:
                route += f"/{self.route_tier}"
            parts.append(route)
        parts.append(f"{self.duration_ms:.1f}ms")
        return " ".join(parts)


__all__ = ["CallRecord", "Outcome", "ResolutionKind"]
