"""Structured, retry-safe errors -- the self-correcting half of the tool surface.

An error is a *turn* in a conversation with a model, not a stack trace for a
human. When ``search_incidents`` is called with ``primary_type="BURGLERY"`` the
useful response is not "0 rows"; it is "there is no such category, the nearest
one is BURGLARY, here are the valid values". The model can act on that without
the user ever seeing a failure. That loop is the whole reason these exist.

**Raised, not returned.** These are exceptions, and the tools let them
propagate. A failure returned as an ``error`` field on an otherwise successful
envelope is indistinguishable, to a model reading a JSON blob, from a successful
empty result -- and the MCP protocol has a dedicated error channel precisely so
a client can tell the two apart and re-prompt. Putting a failure in the success
payload gives that up.

**The message carries everything.** MCP transmits a tool failure as *text*: the
structured fields below do not survive the wire as fields. So
:meth:`ToolError.__str__` renders the whole teaching message -- what was wrong,
what was received, what would have been right -- and that string is what the
model reads. :meth:`ToolError.details` returns the same information as a mapping
for structured logging, where the fields *do* matter: "which enum values does
the model invent" is one of the telemetry questions this project cares about,
and answering it by grepping prose would be miserable.

**Not every failure is an error.** A lookup that matches nothing, or a search
whose filters exclude everything, returns an empty *result* with the filters
echoed back. Nothing was wrong with the request; the answer is that there are
none, and saying so in the error channel would teach the model to retry a query
that is already correct.

**Why these subclass FastMCP's own ``ToolError``.** That base class is how the
framework is told a failure is deliberate and safe to show, and the difference
is not cosmetic. Measured against a server raising anything else: FastMCP
delivers a ``ToolError``'s message to the client **verbatim**, and logs a single
line; any other exception is re-wrapped behind an "Error calling tool" prefix,
is liable to be replaced wholesale when ``mask_error_details`` is on, and dumps
a full rendered traceback to stderr on every occurrence -- for what is here the
*expected* path, since a model guessing a category is how the self-correcting
loop starts. An earlier version of this translated at the server boundary
instead; it was dead code, because FastMCP catches and wraps the exception below
any middleware, so the translation never ran.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from fastmcp.exceptions import ToolError as _FastMCPToolError

#: Machine-readable error kinds. Closed on purpose: telemetry buckets on this,
#: and a free-form string would make "malformed-arg rate by kind" unanswerable.
ErrorCode = Literal[
    "unknown_value",
    "invalid_argument",
    "stale_cursor",
    "unavailable",
]

#: How many valid values to spell out in the rendered message. Offense
#: categories number a few dozen and listing all of them is the single most
#: useful thing an error can do. Community areas (77) and wards (50) are past
#: the point where an inline list helps more than it costs, so they truncate --
#: :meth:`ToolError.details` still carries the full set for the logs.
MAX_LISTED_VALUES = 40

#: Similarity floor for offering a correction. Below this, difflib starts
#: proposing matches that share a prefix and nothing else, and a confidently
#: wrong suggestion is worse than none: the model will take it.
NEAREST_CUTOFF = 0.6


class ToolError(_FastMCPToolError):
    """Base for every failure a tool reports to the model.

    Subclasses FastMCP's ``ToolError`` so the framework passes the rendered
    message through untouched rather than re-wrapping it -- see the module
    docstring for the measured difference.

    Attributes:
        code: The machine-readable kind, for telemetry.
        message: The one-line statement of what went wrong.
        field: The argument at fault, named exactly as the tool declares it, so
            the model knows which one to change.
        received: The offending value, as supplied.
        valid_values: The full set of acceptable values, when it is enumerable.
        nearest_match: The closest valid value to ``received``, when one is
            close enough to be worth proposing.
        hint: One sentence of what to do next, when the fix is not simply
            "use a valid value" -- a different tool, a narrower span, a retry
            without a cursor.
    """

    code: ErrorCode = "invalid_argument"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        received: Any = None,
        valid_values: Sequence[str] | None = None,
        nearest_match: str | None = None,
        hint: str | None = None,
    ) -> None:
        """Build an error.

        Args:
            message: What went wrong, in one line.
            field: The offending argument's name.
            received: The offending value.
            valid_values: Acceptable values, if enumerable.
            nearest_match: An explicit suggestion. When omitted and both
                ``received`` and ``valid_values`` are present, one is inferred
                with :func:`suggest`.
            hint: What to do next.
        """
        super().__init__(message)
        self.message = message
        self.field = field
        self.received = received
        self.valid_values = tuple(valid_values) if valid_values is not None else None
        if nearest_match is None and self.valid_values and isinstance(received, str):
            nearest_match = suggest(received, self.valid_values)
        self.nearest_match = nearest_match
        self.hint = hint

    def details(self) -> dict[str, Any]:
        """Return the error as structured data, for logs and telemetry.

        Keys with nothing in them are omitted rather than set to None, so a log
        line stays readable and an aggregation over ``field`` does not have to
        filter nulls.

        Returns:
            The populated fields, including the full ``valid_values`` even when
            the rendered message truncated them.
        """
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        for key, value in (
            ("field", self.field),
            ("received", self.received),
            ("valid_values", list(self.valid_values) if self.valid_values else None),
            ("nearest_match", self.nearest_match),
            ("hint", self.hint),
        ):
            if value is not None:
                payload[key] = value
        return payload

    def __str__(self) -> str:
        """Render the full teaching message -- the only part the model sees.

        Returns:
            The message followed by whichever of the offending field, the
            received value, the suggestion, the valid values and the hint are
            known. Ordered so the correction comes before the inventory: a
            model that reads one line further than it needs to should already
            have the answer.
        """
        parts = [self.message]
        if self.field is not None:
            received = f" (received {self.received!r})" if self.received is not None else ""
            parts.append(f"Offending argument: {self.field}{received}.")
        if self.nearest_match is not None:
            parts.append(f"Did you mean {self.nearest_match!r}?")
        if self.valid_values:
            shown = list(self.valid_values[:MAX_LISTED_VALUES])
            listed = ", ".join(repr(v) for v in shown)
            if len(self.valid_values) > MAX_LISTED_VALUES:
                listed += f", ... ({len(self.valid_values) - MAX_LISTED_VALUES} more)"
            parts.append(f"Valid values: {listed}.")
        if self.hint is not None:
            parts.append(self.hint)
        return " ".join(parts)


class UnknownValueError(ToolError):
    """A value that had to come from a closed set did not.

    The set is usually data-derived -- offense categories, the geographies that
    actually occur -- which is why it is passed in rather than baked into a
    type: a ``Literal`` frozen at import time goes stale against the dataset the
    moment the city mints a code.
    """

    code: ErrorCode = "unknown_value"


class InvalidArgumentError(ToolError):
    """An argument was malformed, out of range, or inconsistent with another.

    Covers what no enum can express: an inverted date range, a radius past the
    cap, a lookup given both identifiers or neither.
    """

    code: ErrorCode = "invalid_argument"

    @classmethod
    def from_value_error(
        cls, exc: ValueError, *, field: str | None = None, hint: str | None = None
    ) -> InvalidArgumentError:
        """Adapt a store-layer :class:`ValueError` into a teaching error.

        Both store modules validate in ``__post_init__`` and raise ``ValueError``
        with the offending field *named in the message* -- deliberately, so this
        conversion loses nothing. The server still validates the same rules up
        front where it can produce a better message; this is the backstop that
        keeps a rule the server forgot to mirror from surfacing as an unhandled
        exception.

        Args:
            exc: The store-layer error.
            field: The argument at fault, when the caller knows it. The store
                messages name fields in prose, not structurally, so this is not
                parsed out of the message -- guessing wrong would point the
                model at the wrong argument.
            hint: What to do next.

        Returns:
            The equivalent tool error.
        """
        return cls(str(exc), field=field, hint=hint)


class StaleCursorError(ToolError):
    """A pagination cursor does not belong to the query it was replayed against.

    Cursors carry a fingerprint of the normalized filters for exactly this
    reason. An unbound cursor replayed against a different query returns a
    plausible, non-empty, silently-skipped page -- the model has no way to
    detect that it just paged into the middle of someone else's result set, so
    the check has to be here.
    """

    code: ErrorCode = "stale_cursor"

    def __init__(self, message: str, **kwargs: Any) -> None:
        """Build the error, defaulting the hint to the actual remedy."""
        kwargs.setdefault("hint", "Re-issue the search without a cursor to start from page one.")
        super().__init__(message, **kwargs)


class DataUnavailableError(ToolError):
    """The request was well-formed but the data needed to answer it is missing.

    Not the model's fault and not fixable by rewriting the call -- an unbuilt
    rollup database, a store that will not connect. Named separately from
    ``invalid_argument`` so telemetry does not read an operational outage as a
    spike in the model getting arguments wrong.
    """

    code: ErrorCode = "unavailable"


def suggest(received: str, options: Iterable[str]) -> str | None:
    """Find the valid value a mistyped one most likely meant.

    Case-insensitive, because the normalization layer upper-cases categories
    before they ever reach a column and a lower-case guess is a spelling
    question, not a casing one.

    Args:
        received: The value as supplied.
        options: The valid values.

    Returns:
        The closest option, or None if nothing clears :data:`NEAREST_CUTOFF`.
    """
    candidates = list(options)
    folded = {c.upper(): c for c in candidates}
    matches = difflib.get_close_matches(
        received.strip().upper(), folded, n=1, cutoff=NEAREST_CUTOFF
    )
    return folded[matches[0]] if matches else None
