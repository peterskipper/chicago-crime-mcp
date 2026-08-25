"""The one place every tool call is observed.

FastMCP hands middleware the tool's name, its arguments, and -- crucially -- the
serialized result. Since every tool already returns an envelope describing its
own answer, that is enough to build a complete record without any tool knowing
telemetry exists. :mod:`chicago_crime_mcp.server.tools` imports nothing from
here, and does not have to.

**Errors arrive as our own class, and that was verified rather than assumed.**
FastMCP catches a tool's exception below the middleware chain, so an earlier
version of this project's error handling -- translating at the server boundary
-- was dead code. What makes this one live is that
:class:`chicago_crime_mcp.server.errors.ToolError` subclasses FastMCP's own, and
FastMCP re-raises an exception that is already a ``ToolError`` unchanged rather
than wrapping it. Driven through a real in-memory client, the ``except`` below
receives an ``UnknownValueError`` with its structured fields intact. The
distinction matters enough that ``tests/test_app.py`` drives a real server and
not a stubbed chain.

**An unhandled exception is logged too**, under ``error_code="unhandled"`` --
outside the closed :data:`~chicago_crime_mcp.server.errors.ErrorCode` vocabulary
on purpose, because it is not a teaching error and should never be counted as
one. A teaching error is the self-correcting loop working; ``unhandled`` is a
bug, and the query that separates them is the point.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from fastmcp.server.middleware import Middleware, MiddlewareContext

from chicago_crime_mcp.server.errors import ToolError
from chicago_crime_mcp.telemetry.record import CallRecord, Outcome, ResolutionKind
from chicago_crime_mcp.telemetry.sink import TelemetrySink

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

#: The tool whose answer shape gets its own column. See
#: :data:`~chicago_crime_mcp.telemetry.record.ResolutionKind`.
RESOLVE_TOOL = "resolve_neighborhood"


class CallTelemetryMiddleware(Middleware):
    """Record every tool call: arguments in, outcome and route out."""

    def __init__(self, sink: TelemetrySink | None = None) -> None:
        """Create the middleware.

        Args:
            sink: Where records go. Defaults to one configured from the
                environment.
        """
        self.sink = sink or TelemetrySink()

    async def on_call_tool(self, context: MiddlewareContext, call_next: Any) -> Any:
        """Time the call, classify its outcome, write one record, and get out of the way.

        The record is written on every path, including the failing ones, and the
        exception is re-raised untouched -- the model's teaching message must not
        change because something is being logged.

        Args:
            context: The call being made.
            call_next: The rest of the middleware chain.

        Returns:
            The tool's result, unchanged.

        Raises:
            Exception: Whatever the tool raised, unchanged.
        """
        started = time.perf_counter()
        tool = getattr(context.message, "name", "?")
        args = dict(getattr(context.message, "arguments", None) or {})
        identity = self._identity(context)
        # Starts unbound-safe: if call_next raises a BaseException we do not
        # catch -- cancellation, above all -- the finally must not turn that into
        # an UnboundLocalError and hide it.
        record: CallRecord | None = None

        try:
            result = await call_next(context)
        except ToolError as exc:
            record = self._error_record(
                tool, args, identity, started, code=str(exc.code), details=exc.details()
            )
            raise
        except Exception as exc:
            record = self._error_record(
                tool,
                args,
                identity,
                started,
                code="unhandled",
                details={"message": f"{type(exc).__name__}: {exc}"},
            )
            raise
        else:
            record = self._result_record(tool, args, identity, started, result)
            return result
        finally:
            if record is not None:
                self.sink.write(record)

    @staticmethod
    def _identity(context: MiddlewareContext) -> dict[str, Any]:
        """Read whatever identity the transport exposes.

        MCP gives a server no conversation id and no user prompt -- only
        structured arguments -- so this is the whole of what can be correlated.
        Each field is read defensively because which ones are populated is a
        property of the transport, not of the protocol: a session id is present
        under both stdio and HTTP, while ``client_id`` and ``transport`` are
        ``None`` unless the client volunteers them.

        Args:
            context: The call being made.

        Returns:
            The ``session_id`` / ``client_id`` / ``transport`` fields.
        """
        fastmcp_context = getattr(context, "fastmcp_context", None)

        def read(name: str) -> Any:
            """Read one attribute, treating any failure as 'not available'.

            These are properties, not plain attributes, and some of them reach
            for a request context. ``getattr``'s default covers a missing name
            but not a raising property, and no identity field is worth failing a
            tool call over.
            """
            try:
                return getattr(fastmcp_context, name, None)
            except Exception:  # pragma: no cover - defensive
                return None

        return {name: read(name) for name in ("session_id", "client_id", "transport")}

    @staticmethod
    def _error_record(
        tool: str,
        args: dict[str, Any],
        identity: dict[str, Any],
        started: float,
        *,
        code: str,
        details: Mapping[str, Any],
    ) -> CallRecord:
        """Build the record for a failed call.

        Args:
            tool: The tool called.
            args: The arguments as received.
            identity: The transport identity fields.
            started: ``perf_counter`` reading from before the call.
            code: The error kind, or ``"unhandled"``.
            details: The error's structured fields.

        Returns:
            The record.
        """
        received = details.get("received")
        return CallRecord(
            tool=tool,
            outcome="error",
            duration_ms=(time.perf_counter() - started) * 1000,
            args=args,
            error_code=code,
            error_field=details.get("field"),
            error_received=None if received is None else str(received),
            error_nearest_match=details.get("nearest_match"),
            resolution_kind="none" if tool == RESOLVE_TOOL else None,
            **identity,
        )

    @classmethod
    def _result_record(
        cls,
        tool: str,
        args: dict[str, Any],
        identity: dict[str, Any],
        started: float,
        result: Any,
    ) -> CallRecord:
        """Build the record for a successful call, reading the envelope back.

        Everything here comes out of the serialized response rather than out of
        the tool, which is what keeps this generic across six tools that return
        four different payload shapes.

        Args:
            tool: The tool called.
            args: The arguments as received.
            identity: The transport identity fields.
            started: ``perf_counter`` reading from before the call.
            result: FastMCP's ``ToolResult``.

        Returns:
            The record.
        """
        duration_ms = (time.perf_counter() - started) * 1000
        body = getattr(result, "structured_content", None)
        if not isinstance(body, dict):
            # A tool returning something FastMCP cannot structure. None of ours
            # do; recorded rather than dropped so that stops being true loudly.
            return CallRecord(
                tool=tool, outcome="ok", duration_ms=duration_ms, args=args, **identity
            )

        route = body.get("route") or {}
        warning_codes = [
            code
            for warning in body.get("warnings") or []
            if isinstance(warning, dict) and (code := warning.get("code")) is not None
        ]
        outcome: Outcome = "empty" if "empty_result" in warning_codes else "ok"
        return CallRecord(
            tool=tool,
            outcome=outcome,
            duration_ms=duration_ms,
            args=args,
            route_store=route.get("store"),
            route_tier=route.get("tier"),
            route_table=route.get("table"),
            route_reason=route.get("reason"),
            route_elapsed_ms=route.get("elapsed_ms"),
            row_count=body.get("row_count"),
            truncated=body.get("truncated"),
            cursor_issued=body.get("cursor") is not None,
            taxonomy_mode=body.get("taxonomy_mode"),
            warning_codes=warning_codes,
            result_bytes=len(json.dumps(body, default=str)),
            resolution_kind=cls._resolution_kind(tool, body),
            **identity,
        )

    @staticmethod
    def _resolution_kind(tool: str, body: Mapping[str, Any]) -> ResolutionKind | None:
        """Classify how ``resolve_neighborhood`` answered.

        ``resolved=False`` is the soft miss -- the name matched nothing and the
        candidates are guesses the model was told not to filter on. Those names
        are the alias backlog, and they are invisible in any count of errors
        because the call succeeded.

        Args:
            tool: The tool called.
            body: The serialized response.

        Returns:
            The match kind of the winning candidate, or None for other tools.
        """
        if tool != RESOLVE_TOOL:
            return None
        if not body.get("resolved", False):
            return "suggestion"
        candidates = body.get("candidates") or []
        if candidates and isinstance(candidates[0], dict):
            kind = candidates[0].get("match_kind")
            if kind in ("exact", "alias", "containing", "suggestion"):
                return kind
        return None


__all__ = ["RESOLVE_TOOL", "CallTelemetryMiddleware"]
