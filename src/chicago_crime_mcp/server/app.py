"""The FastMCP application: lifespan, tool registration, error telemetry.

Three jobs, and only three -- everything else lives in the modules this wires
together.

**Lifespan.** Both stores are connected once at startup and closed at shutdown,
by :class:`~chicago_crime_mcp.server.context.ServerContext`. Opening eagerly
means a missing rollup database or an unreachable Postgres fails where an
operator is looking, rather than as a puzzling tool error later.

**Registration.** The tools are plain functions in
:mod:`chicago_crime_mcp.server.tools`, registered here. FastMCP derives each
tool's schema from its signature and its description from its docstring, which
is why those docstrings are written for a model to act on rather than for a
developer to skim.

**Error telemetry, not error translation.** Our errors reach the model intact
because :class:`~chicago_crime_mcp.server.errors.ToolError` subclasses FastMCP's
own, not because anything here converts them: FastMCP catches a tool's exception
*below* the middleware chain, so a boundary translation would never run. (It was
written that way first, and was dead code.) The middleware records the
structured fields before the wire flattens them into text, which is where the
telemetry questions -- which values the model invents, which argument it gets
wrong -- are actually answerable.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext

from chicago_crime_mcp.server.context import ServerContext, set_context
from chicago_crime_mcp.server.errors import ToolError
from chicago_crime_mcp.server.tools.aggregate_incidents import aggregate_incidents
from chicago_crime_mcp.server.tools.describe_schema import describe_schema
from chicago_crime_mcp.server.tools.get_incident import get_incident
from chicago_crime_mcp.server.tools.nearby_incidents import nearby_incidents
from chicago_crime_mcp.server.tools.search_incidents import search_incidents

log = logging.getLogger(__name__)

#: What the server tells a client it is for. Read before any tool is called, so
#: it is the first chance to steer a question to the right tool.
INSTRUCTIONS = """
Chicago crime data: 12 years of offenses recorded in CPD's CLEAR system,
published through the Chicago Data Portal.

Call describe_schema first. It returns every valid offense category and
geography value, the window the data covers, and the caps requests are held to,
so no filter value has to be guessed.

Choosing a tool:
  - "how many", "which is most common", "has it changed"  -> aggregate_incidents
  - "which offenses", "show me the reports"               -> search_incidents
  - "what happens around this address"                    -> nearby_incidents
  - "tell me about this specific one"                     -> get_incident

Every response carries the filters as actually applied, which store answered,
which offense taxonomy was used, and any warnings. Read the warnings before
stating a trend: the newest period is always still filling, and offense codes
that were introduced or retired mid-span can move a series for administrative
reasons.
""".strip()

#: The five tools, in the order a caller meets them.
TOOLS = (
    describe_schema,
    get_incident,
    search_incidents,
    aggregate_incidents,
    nearby_incidents,
)


class ToolErrorTelemetryMiddleware(Middleware):
    """Log the structured form of every teaching error, then re-raise it.

    Translation is **not** done here, and cannot be: FastMCP catches a tool's
    exception below the middleware chain, so an error raised inside a tool
    arrives here already wrapped. Our errors subclass FastMCP's ``ToolError``
    instead, which is what gets the rendered message to the model untouched --
    see :mod:`chicago_crime_mcp.server.errors`.

    What this *can* do is record the structured fields before they collapse into
    a string on the wire. "Which enum values does the model invent" and
    "malformed-arg rate by field" are the telemetry questions this project cares
    about, and answering them by grepping prose would be miserable. This is the
    seam Phase 4's per-call logging extends.
    """

    async def on_call_tool(self, context: MiddlewareContext, call_next: Any) -> Any:
        """Run the tool, logging a structured record of any teaching error.

        Args:
            context: The call being made.
            call_next: The rest of the middleware chain.

        Returns:
            The tool's result.

        Raises:
            Exception: Whatever the tool raised, unchanged.
        """
        try:
            return await call_next(context)
        except ToolError as exc:
            log.info(
                "tool error: tool=%s %s",
                getattr(context.message, "name", "?"),
                exc.details(),
            )
            raise


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerContext]:
    """Open the store connections for the life of the server.

    Args:
        server: The FastMCP application, unused but part of the protocol.

    Yields:
        The open context, which the tools reach through
        :func:`~chicago_crime_mcp.server.context.get_context`.
    """
    context = ServerContext()
    context.open()
    set_context(context)
    try:
        yield context
    finally:
        set_context(None)
        context.close()


def create_app() -> FastMCP:
    """Build the server with its tools, instructions and middleware.

    Returns:
        The configured application, not yet running.
    """
    app: FastMCP = FastMCP(
        name="chicago-crime",
        instructions=INSTRUCTIONS,
        lifespan=lifespan,
    )
    app.add_middleware(ToolErrorTelemetryMiddleware())
    for tool in TOOLS:
        app.tool(tool)
    return app


def main(argv: list[str] | None = None) -> None:
    """Run the server.

    Transport is a runtime choice, not a build-time one: stdio for a local
    client, Streamable HTTP for the deployed service. Nothing else about the
    server differs between the two.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description="Run the Chicago crime MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        help="stdio for a local client, http for the deployed service.",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args(argv)

    # stdio speaks the protocol over stdout, so logs must go to stderr -- which
    # is where basicConfig sends them.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    app = create_app()
    if args.transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport="http", host=args.host, port=args.port)


__all__ = [
    "INSTRUCTIONS",
    "TOOLS",
    "ToolErrorTelemetryMiddleware",
    "create_app",
    "lifespan",
    "main",
]
