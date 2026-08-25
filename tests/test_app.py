"""Tests for the FastMCP wiring: registration, schemas, error delivery.

These assert the contract the *client* sees, which is not quite the contract the
Python functions have. Two things only exist at this layer: the JSON schema
FastMCP derives from each signature -- the model's first and cheapest defence
against a malformed call -- and whether a structured error reaches the model
intact rather than re-wrapped or masked.

No stores are needed: nothing here calls a tool for real.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import asyncio

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError as FastMCPToolError
from fastmcp.server.middleware import Middleware
from pydantic import BaseModel

from chicago_crime_mcp.server import tools
from chicago_crime_mcp.server.app import TOOLS, create_app
from chicago_crime_mcp.server.envelope import ResultWarning, RouteInfo, ToolResult
from chicago_crime_mcp.server.errors import UnknownValueError
from chicago_crime_mcp.telemetry.middleware import CallTelemetryMiddleware
from tests.helpers import RecordingSink


class _Payload(BaseModel):
    """A stand-in payload for the envelope tests."""

    note: str = "x"


class _Filters(BaseModel):
    """A stand-in filter echo for the envelope tests."""

    where: str = "x"


#: The real envelope, parametrized. Using ``ToolResult`` rather than a lookalike
#: is the point of these tests: the middleware's claim is that it can read *this*
#: shape, and a hand-rolled model with the same field names would still pass if
#: the envelope changed underneath it.
Envelope = ToolResult[_Payload, _Filters]


def _envelope(**kwargs) -> Envelope:
    """Build an envelope, filling the fields these tests do not care about."""
    kwargs.setdefault("data", _Payload())
    kwargs.setdefault("filters_applied", _Filters())
    return Envelope(**kwargs)


@pytest.fixture(scope="module")
def listed():
    """The tools as a client sees them, keyed by name."""
    app = create_app()
    return {t.name: t for t in asyncio.run(app.list_tools())}


def test_every_tool_is_registered(listed):
    assert set(listed) == {
        "describe_schema",
        "get_incident",
        "search_incidents",
        "aggregate_incidents",
        "nearby_incidents",
        "resolve_neighborhood",
    }
    assert len(TOOLS) == len(listed)


def test_tools_match_the_package_exports():
    """app.TOOLS and the package's __all__ cannot drift apart silently."""
    assert {t.__name__ for t in TOOLS} == set(tools.__all__)


def test_every_tool_has_a_description(listed):
    """The docstring is the model's instructions; an empty one is a broken tool."""
    for name, tool in listed.items():
        assert tool.description and len(tool.description) > 100, name


def test_every_parameter_is_documented(listed):
    """A described argument is the cheapest place to prevent a malformed call."""
    for name, tool in listed.items():
        for parameter, spec in (tool.parameters.get("properties") or {}).items():
            assert spec.get("description"), f"{name}.{parameter}"


@pytest.mark.parametrize(
    ("tool", "parameter", "expected"),
    [
        ("search_incidents", "taxonomy", {"source", "comparable"}),
        ("aggregate_incidents", "taxonomy", {"source", "comparable"}),
        ("nearby_incidents", "taxonomy", {"source", "comparable"}),
        ("aggregate_incidents", "grain", {"month", "quarter", "year"}),
        (
            "aggregate_incidents",
            "geography",
            {"citywide", "beat", "district", "community_area", "neighborhood", "ward"},
        ),
    ],
)
def test_static_closed_sets_are_enums_in_the_schema(listed, tool, parameter, expected):
    """Static sets are the schema's job, so the framework rejects before we do.

    Only the *data-derived* sets -- categories, geography values -- are checked
    in Python, because they cannot be frozen at import time without going stale
    against the data.
    """
    spec = listed[tool].parameters["properties"][parameter]
    assert set(spec["enum"]) == expected


def test_taxonomy_defaults_to_source_in_the_schema(listed):
    """Obligation 1: the default is declared, not inferred."""
    for name in ("search_incidents", "aggregate_incidents", "nearby_incidents"):
        assert listed[name].parameters["properties"]["taxonomy"]["default"] == "source"


@pytest.mark.parametrize(
    ("tool", "required"),
    [
        ("describe_schema", set()),
        ("get_incident", set()),
        ("search_incidents", {"start", "end"}),
        ("aggregate_incidents", {"start", "end"}),
        ("nearby_incidents", {"latitude", "longitude", "radius_m", "start", "end"}),
    ],
)
def test_required_arguments(listed, tool, required):
    assert set(listed[tool].parameters.get("required") or []) == required


def test_query_tool_payloads_are_named_objects(listed):
    """`data` is always an object with named fields, never sometimes an array.

    Asserted on the schema FastMCP publishes, which inlines the model reference
    rather than emitting a ``$ref``. So the check is on the resulting shape:
    ``data`` is an ``object`` with properties, even for the three payloads that
    carry a single list.
    """
    for name in ("get_incident", "search_incidents", "aggregate_incidents", "nearby_incidents"):
        data = listed[name].output_schema["properties"]["data"]
        assert data["type"] == "object", name
        assert data["properties"], name
        assert data["type"] != "array", name


def test_envelope_names_the_taxonomy_mode(listed):
    """Obligation 2, as the client sees it."""
    for name in ("search_incidents", "aggregate_incidents", "nearby_incidents"):
        assert "taxonomy_mode" in listed[name].output_schema["properties"]


# --- how errors reach the model ---------------------------------------------


def test_our_errors_are_fastmcp_tool_errors():
    """This inheritance is what gets the teaching message to the model intact.

    FastMCP delivers a ``ToolError``'s message verbatim; any other exception is
    re-wrapped behind a prefix, is liable to be replaced when
    ``mask_error_details`` is on, and dumps a rendered traceback on every
    occurrence -- for what is the expected path here. Translating at the server
    boundary cannot substitute, because FastMCP catches the exception below the
    middleware chain.
    """
    assert issubclass(UnknownValueError, FastMCPToolError)


def test_rendered_message_survives_as_the_exception_string():
    """MCP sends a tool failure as text, so __str__ has to carry everything."""
    rendered = str(
        UnknownValueError(
            "no such category.",
            field="types",
            received="BATERY",
            valid_values=("BATTERY", "THEFT"),
        )
    )
    assert "types" in rendered
    assert "BATERY" in rendered
    assert "Did you mean 'BATTERY'?" in rendered
    assert "Valid values" in rendered


def test_a_teaching_error_reaches_middleware_as_our_own_class():
    """The load-bearing fact behind every error record, driven end to end.

    FastMCP catches a tool's exception *below* the middleware chain, which is
    why an earlier boundary translation in this project was dead code. What
    makes the telemetry path live instead is that our ``ToolError`` subclasses
    FastMCP's, so FastMCP re-raises it rather than wrapping it, and the
    structured fields survive to be logged.

    That is a claim about FastMCP's behaviour, so a stubbed ``call_next`` cannot
    check it -- a stub would pass no matter what the framework did. This drives
    a real server through a real client.
    """
    sink = RecordingSink()
    seen = {}

    class Capture(Middleware):
        async def on_call_tool(self, context, call_next):
            try:
                return await call_next(context)
            except Exception as exc:
                seen["exc"] = exc
                raise

    app = FastMCP(name="probe")
    app.add_middleware(CallTelemetryMiddleware(sink=sink))
    app.add_middleware(Capture())

    @app.tool
    def picky(types: str) -> dict:
        """A tool that rejects its argument."""
        raise UnknownValueError(
            "no such category.", field="types", received=types, valid_values=("BATTERY", "THEFT")
        )

    async def call():
        async with Client(app) as client:
            await client.call_tool("picky", {"types": "BATERY"})

    with pytest.raises(FastMCPToolError):
        asyncio.run(call())

    assert isinstance(seen["exc"], UnknownValueError), (
        "FastMCP wrapped our error instead of re-raising it; the telemetry "
        "middleware's except-clause would be dead code"
    )
    assert seen["exc"].details()["field"] == "types"

    record = sink.only
    assert (record.tool, record.outcome) == ("picky", "error")
    assert (record.error_code, record.error_field) == ("unknown_value", "types")
    assert record.error_received == "BATERY"
    assert record.error_nearest_match == "BATTERY"


def test_a_successful_call_is_recorded_from_the_serialized_envelope():
    """The middleware reads the response, so no tool reports anything twice."""
    sink = RecordingSink()
    app = FastMCP(name="probe")
    app.add_middleware(CallTelemetryMiddleware(sink=sink))

    @app.tool
    def counted() -> Envelope:
        """A tool returning an envelope-shaped result."""
        return _envelope(
            row_count=3,
            truncated=True,
            cursor="abc",
            taxonomy_mode="comparable",
            route=RouteInfo(
                store="duckdb", tier="rollup", table="monthly", reason="x", elapsed_ms=1.5
            ),
            warnings=[ResultWarning(code="provisional", message="m")],
        )

    async def call():
        async with Client(app) as client:
            await client.call_tool("counted", {})

    asyncio.run(call())

    record = sink.only
    assert record.outcome == "ok"
    assert (record.row_count, record.truncated, record.cursor_issued) == (3, True, True)
    assert (record.route_store, record.route_tier, record.route_table) == (
        "duckdb",
        "rollup",
        "monthly",
    )
    assert record.route_elapsed_ms == 1.5
    assert record.taxonomy_mode == "comparable"
    assert record.warning_codes == ["provisional"]
    # Measured, not echoed: the envelope carries no size, so this is the
    # middleware serializing the response it actually saw.
    assert record.result_bytes > 0
    assert record.duration_ms > 0


def test_a_real_schema_rejection_is_classified_as_schema_validation():
    """Driven for real, because the claim is about what FastMCP raises.

    A call missing a required argument never enters the tool, so no teaching
    error can be raised and FastMCP rejects it itself. The stubbed tests assert
    we classify a FastMCPToolError correctly; only this one shows that argument
    rejection actually arrives as one.
    """
    sink = RecordingSink()
    app = FastMCP(name="probe")
    app.add_middleware(CallTelemetryMiddleware(sink=sink))

    @app.tool
    def needs_a_span(start: str, end: str) -> Envelope:
        """A tool with required arguments."""
        return _envelope(row_count=0, route=RouteInfo(store="duckdb", reason="x", elapsed_ms=1.0))

    async def call():
        async with Client(app) as client:
            await client.call_tool("needs_a_span", {})

    with pytest.raises(FastMCPToolError):
        asyncio.run(call())

    record = sink.only
    assert record.outcome == "error"
    assert record.error_code == "schema_validation", (
        "argument rejection was misfiled; the rollup would report it as a bug"
    )
    assert record.error_message


def test_an_empty_result_is_recorded_as_empty_not_ok():
    """'Valid filters, nothing matched' is the signal; it must not read as success."""
    sink = RecordingSink()
    app = FastMCP(name="probe")
    app.add_middleware(CallTelemetryMiddleware(sink=sink))

    @app.tool
    def nothing() -> Envelope:
        """A tool whose filters matched no rows."""
        return _envelope(
            row_count=0,
            route=RouteInfo(store="postgres", reason="x", elapsed_ms=1.0),
            warnings=[ResultWarning(code="empty_result", message="m")],
        )

    async def call():
        async with Client(app) as client:
            await client.call_tool("nothing", {})

    asyncio.run(call())
    assert sink.only.outcome == "empty"
