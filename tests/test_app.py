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
from fastmcp.exceptions import ToolError as FastMCPToolError

from chicago_crime_mcp.server import tools
from chicago_crime_mcp.server.app import TOOLS, ToolErrorTelemetryMiddleware, create_app
from chicago_crime_mcp.server.errors import UnknownValueError


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
            {"citywide", "beat", "district", "community_area", "ward"},
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


def _run(middleware, call_next):
    """Drive the middleware once with a stubbed call chain."""

    class _Message:
        name = "aggregate_incidents"

    class _Context:
        message = _Message()

    return asyncio.run(middleware.on_call_tool(_Context(), call_next))


def test_middleware_logs_the_structured_form(caplog):
    """The fields are logged before the wire flattens them into a string."""

    async def call_next(_context):
        raise UnknownValueError(
            "no such category.", field="types", received="BATERY",
            valid_values=("BATTERY", "THEFT"),
        )

    with caplog.at_level("INFO", logger="chicago_crime_mcp.server.app"):
        with pytest.raises(UnknownValueError):
            _run(ToolErrorTelemetryMiddleware(), call_next)
    record = next(r for r in caplog.records if "tool error" in r.getMessage())
    assert "'field': 'types'" in record.getMessage()
    assert "'code': 'unknown_value'" in record.getMessage()


def test_middleware_re_raises_unchanged(caplog):
    """It observes; it must not alter what the caller receives."""
    error = UnknownValueError("no such category.", field="types")

    async def call_next(_context):
        raise error

    with pytest.raises(UnknownValueError) as exc:
        _run(ToolErrorTelemetryMiddleware(), call_next)
    assert exc.value is error


def test_middleware_passes_a_successful_call_through():
    async def call_next(_context):
        return "result"

    assert _run(ToolErrorTelemetryMiddleware(), call_next) == "result"


def test_middleware_does_not_touch_unrelated_errors():
    """An operational bug must not be dressed up as a bad argument."""

    async def call_next(_context):
        raise RuntimeError("something broke")

    with pytest.raises(RuntimeError):
        _run(ToolErrorTelemetryMiddleware(), call_next)
