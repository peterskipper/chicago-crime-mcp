"""Tests for how a call becomes a record.

The framework contract -- that FastMCP delivers our error class to the
middleware intact, and hands it a serialized envelope -- is proved against a
real server in ``tests/test_app.py``, because a stub would pass no matter what
the framework did. What is left here is our own classification logic, and stubs
are the right instrument for that: they can produce shapes a real tool never
would, which is exactly where the branches are.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chicago_crime_mcp.server.errors import InvalidArgumentError, UnknownValueError
from chicago_crime_mcp.telemetry.middleware import CallTelemetryMiddleware
from tests.helpers import RecordingSink


def _drive(body=None, *, raises=None, tool="search_incidents", args=None, context=None):
    """Run the middleware once and return the record it wrote.

    Args:
        body: The ``structured_content`` the stubbed chain returns.
        raises: An exception for the stubbed chain to raise instead.
        tool: The tool name on the stubbed message.
        args: The arguments on the stubbed message.
        context: A replacement ``fastmcp_context``.

    Returns:
        A ``(record, sink)`` pair.
    """
    sink = RecordingSink()
    middleware = CallTelemetryMiddleware(sink=sink)
    call_context = SimpleNamespace(
        message=SimpleNamespace(name=tool, arguments=args or {}),
        fastmcp_context=context,
    )

    async def call_next(_context):
        if raises is not None:
            raise raises
        return SimpleNamespace(structured_content=body)

    async def run():
        return await middleware.on_call_tool(call_context, call_next)

    if raises is not None:
        with pytest.raises(type(raises)):
            asyncio.run(run())
    else:
        asyncio.run(run())
    return sink.only, sink


def _envelope(**kwargs):
    """A serialized envelope with the fields these tests do not vary."""
    return {
        "row_count": 1,
        "truncated": False,
        "cursor": None,
        "taxonomy_mode": "source",
        "route": {"store": "postgres", "reason": "point lookup", "elapsed_ms": 2.0},
        "warnings": [],
    } | kwargs


# --- what the model sent -----------------------------------------------------


def test_arguments_are_recorded_as_received_not_normalized():
    """The telemetry question is what the *model* sent, typos and all."""
    record, _ = _drive(_envelope(), args={"district": "10", "types": ["battery"]})
    assert record.args == {"district": "10", "types": ["battery"]}


def test_the_recorded_arguments_are_a_copy():
    """Mutating the message afterwards must not rewrite history."""
    args = {"district": "10"}
    record, _ = _drive(_envelope(), args=args)
    args["district"] = "999"
    assert record.args == {"district": "10"}


# --- outcome classification --------------------------------------------------


def test_rows_returned_is_ok():
    record, _ = _drive(_envelope(row_count=5))
    assert record.outcome == "ok"


def test_the_empty_result_warning_is_what_makes_a_call_empty():
    """Not the row count: describe_schema returns no rows and is not empty."""
    record, _ = _drive(
        _envelope(row_count=0, warnings=[{"code": "empty_result", "message": "m"}])
    )
    assert record.outcome == "empty"


def test_zero_rows_without_the_warning_is_still_ok():
    """The negative half of the rule above."""
    record, _ = _drive(_envelope(row_count=0))
    assert record.outcome == "ok"


def test_other_warnings_do_not_make_a_call_empty():
    record, _ = _drive(_envelope(warnings=[{"code": "provisional", "message": "m"}]))
    assert record.outcome == "ok"
    assert record.warning_codes == ["provisional"]


def test_warning_codes_are_collected_in_order_and_messages_dropped():
    """Codes are the closed vocabulary; prose is not what gets counted."""
    record, _ = _drive(
        _envelope(
            warnings=[
                {"code": "provisional", "message": "a"},
                {"code": "truncated", "message": "b"},
            ]
        )
    )
    assert record.warning_codes == ["provisional", "truncated"]


def test_a_malformed_warning_entry_is_skipped_rather_than_crashing():
    """Telemetry must not fail a call over a shape it did not expect."""
    record, _ = _drive(_envelope(warnings=["not a dict", {"message": "no code"}]))
    assert record.warning_codes == []
    assert record.outcome == "ok"


# --- the route ---------------------------------------------------------------


def test_route_fields_are_flattened():
    record, _ = _drive(
        _envelope(
            route={
                "store": "duckdb",
                "tier": "scan",
                "table": "incidents",
                "reason": "span below the month grain",
                "elapsed_ms": 7.25,
            }
        )
    )
    assert record.route_store == "duckdb"
    assert record.route_tier == "scan"
    assert record.route_table == "incidents"
    assert record.route_reason == "span below the month grain"
    assert record.route_elapsed_ms == 7.25


def test_a_store_without_a_tier_records_no_tier():
    """Postgres has one tier, so inventing a name for it would be a fiction."""
    record, _ = _drive(_envelope())
    assert record.route_store == "postgres"
    assert record.route_tier is None
    assert record.route_table is None


def test_a_missing_route_does_not_crash_the_record():
    record, _ = _drive({"row_count": 0})
    assert record.route_store is None
    assert record.outcome == "ok"


def test_cursor_issued_is_a_flag_not_the_cursor_itself():
    """An opaque cursor is state, not a fact worth keeping; its presence is."""
    with_cursor, _ = _drive(_envelope(cursor="opaque-token"))
    without, _ = _drive(_envelope(cursor=None))
    assert with_cursor.cursor_issued is True
    assert without.cursor_issued is False
    assert "opaque-token" not in str(with_cursor.to_dict())


def test_result_bytes_grows_with_the_response():
    """It is measured from the response, not echoed out of it."""
    small, _ = _drive(_envelope(row_count=1))
    large, _ = _drive(_envelope(row_count=1, data={"incidents": ["x" * 500]}))
    assert large.result_bytes > small.result_bytes


def test_a_result_fastmcp_could_not_structure_is_still_recorded():
    """None of our tools do this; it should stop being true loudly, not silently."""
    record, _ = _drive(None)
    assert record.outcome == "ok"
    assert record.row_count is None


# --- errors ------------------------------------------------------------------


def test_a_teaching_error_records_its_structured_fields():
    error = UnknownValueError(
        "no such category.", field="types", received="BATERY", valid_values=("BATTERY",)
    )
    record, _ = _drive(raises=error)
    assert record.outcome == "error"
    assert record.error_code == "unknown_value"
    assert record.error_field == "types"
    assert record.error_received == "BATERY"
    assert record.error_nearest_match == "BATTERY"


def test_an_error_without_a_suggestion_records_none():
    """The negative half: a suggestion is offered, not always present."""
    record, _ = _drive(raises=InvalidArgumentError("bad span", field="start_date"))
    assert record.error_field == "start_date"
    assert record.error_nearest_match is None
    assert record.error_received is None


def test_a_non_string_received_value_is_stringified():
    """Grouping on 'which values does the model invent' needs one type."""
    record, _ = _drive(raises=InvalidArgumentError("too big", field="limit", received=10_000))
    assert record.error_received == "10000"


def test_an_unhandled_exception_is_recorded_but_not_as_a_teaching_error():
    """A bug and a self-correcting loop must be separable in the logs."""
    record, _ = _drive(raises=RuntimeError("connection reset"))
    assert record.outcome == "error"
    assert record.error_code == "unhandled"
    assert "RuntimeError: connection reset" not in (record.error_field or "")
    assert record.error_field is None


def test_the_exception_reaches_the_caller_unchanged():
    """It observes; it must not alter what the model is told."""
    sink = RecordingSink()
    middleware = CallTelemetryMiddleware(sink=sink)
    error = UnknownValueError("no such category.", field="types")

    async def call_next(_context):
        raise error

    context = SimpleNamespace(
        message=SimpleNamespace(name="search_incidents", arguments={}), fastmcp_context=None
    )
    with pytest.raises(UnknownValueError) as caught:
        asyncio.run(middleware.on_call_tool(context, call_next))
    assert caught.value is error


def test_a_base_exception_is_re_raised_rather_than_masked():
    """Cancellation must not come back as an UnboundLocalError from the finally."""
    sink = RecordingSink()
    middleware = CallTelemetryMiddleware(sink=sink)

    async def call_next(_context):
        raise asyncio.CancelledError

    context = SimpleNamespace(
        message=SimpleNamespace(name="search_incidents", arguments={}), fastmcp_context=None
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(middleware.on_call_tool(context, call_next))
    assert sink.records == []


# --- resolve_neighborhood, the alias backlog ---------------------------------


def test_an_unresolved_name_is_recorded_as_a_soft_miss():
    """The whole point: this call *succeeded*, so no error count would find it."""
    record, _ = _drive(
        {"resolved": False, "candidates": [{"match_kind": "suggestion"}], "route": {}},
        tool="resolve_neighborhood",
        args={"name": "Bronzeville"},
    )
    assert record.resolution_kind == "suggestion"
    assert record.outcome == "ok"
    assert record.args == {"name": "Bronzeville"}


@pytest.mark.parametrize("kind", ["exact", "alias", "containing"])
def test_a_resolved_name_records_the_winning_match_kind(kind):
    record, _ = _drive(
        {"resolved": True, "candidates": [{"match_kind": kind}], "route": {}},
        tool="resolve_neighborhood",
    )
    assert record.resolution_kind == kind


def test_a_hard_miss_is_recorded_as_none():
    """Nothing matched at all, and an error was raised."""
    record, _ = _drive(
        raises=UnknownValueError("no such place.", field="name", received="Nowheresville"),
        tool="resolve_neighborhood",
    )
    assert record.resolution_kind == "none"
    assert record.outcome == "error"


def test_other_tools_never_carry_a_resolution_kind():
    """The negative half: this column belongs to one tool."""
    ok, _ = _drive(_envelope())
    failed, _ = _drive(raises=UnknownValueError("x", field="types"))
    assert ok.resolution_kind is None
    assert failed.resolution_kind is None


# --- identity ----------------------------------------------------------------


def test_identity_is_read_from_the_fastmcp_context():
    context = SimpleNamespace(session_id="s-1", client_id="c-1", transport="http")
    record, _ = _drive(_envelope(), context=context)
    assert (record.session_id, record.client_id, record.transport) == ("s-1", "c-1", "http")


def test_a_missing_identity_field_records_none():
    """Which fields exist is a property of the transport, not of the protocol."""
    record, _ = _drive(_envelope(), context=SimpleNamespace(session_id="s-1"))
    assert record.session_id == "s-1"
    assert record.client_id is None
    assert record.transport is None


def test_an_identity_property_that_raises_does_not_fail_the_call():
    """These are properties that reach for a request context; none is worth a 500."""

    class Hostile:
        session_id = "s-1"

        @property
        def transport(self):
            raise RuntimeError("no active request")

        @property
        def client_id(self):
            raise RuntimeError("no active request")

    record, _ = _drive(_envelope(), context=Hostile())
    assert record.session_id == "s-1"
    assert record.transport is None
    assert record.client_id is None
