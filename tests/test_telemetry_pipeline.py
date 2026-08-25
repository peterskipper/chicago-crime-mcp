"""The whole pipeline joined up: a real server writes a log, the rollup reads it.

Every other telemetry test checks one seam. The middleware tests build a
transcript by hand; the rollup tests build records by hand. Both would keep
passing if the two halves disagreed about the file — a record field renamed on
one side and not the other reads as NULL forever, which is exactly the failure
the declared schema is meant to prevent and exactly the failure a hand-built
fixture cannot show.

So this drives a real FastMCP server through a real client, lets the real sink
write a real file, and runs the real rollup over it. Hermetic: the tools here are
stand-ins with the shapes that matter (a list-valued argument, an envelope, an
empty result, a teaching error), so no database is needed.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import asyncio

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError as FastMCPToolError
from pydantic import BaseModel

from chicago_crime_mcp.server.envelope import ResultWarning, RouteInfo, ToolResult
from chicago_crime_mcp.server.errors import UnknownValueError
from chicago_crime_mcp.telemetry import rollup
from chicago_crime_mcp.telemetry.middleware import CallTelemetryMiddleware
from chicago_crime_mcp.telemetry.sink import TelemetryConfig, TelemetrySink


class _Payload(BaseModel):
    """A stand-in payload."""

    note: str = "x"


class _Filters(BaseModel):
    """A stand-in filter echo."""

    where: str = "x"


Envelope = ToolResult[_Payload, _Filters]


def _build(sink: TelemetrySink) -> FastMCP:
    """A server whose tools cover the outcome and argument shapes that matter."""
    app = FastMCP(name="pipeline")
    app.add_middleware(CallTelemetryMiddleware(sink=sink))

    @app.tool
    def aggregate_incidents(geography: str, types: list[str], limit: int = 50) -> Envelope:
        """Aggregate offenses."""
        return Envelope(
            data=_Payload(),
            filters_applied=_Filters(),
            row_count=12,
            route=RouteInfo(
                store="duckdb", tier="rollup", table="monthly", reason="grain", elapsed_ms=4.0
            ),
            taxonomy_mode="comparable",
            warnings=[ResultWarning(code="provisional", message="still filling")],
        )

    @app.tool
    def search_incidents(types: list[str], limit: int = 50) -> Envelope:
        """Search offenses, teaching on a bad category and finding none on a good one."""
        if "BATERY" in types:
            raise UnknownValueError(
                "no such category.",
                field="types",
                received="BATERY",
                valid_values=("BATTERY", "THEFT"),
            )
        return Envelope(
            data=_Payload(),
            filters_applied=_Filters(),
            row_count=0,
            route=RouteInfo(store="postgres", reason="selective", elapsed_ms=9.0),
            warnings=[ResultWarning(code="empty_result", message="nothing matched")],
        )

    @app.tool
    def resolve_neighborhood(name: str) -> dict:
        """Resolve a neighborhood name, guessing when it has no boundary."""
        return {
            "query": name,
            "resolved": name == "Wicker Park",
            "candidates": [
                {"match_kind": "exact" if name == "Wicker Park" else "suggestion", "label": name}
            ],
            "route": {"store": "reference", "reason": "pinned", "elapsed_ms": 0.3},
        }

    return app


@pytest.fixture
def rolled(tmp_path):
    """Drive a real server, then return a connection over the log it produced."""
    sink = TelemetrySink(TelemetryConfig(log_dir=tmp_path / "telemetry"))
    app = _build(sink)

    async def drive():
        async with Client(app) as client:
            await client.call_tool(
                "aggregate_incidents", {"geography": "community_area", "types": ["BATTERY"]}
            )
            await client.call_tool("search_incidents", {"types": ["ARSON"]})
            with pytest.raises(FastMCPToolError):
                await client.call_tool("search_incidents", {"types": ["BATERY"]})
            with pytest.raises(FastMCPToolError):
                await client.call_tool("search_incidents", {})  # schema rejection
            await client.call_tool("resolve_neighborhood", {"name": "Wicker Park"})
            await client.call_tool("resolve_neighborhood", {"name": "Bronzeville"})
            await client.call_tool("resolve_neighborhood", {"name": "Bronzeville"})

    asyncio.run(drive())
    sink.close()

    conn = rollup.connect(tmp_path / "telemetry")
    yield conn
    conn.close()


def test_the_rollup_reads_every_record_the_server_wrote(rolled):
    assert rolled.execute("SELECT count(*) FROM calls").fetchone()[0] == 7


def test_no_declared_column_reads_null_for_every_row(rolled):
    """The failure the declared schema exists to prevent, stated as a test.

    A field renamed on the record side and not in COLUMNS would parse without
    complaint and be NULL forever. Any column that is null in *every* row is
    either that bug or a column no tool exercises, and both want a human.
    """
    # Derived from COLUMNS rather than copied, so a new record field has to be
    # either exercised above or justified here. Only two are legitimately empty:
    # both are volunteered by the client, and the in-memory transport volunteers
    # neither.
    not_exercised = {"client_id", "transport"}
    for column in set(rollup.COLUMNS) - not_exercised:
        # Column names come from our own COLUMNS constant, never from input.
        non_null = rolled.execute(
            f"SELECT count(*) FROM calls WHERE {column} IS NOT NULL"
        ).fetchone()[0]
        assert non_null > 0, f"{column} was NULL in every row -- schema drift?"


def test_a_list_valued_argument_survives_the_round_trip(rolled):
    """`types` is a list; JSON typing of args is where a naive schema breaks."""
    row = rolled.execute(
        "SELECT json_extract(args, '$.types') FROM calls WHERE tool = 'aggregate_incidents'"
    ).fetchone()[0]
    assert "BATTERY" in row


def test_outcomes_are_classified_across_the_real_paths(rolled):
    counts = dict(
        rolled.execute("SELECT outcome, count(*) FROM calls GROUP BY outcome").fetchall()
    )
    assert counts == {"ok": 4, "empty": 1, "error": 2}


def test_the_error_families_stay_separable_end_to_end(rolled):
    """A teaching error and a schema rejection travelled the whole pipeline."""
    codes = dict(
        rolled.execute(
            "SELECT error_code, count(*) FROM calls WHERE outcome = 'error' GROUP BY error_code"
        ).fetchall()
    )
    assert codes == {"unknown_value": 1, "schema_validation": 1}


def test_the_alias_backlog_names_only_the_unresolved(rolled):
    """The end-to-end version of the point: a soft miss is a successful call."""
    rows = rollup.resolution_misses(rolled).rows
    assert [(r["name"], r["asked"]) for r in rows] == [("Bronzeville", 2)]


def test_the_empty_report_names_the_filters_that_found_nothing(rolled):
    rows = rollup.empty_by_filters(rolled).rows
    assert rows[0]["tool"] == "search_incidents"
    assert rows[0]["filters"] == ["types"], "limit leaked into the filter combination"


def test_every_report_renders_over_real_records(rolled):
    """A report that crashes on real data is worse than one that finds nothing."""
    reports = [fn(rolled) for fn in rollup.REPORTS]
    text = rollup.render(reports)
    assert "aggregate_incidents" in text
    assert "Bronzeville" in text
    assert "duckdb" in text
