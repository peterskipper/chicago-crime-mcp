"""The MCP tool surface: five purpose-built tools, no SQL escape hatch.

Each tool is a plain function taking typed arguments and returning a validated
model. They share one shape:

1. read the vocabulary for the current rollup build;
2. validate the data-derived arguments against it, raising a teaching error that
   names the field, echoes the value and offers the nearest valid one;
3. build the store layer's frozen query dataclass;
4. run it, on whichever store answers that question shape;
5. map the result into its payload model and assemble the envelope, attaching
   the warnings the store's facts imply.

Steps 2 and 5 are the reason this layer exists at all. The stores return facts
and phrase nothing; the tools do the phrasing, so a wording change never
reaches into a query plan and the same facts stay loggable and cacheable.

Docstrings follow the Google Python style.
"""

from chicago_crime_mcp.server.tools.aggregate_incidents import aggregate_incidents
from chicago_crime_mcp.server.tools.describe_schema import describe_schema
from chicago_crime_mcp.server.tools.get_incident import get_incident
from chicago_crime_mcp.server.tools.nearby_incidents import nearby_incidents
from chicago_crime_mcp.server.tools.search_incidents import search_incidents

__all__ = [
    "aggregate_incidents",
    "describe_schema",
    "get_incident",
    "nearby_incidents",
    "search_incidents",
]
