"""chicago-crime-mcp: a constrained MCP server over Chicago crime data.

Architecture: ingest -> store -> serve. The server never calls Socrata live;
it serves from a local store with deliberate query routing. See README.
"""

__version__ = "0.1.0"
