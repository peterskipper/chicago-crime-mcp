"""Storage & query routing. Postgres/PostGIS for point lookups and spatial
joins, DuckDB (over Parquet + materialized rollups) for OLAP aggregates. Each
query routes to the right engine. There is no cache tier: one was planned and
cut after measurement -- see "Why there is no cache" in the README.
"""
