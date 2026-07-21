"""Storage & query routing. Postgres/PostGIS for point lookups and spatial
joins, DuckDB (over Parquet + materialized rollups) for OLAP aggregates, Redis
for cache-aside on hot results. Each query routes to the right engine.
"""
