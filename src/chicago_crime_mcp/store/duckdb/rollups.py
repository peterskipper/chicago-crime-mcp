"""Build the DuckDB materialized rollup tables over the Parquet dataset.

This is the aggregate arm of the query router. Three tiers serve a question,
cheapest first:

1. point lookup / radius / selective search -> **Postgres** (see store.postgres)
2. aggregate on an aligned grain (month, quarter or year x one geography)
   -> **these materialized rollups**
3. aggregate on an arbitrary range ("last 30 days", Jan 5 - Feb 20)
   -> **a live DuckDB scan** over the same Parquet, via the ``incidents`` view

Tier 3 matters: month grain genuinely cannot answer an arbitrary date range, and
a documented fallback beats pretending otherwise. The rollups are a latency and
cacheability tier (they are what Redis will front), not a rescue for a slow
query -- a live scan of 2.88M rows already answers in ~24ms.

**Refresh is always a full rebuild.** ``CREATE OR REPLACE TABLE`` over the whole
dataset measured 0.5s at 2.88M rows, so incremental rollup maintenance would add
real complexity to save milliseconds. This is a deliberate contrast with the
Postgres loader, where a 2.9M-row ``COPY`` did justify a separate upsert path.

The grain, measure and null-handling decisions live in ``rollups.sql`` next to
the SQL they govern; that file is the single source of truth for the DDL.

Note on the package name: this module lives in ``store/duckdb/`` (mirroring
``store/postgres/``) yet ``import duckdb`` below resolves to the third-party
package -- Python 3 imports are absolute, so the sibling directory cannot shadow
it.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from chicago_crime_mcp.reference import IUCR_REFERENCE_PATH
from chicago_crime_mcp.store.config import DEFAULT_DUCKDB_PATH, DEFAULT_PARQUET_ROOT

log = logging.getLogger(__name__)

ROLLUPS_PATH = Path(__file__).parent / "rollups.sql"

# The view every rollup reads from, and the fallback surface for tier-3 live
# scans. Named to match the Postgres table so a query means the same thing in
# either engine.
SOURCE_VIEW = "incidents"

# The committed IUCR snapshot, loaded as a table so the rollups can tag each
# incident with its stable_category.
REFERENCE_TABLE = "iucr_reference"

# Rollup tables built by rollups.sql, finest geography first. Kept here so the
# builder can report per-table row counts and the tests can assert the
# SUM(incidents) invariant across every one of them without hardcoding a
# second list.
ROLLUP_TABLES = (
    "rollup_citywide",
    "rollup_beat",
    "rollup_district",
    "rollup_community_area",
    "rollup_ward",
)

# Not a rollup: one row per IUCR code with its observed lifespan. Kept out of
# ROLLUP_TABLES because nothing routes an aggregate query to it -- it is metadata
# the tool layer consults to warn that a requested span crosses a code's
# introduction or retirement.
COVERAGE_TABLE = "code_coverage"


def ensure_parquet_view(
    conn: duckdb.DuckDBPyConnection,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
) -> None:
    """(Re)create the ``incidents`` view over the Hive-partitioned Parquet.

    DuckDB stores a view as its literal SQL text, so the Parquet path is baked in
    at creation time. Recreating the view on every connect means a moved or
    reconfigured ``parquet_root`` can never leave a stale path behind in the
    database file. The path is resolved to an absolute one so the view does not
    depend on the process working directory.

    ``hive_partitioning=1`` reconstructs the ``year`` column from the
    ``year=<YYYY>`` directory names; it is not stored in the files.

    Args:
        conn: An open DuckDB connection.
        parquet_root: Root of the Hive-partitioned Parquet dataset.

    Raises:
        FileNotFoundError: If no partitions exist under ``parquet_root``. Checked
            eagerly here because a glob matching nothing would otherwise surface
            as a confusing error at first query.
    """
    root = parquet_root.resolve()
    if not list(root.glob("year=*/part.parquet")):
        raise FileNotFoundError(f"no year=*/part.parquet partitions under {root}")

    # The path must be inlined, not bound: DuckDB cannot prepare a DDL statement,
    # so a `?` placeholder here fails to bind. Single quotes are doubled to escape
    # them -- the value is a local filesystem path from config, never user input.
    pattern = str(root / "year=*" / "part.parquet").replace("'", "''")
    conn.execute(
        f"CREATE OR REPLACE VIEW {SOURCE_VIEW} AS "
        f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=1)"
    )


def ensure_iucr_reference(
    conn: duckdb.DuckDBPyConnection,
    reference_path: Path = IUCR_REFERENCE_PATH,
) -> None:
    """(Re)load the committed IUCR snapshot into the ``iucr_reference`` table.

    A table, not a view, because the CSV is small (~435 rows) and a view would
    re-read and re-parse the file on every query. Reloaded on every non-read-only
    connect for the same reason the Parquet view is recreated: the file is
    git-tracked and can change under the database between runs.

    Every column is read as text (``all_varchar``): IUCR codes are zero-padded
    identifiers like ``0110`` and include non-numeric ones like ``142A``, so type
    inference would mangle them. ``stable_category`` is left blank for codes with
    no curated override, and ``nullstr`` turns those blanks into NULL so the
    coalesce in ``incidents_tagged`` falls through to the canonical type.

    Args:
        conn: An open DuckDB connection.
        reference_path: Path to the IUCR snapshot CSV.

    Raises:
        FileNotFoundError: If the snapshot is missing.
    """
    path = reference_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"IUCR reference snapshot not found at {path}")

    # Inlined for the same reason as the Parquet path: DuckDB cannot prepare DDL.
    # The value comes from packaged data, never from user input.
    literal = str(path).replace("'", "''")
    conn.execute(
        f"CREATE OR REPLACE TABLE {REFERENCE_TABLE} AS "
        "SELECT lpad(iucr, 4, '0') AS iucr, primary_description, stable_category "
        f"FROM read_csv('{literal}', all_varchar=true, nullstr='')"
    )


def connect(
    duckdb_path: Path = DEFAULT_DUCKDB_PATH,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open the rollup database with the ``incidents`` view in place.

    Creates the parent directory of ``duckdb_path`` if needed, so a fresh
    checkout works without a manual mkdir.

    Args:
        duckdb_path: Path to the persistent DuckDB database file.
        parquet_root: Root of the Hive-partitioned Parquet dataset.
        read_only: Open read-only. Query paths should pass ``True``; the view and
            the reference table are then assumed to already exist (a read-only
            connection cannot create them), which holds for any database a build
            has run against.

    Returns:
        An open connection.
    """
    if not read_only:
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(duckdb_path), read_only=read_only)
    if not read_only:
        ensure_parquet_view(conn, parquet_root)
        ensure_iucr_reference(conn)
    return conn


def build(conn: duckdb.DuckDBPyConnection) -> dict:
    """Rebuild every rollup table from the ``incidents`` view.

    Runs ``rollups.sql`` inside a single transaction, so a failure part-way
    leaves the previous rollups intact rather than a half-rebuilt set.

    Args:
        conn: An open connection with the ``incidents`` view present (use
            :func:`connect`).

    Returns:
        A summary dict mapping each rollup table name (plus ``code_coverage``) to
        its row count, plus ``source_rows`` (the number of incidents rolled up).
    """
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(ROLLUPS_PATH.read_text())
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")

    tables = (*ROLLUP_TABLES, COVERAGE_TABLE)
    summary = {
        table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in tables
    }
    summary["source_rows"] = conn.execute("SELECT source_rows FROM rollup_meta").fetchone()[0]

    for table in tables:
        log.info("built %s: %d rows", table, summary[table])
    return summary


def main() -> None:
    """CLI entry point: rebuild the rollup tables from the configured Parquet.

    Reads paths from the environment via :class:`StoreConfig`.
    """
    import argparse

    from chicago_crime_mcp.store.config import StoreConfig

    parser = argparse.ArgumentParser(description="Rebuild the DuckDB rollup tables.")
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=None,
        help="Root of the Hive-partitioned Parquet dataset (defaults to config).",
    )
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=None,
        help="Path to the DuckDB database file (defaults to config).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = StoreConfig.from_env()

    conn = connect(
        duckdb_path=args.duckdb_path or config.duckdb_path,
        parquet_root=args.parquet_root or config.parquet_root,
    )
    try:
        summary = build(conn)
    finally:
        conn.close()
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
