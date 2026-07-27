"""Full-refresh bulk load of the Parquet dataset into Postgres/PostGIS.

The loader rebuilds the ``incidents`` table from the Hive-partitioned Parquet:
it applies the schema, truncates, streams every partition in via ``COPY``, then
(re)builds the indexes. Indexes are dropped before the ``COPY`` and recreated
after - maintaining them per-row across an ~8M-row load is far slower than
building each once at the end.

Two entry points share the Parquet-reading helpers:

* ``load`` -- full-refresh rebuild (truncate + COPY + reindex). Repeatable; each
  run fully replaces the table. Used for the initial load and as the periodic
  reconciliation sweep.
* ``upsert`` -- apply (whole) changed partitions to a *live* table via
  COPY-to-staging + ``INSERT ... ON CONFLICT (id) DO UPDATE``. Used for the
  nightly incremental. It never truncates and never drops indexes. It also never
  deletes, so a row retracted at the source lingers until the next full-refresh
  ``load`` reconciles it -- an accepted tradeoff for the nightly path.

Both are year-agnostic: partitions are discovered by globbing ``year=*`` (see
``discover_partitions``/``select_partitions``), so a new current-year partition
(e.g. ``year=2026``) flows through with no code change once ingest lands it.

The set of columns to load is read from the Parquet files themselves, so it can
never drift from what ``ingest`` wrote: if a Parquet column has no matching table
column, ``COPY`` fails loudly. ``geom`` (a generated column) and ``year`` (a
partition key) are absent from the payload, so "Parquet columns" is exactly the
set to copy. Index definitions live solely in ``schema.sql``; the loader
recreates them by re-running that file and drops them dynamically from
``pg_indexes``, so there is no second copy of the DDL to drift.

The connection is injected so tests can supply their own.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pyarrow.parquet as pq
from psycopg import sql

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Root of the Hive-partitioned Parquet dataset (``year=<YYYY>/part.parquet``).
DEFAULT_PARQUET_ROOT = Path("data/parquet")

# Rows per Parquet read batch handed to COPY. Bounds peak memory while keeping
# the COPY stream busy.
READ_BATCH_SIZE = 50_000


def discover_partitions(parquet_root: Path = DEFAULT_PARQUET_ROOT) -> list[Path]:
    """Find the Parquet partition files under a Hive-partitioned root.

    Args:
        parquet_root: Directory holding ``year=<YYYY>/part.parquet`` partitions.

    Returns:
        The partition file paths, sorted by year for deterministic load order.

    Raises:
        FileNotFoundError: If no partitions are found under ``parquet_root``.
    """
    partitions = sorted(parquet_root.glob("year=*/part.parquet"))
    if not partitions:
        raise FileNotFoundError(f"no year=*/part.parquet partitions under {parquet_root}")
    return partitions


def _year_of(partition: Path) -> int:
    """Parse the year from a ``year=<YYYY>/part.parquet`` partition path."""
    return int(partition.parent.name.split("=", 1)[1])


def select_partitions(
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    years: list[int] | None = None,
) -> list[Path]:
    """Discover partitions, optionally narrowed to a set of years.

    Args:
        parquet_root: Directory holding ``year=<YYYY>/part.parquet`` partitions.
        years: If given, keep only these years; otherwise return all partitions.
            Requested years without a partition on disk are silently omitted
            (nothing to apply for them).

    Returns:
        Matching partition file paths, sorted by year.

    Raises:
        FileNotFoundError: If ``parquet_root`` holds no partitions at all.
    """
    partitions = discover_partitions(parquet_root)
    if years is None:
        return partitions
    wanted = {int(y) for y in years}
    return [p for p in partitions if _year_of(p) in wanted]


def partition_columns(partition: Path) -> list[str]:
    """Return the column names of a Parquet partition, in file order.

    These are the columns the loader copies; reading them from the data keeps the
    loader in sync with whatever ``ingest`` wrote.

    Args:
        partition: A ``part.parquet`` partition file.

    Returns:
        The column names in the order they appear in the file.
    """
    return pq.ParquetFile(partition).schema_arrow.names


def iter_rows(partition: Path, columns: list[str]) -> Iterator[tuple]:
    """Yield a partition's rows as tuples ordered to match ``columns``.

    Reads in batches to bound memory. Arrow decodes nulls to ``None`` and
    timestamps to naive ``datetime`` objects, which psycopg adapts directly - so
    no per-value cleaning is needed before ``COPY``.

    Args:
        partition: A ``part.parquet`` partition file.
        columns: Column names to emit, in the desired tuple order.

    Yields:
        One tuple per row, values ordered to match ``columns``.
    """
    parquet = pq.ParquetFile(partition)
    for batch in parquet.iter_batches(batch_size=READ_BATCH_SIZE, columns=columns):
        values = [batch.column(name).to_pylist() for name in columns]
        yield from zip(*values, strict=True)


def apply_schema(conn: psycopg.Connection, schema_path: Path = SCHEMA_PATH) -> None:
    """Execute ``schema.sql`` (extension, table, indexes); all statements are idempotent.

    Args:
        conn: An open connection.
        schema_path: Path to the schema DDL file.
    """
    conn.execute(schema_path.read_text())


def _drop_secondary_indexes(conn: psycopg.Connection) -> None:
    """Drop every index on ``incidents`` except the primary key.

    Names come from ``pg_indexes`` rather than being hardcoded, so the loader
    stays in sync with whatever indexes ``schema.sql`` defines.

    Args:
        conn: An open connection.
    """
    conn.execute(
        """
        DO $$
        DECLARE r record;
        BEGIN
            FOR r IN
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'incidents' AND indexname <> 'incidents_pkey'
            LOOP
                EXECUTE format('DROP INDEX %I', r.indexname);
            END LOOP;
        END $$;
        """
    )


def load(
    conn: psycopg.Connection,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
) -> dict:
    """Full-rebuild ``incidents`` from the Parquet dataset.

    Runs as a single transaction so a failure leaves the existing table intact:
    apply schema, truncate, drop secondary indexes, ``COPY`` every partition,
    recreate the indexes, and ``ANALYZE`` for fresh planner statistics.

    Args:
        conn: An open connection.
        parquet_root: Root of the Hive-partitioned Parquet dataset.

    Returns:
        A summary dict: ``partitions`` (count) and ``rows`` (total copied).
    """
    partitions = discover_partitions(parquet_root)
    columns = partition_columns(partitions[0])
    copy_stmt = sql.SQL("COPY incidents ({}) FROM STDIN").format(
        sql.SQL(", ").join(map(sql.Identifier, columns))
    )
    rows = 0

    with conn.transaction():
        apply_schema(conn)
        conn.execute("TRUNCATE incidents")
        _drop_secondary_indexes(conn)

        with conn.cursor() as cur, cur.copy(copy_stmt) as copy:
            for partition in partitions:
                before = rows
                for row in iter_rows(partition, columns):
                    copy.write_row(row)
                    rows += 1
                log.info("copied %d rows from %s", rows - before, partition)

        # Recreate the indexes dropped above (schema.sql is the single source).
        apply_schema(conn)
        conn.execute("ANALYZE incidents")

    log.info("loaded %d rows from %d partitions", rows, len(partitions))
    return {"partitions": len(partitions), "rows": rows}


STAGING_TABLE = "incidents_staging"


def upsert(
    conn: psycopg.Connection,
    parquet_root: Path = DEFAULT_PARQUET_ROOT,
    years: list[int] | None = None,
) -> dict:
    """Apply changed partitions to a live ``incidents`` table, keyed on ``id``.

    The nightly path. For each selected partition it COPYs the (already
    id-deduped) rows into a session-temp staging table, then
    ``INSERT ... SELECT ... ON CONFLICT (id) DO UPDATE`` merges them into
    ``incidents``. Existing rows are overwritten column-for-column and ``geom`` is
    recomputed from the incoming lat/long (it is a generated column); new rows are
    inserted. Runs in one transaction, so a failure leaves the table untouched.

    Because ``ingest`` rewrites a whole year partition per incremental run, the
    unit of work here is the whole partition (not a surgical row delta); pass
    ``years`` to restrict to the years that actually changed. Re-upserting an
    unchanged row is harmless.

    This never deletes: a row retracted at the source is not removed here -- the
    periodic full-refresh :func:`load` is the reconciliation sweep for deletions.

    Args:
        conn: An open connection.
        parquet_root: Root of the Hive-partitioned Parquet dataset.
        years: If given, upsert only these years' partitions; otherwise all.

    Returns:
        A summary dict: ``partitions`` (count applied) and ``rows`` (total upserted).
    """
    partitions = select_partitions(parquet_root, years)
    if not partitions:
        log.info("upsert: no matching partitions (years=%s); nothing to do", years)
        return {"partitions": 0, "rows": 0}

    columns = partition_columns(partitions[0])
    col_list = sql.SQL(", ").join(map(sql.Identifier, columns))
    staging = sql.Identifier(STAGING_TABLE)
    rows = 0

    with conn.transaction():
        apply_schema(conn)  # ensure table/indexes exist (idempotent); no-op when live

        # Staging holds exactly the payload columns (correct types, no generated
        # `geom`, no constraints) -- built from `incidents` so it never drifts.
        conn.execute(
            sql.SQL(
                "CREATE TEMP TABLE {staging} ON COMMIT DROP AS "
                "SELECT {cols} FROM incidents WITH NO DATA"
            ).format(staging=staging, cols=col_list)
        )

        copy_stmt = sql.SQL("COPY {staging} ({cols}) FROM STDIN").format(
            staging=staging, cols=col_list
        )
        with conn.cursor() as cur, cur.copy(copy_stmt) as copy:
            for partition in partitions:
                before = rows
                for row in iter_rows(partition, columns):
                    copy.write_row(row)
                    rows += 1
                log.info("staged %d rows from %s", rows - before, partition)

        # Overwrite every non-id column on conflict; `geom` is regenerated by the
        # table from the new lat/long and is not part of the payload.
        assignments = sql.SQL(", ").join(
            sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(c))
            for c in columns
            if c != "id"
        )
        conn.execute(
            sql.SQL(
                "INSERT INTO incidents ({cols}) SELECT {cols} FROM {staging} "
                "ON CONFLICT (id) DO UPDATE SET {assignments}"
            ).format(cols=col_list, staging=staging, assignments=assignments)
        )
        conn.execute("ANALYZE incidents")

    log.info("upserted %d rows from %d partitions", rows, len(partitions))
    return {"partitions": len(partitions), "rows": rows}


def main() -> None:
    """CLI entry point: load Parquet into the configured Postgres database.

    ``--mode refresh`` (default) full-rebuilds the table; ``--mode upsert`` merges
    changed partitions into a live table (optionally scoped with ``--years``).
    Reads connection settings from the environment via :class:`StoreConfig`.
    """
    import argparse

    from chicago_crime_mcp.store.config import StoreConfig

    parser = argparse.ArgumentParser(description="Load Parquet incidents into Postgres.")
    parser.add_argument(
        "--mode",
        choices=("refresh", "upsert"),
        default="refresh",
        help="refresh = full rebuild (default); upsert = merge changed partitions.",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=None,
        help="Root of the Hive-partitioned Parquet dataset (defaults to config).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="upsert only: restrict to these years (defaults to all partitions).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = StoreConfig.from_env()
    parquet_root = args.parquet_root or config.parquet_root

    with psycopg.connect(config.database_url) as conn:
        if args.mode == "upsert":
            summary = upsert(conn, parquet_root=parquet_root, years=args.years)
        else:
            summary = load(conn, parquet_root=parquet_root)
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
