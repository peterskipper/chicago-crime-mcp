"""One-off local migration: drop the redundant ``year`` column from existing Parquet.

Phase 1 wrote a ``year`` column into every partition, pulled straight from the
Socrata feed. It is redundant with the Hive partition (``year=<YYYY>``, derived
from ``date``) and would collide with the partition value DuckDB/Postgres
synthesize on read. Newer ingest runs no longer write it (see
``ingest/schema.py``), but partitions already on disk still carry it.

This rewrites each ``<base>/year=*/part.parquet`` in place with the ``year``
column removed. It reads and writes local files only - no network, no re-pull.
Idempotent: partitions that already lack ``year`` are left untouched. The
``data/`` tree is gitignored, so this touches nothing tracked.

Run once from the repo root::

    python scripts/drop_year_column.py            # rewrite data/parquet
    python scripts/drop_year_column.py --dry-run  # report only

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from chicago_crime_mcp.ingest.backfill import PARQUET_DIR, ROW_GROUP_SIZE

log = logging.getLogger(__name__)


def rewrite_partition(path: Path, dry_run: bool = False) -> bool:
    """Drop the ``year`` column from a single partition file if present.

    Args:
        path: Path to a ``part.parquet`` file.
        dry_run: If True, report what would change without writing.

    Returns:
        True if the file carried a ``year`` column (i.e. was or would be
        rewritten), False if it was already clean.
    """
    df = pd.read_parquet(path)
    if "year" not in df.columns:
        log.info("clean, skipping: %s", path)
        return False
    if dry_run:
        log.info("would rewrite (drop `year`): %s", path)
        return True
    df.drop(columns=["year"]).to_parquet(path, index=False, row_group_size=ROW_GROUP_SIZE)
    log.info("rewrote (dropped `year`): %s", path)
    return True


def main(argv: list[str] | None = None) -> None:
    """Rewrite every partition under the dataset root, dropping ``year``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=PARQUET_DIR, help="Partitioned dataset root."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report changes without writing."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = sorted(args.base.glob("year=*/part.parquet"))
    if not paths:
        log.warning("no partitions found under %s", args.base)
        return
    changed = sum(rewrite_partition(p, dry_run=args.dry_run) for p in paths)
    verb = "would change" if args.dry_run else "changed"
    log.info("done: %d/%d partitions %s", changed, len(paths), verb)


if __name__ == "__main__":
    main()
