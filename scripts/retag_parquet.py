"""One-off local migration: add the ``stable_category`` column to existing Parquet.

``stable_category`` is the comparable offense taxonomy -- the curated override
for codes the city has moved between primary types, falling back to
``primary_type_canonical`` for the ~99.8% of codes with no override. It was
introduced later than the partitions on disk: the store layer originally derived
it in a DuckDB view, which meant Postgres had no access to it and the rule lived
in two places. It is now derived once at ingest (see
``ingest/schema.add_stable_category``) and materialized into Parquet, so every
store reads the same column.

Newer ingest runs write it; partitions already on disk do not have it. This adds
it locally -- reading and writing local files only, no network and no re-pull.
Idempotent: partitions that already carry the column are recomputed, which is a
no-op unless the curation changed, and that is exactly how a curation change is
applied. The ``data/`` tree is gitignored, so this touches nothing tracked.

Each partition is written to a sibling temp file and moved into place with
``os.replace``, so an interrupted run leaves every partition readable rather
than truncated.

Run once from the repo root::

    python scripts/retag_parquet.py            # rewrite data/parquet
    python scripts/retag_parquet.py --dry-run  # report only

After running, reload the stores so they pick the column up::

    chicago-crime-load --mode refresh
    chicago-crime-rollup

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from chicago_crime_mcp.ingest import schema
from chicago_crime_mcp.ingest.backfill import PARQUET_DIR, ROW_GROUP_SIZE

log = logging.getLogger(__name__)


def retag_partition(
    path: Path, curated: dict[str, str], dry_run: bool = False
) -> tuple[int, int]:
    """Add or recompute ``stable_category`` for a single partition file.

    Args:
        path: Path to a ``part.parquet`` file.
        curated: An ``iucr -> stable_category`` map from
            :func:`~chicago_crime_mcp.ingest.schema.load_stable_category_map`.
        dry_run: If True, report what would change without writing.

    Returns:
        The partition's row count, and how many of those rows the curation
        actually moves to a different category than their canonical type.
    """
    df = pd.read_parquet(path)
    tagged = schema.add_stable_category(df, curated)
    moved = int((tagged["stable_category"] != tagged["primary_type_canonical"]).sum())

    verb = "would retag" if dry_run else "retagged"
    log.info("%s %s: %d rows, %d remapped", verb, path, len(tagged), moved)
    if dry_run:
        return len(tagged), moved

    # Write beside the target and swap, so an interrupt cannot leave a partial
    # file where a readable partition used to be.
    tmp = path.with_suffix(".parquet.tmp")
    tagged.to_parquet(tmp, index=False, row_group_size=ROW_GROUP_SIZE)
    os.replace(tmp, path)
    return len(tagged), moved


def main(argv: list[str] | None = None) -> None:
    """Add ``stable_category`` to every partition under the dataset root.

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

    curated = schema.load_stable_category_map()
    log.info("curated overrides: %d code(s) -> %s", len(curated), sorted(curated))

    totals = [retag_partition(p, curated, dry_run=args.dry_run) for p in paths]
    rows = sum(r for r, _ in totals)
    moved = sum(m for _, m in totals)
    share = moved / rows if rows else 0.0
    log.info(
        "done: %d partitions, %d rows, %d remapped (%.3f%%)",
        len(paths), rows, moved, share * 100,
    )


if __name__ == "__main__":
    main()
