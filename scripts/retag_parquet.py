"""Local migration: bring existing Parquet partitions up to the current schema.

Derived columns get added over time, and partitions already on disk do not have
them. Rather than re-pulling 2.9M rows from the city to change a column we
compute ourselves, this recomputes every derived column in place, from local
files only -- no network, no re-pull.

It runs :func:`~chicago_crime_mcp.ingest.schema.prepare`, the same pipeline both
ingest paths run, so a migrated partition is byte-for-byte what a fresh backfill
would have written. That is the point of using it rather than calling the one
``add_*`` function a given migration happens to need: the script cannot fall
behind the pipeline, because it *is* the pipeline.

Which means it is idempotent by construction. Re-running changes nothing unless
a derivation changed -- and when one has, applying it is exactly what this is
for. The dry run reports per-column change counts, so "nothing to do" and "3,412
rows move" are distinguishable before anything is written.

Two migrations have used it so far:

* ``stable_category`` -- the comparable offense taxonomy, once it moved out of a
  DuckDB view and into ingest so that Postgres could see it too.
* ``neighborhood`` -- the point-in-polygon tag against the city's 98 published
  boundaries, which the incident feed does not carry.

Each partition is written to a sibling temp file and moved into place with
``os.replace``, so an interrupted run leaves every partition readable rather than
truncated. The ``data/`` tree is gitignored, so this touches nothing tracked.

Run from the repo root::

    python scripts/retag_parquet.py --dry-run  # report only
    python scripts/retag_parquet.py            # rewrite data/parquet

Needs DuckDB's spatial extension, which downloads on first use and then caches to
``~/.duckdb/extensions``.

After running, reload the stores so they pick the columns up::

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

from chicago_crime_mcp.geo.boundaries import NeighborhoodBoundaries
from chicago_crime_mcp.ingest import schema
from chicago_crime_mcp.ingest.backfill import PARQUET_DIR, ROW_GROUP_SIZE

log = logging.getLogger(__name__)

#: The columns ingest derives, and therefore the columns this can repair. Kept
#: explicit so the change report names them; the values themselves come from
#: `schema.prepare`, never from a second implementation here.
DERIVED_COLUMNS = ("primary_type_canonical", "stable_category", "neighborhood")


def _changed(before: pd.DataFrame, after: pd.DataFrame, column: str) -> int:
    """Count rows whose derived value the migration would alter.

    Treats null-to-null as unchanged, so the ~1.8% of rows that legitimately
    have no neighborhood do not read as churn on every re-run.

    Args:
        before: The partition as read from disk.
        after: The same rows after :func:`schema.prepare`.
        column: The derived column to compare.

    Returns:
        The number of differing rows. A column absent from ``before`` counts as
        every row changed, which is what adding a column is.
    """
    if column not in before.columns:
        return len(after)
    old, new = before[column], after[column]
    both_null = old.isna() & new.isna()
    # `.fillna(False)` is load-bearing: comparing a value against a null yields
    # NA, not False, and pandas drops NA from the sum -- so a row that *lost* its
    # tag would go uncounted, which is the one direction most worth reporting.
    return int((~(old.eq(new).fillna(False) | both_null)).sum())


def retag_partition(
    path: Path,
    reference: dict[str, str],
    curated: dict[str, str],
    boundaries: schema.PointLocator,
    dry_run: bool = False,
) -> dict:
    """Recompute every derived column for a single partition file.

    Args:
        path: Path to a ``part.parquet`` file.
        reference: An ``iucr -> primary_description`` map from
            :func:`~chicago_crime_mcp.ingest.schema.load_iucr_reference`.
        curated: An ``iucr -> stable_category`` map from
            :func:`~chicago_crime_mcp.ingest.schema.load_stable_category_map`.
        boundaries: The neighborhood polygons to tag each incident against.
        dry_run: If True, report what would change without writing.

    Returns:
        A summary dict: ``rows``, ``located`` (rows that fell inside a
        neighborhood polygon), and ``changed`` (a per-column count of rows this
        run would alter).
    """
    df = pd.read_parquet(path)
    tagged = schema.prepare(df, reference, curated, boundaries)
    summary = {
        "rows": len(tagged),
        "located": int(tagged["neighborhood"].notna().sum()),
        "changed": {c: _changed(df, tagged, c) for c in DERIVED_COLUMNS},
    }

    changes = ", ".join(f"{c}={n}" for c, n in summary["changed"].items() if n)
    log.info(
        "%s %s: %d rows, %d located%s",
        "would rewrite" if dry_run else "rewrote",
        path,
        summary["rows"],
        summary["located"],
        f", changing {changes}" if changes else ", nothing to change",
    )
    if dry_run:
        return summary

    # Write beside the target and swap, so an interrupt cannot leave a partial
    # file where a readable partition used to be.
    tmp = path.with_suffix(".parquet.tmp")
    tagged.to_parquet(tmp, index=False, row_group_size=ROW_GROUP_SIZE)
    os.replace(tmp, path)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Recompute the derived columns on every partition under the dataset root.

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

    reference = schema.load_iucr_reference()
    curated = schema.load_stable_category_map()
    log.info("curated overrides: %d code(s) -> %s", len(curated), sorted(curated))

    with NeighborhoodBoundaries.load() as boundaries:
        log.info("neighborhood polygons: %d", len(boundaries.names))
        summaries = [
            retag_partition(p, reference, curated, boundaries, dry_run=args.dry_run)
            for p in paths
        ]

    rows = sum(s["rows"] for s in summaries)
    located = sum(s["located"] for s in summaries)
    changed = {
        c: sum(s["changed"][c] for s in summaries) for c in DERIVED_COLUMNS
    }
    log.info(
        "done: %d partitions, %d rows, %d located (%.2f%%)",
        len(paths), rows, located, (located / rows * 100) if rows else 0.0,
    )
    for column, n in changed.items():
        log.info("  %-24s %d rows changed (%.3f%%)", column, n, (n / rows * 100) if rows else 0.0)


if __name__ == "__main__":
    main()
