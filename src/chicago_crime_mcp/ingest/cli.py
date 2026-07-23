"""Command-line entry point for crime data ingestion.

Examples:
    # Backfill the working window (defaults shown)
    python -m chicago_crime_mcp.ingest backfill --start-year 2020 --end-year 2025

    # Re-pull a year even if its partition looks complete
    python -m chicago_crime_mcp.ingest backfill --start-year 2015 --end-year 2015 --force

    # Apply rows updated since the watermark
    python -m chicago_crime_mcp.ingest incremental --overlap-days 1

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta

from dotenv import load_dotenv

from chicago_crime_mcp.ingest import backfill as backfill_mod
from chicago_crime_mcp.ingest import incremental as incremental_mod
from chicago_crime_mcp.ingest.socrata import SodaClient

log = logging.getLogger(__name__)

DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2025


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ingestion CLI.

    Returns:
        A parser with ``backfill`` and ``incremental`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="python -m chicago_crime_mcp.ingest",
        description="Ingest Chicago crime data into partitioned Parquet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("backfill", help="Historical backfill to partitioned Parquet.")
    b.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR,
                   help=f"First year, inclusive (default {DEFAULT_START_YEAR}).")
    b.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR,
                   help=f"Last year, inclusive (default {DEFAULT_END_YEAR}).")
    b.add_argument("--force", action="store_true",
                   help="Re-pull years even if a complete partition exists.")

    i = sub.add_parser("incremental", help="Sync rows updated since the watermark.")
    i.add_argument("--overlap-days", type=int, default=1,
                   help="Re-pull this many days before the watermark (default 1).")

    return parser


def _configure_logging() -> None:
    """Send progress logs to stderr at INFO, quieting per-request httpx logs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _run_backfill(client: SodaClient, args: argparse.Namespace) -> None:
    """Run a backfill and print a per-year summary."""
    results = backfill_mod.backfill(
        client, args.start_year, args.end_year, force=args.force
    )
    total = sum(r["rows"] for r in results)
    log.info(
        "Backfill %d-%d: %s rows across %d years",
        args.start_year, args.end_year, f"{total:,}", len(results),
    )
    for r in results:
        tag = "skipped (complete)" if r["skipped"] else "written"
        log.info("  %d: %s rows (%s)", r["year"], f"{r['rows']:,}", tag)


def _run_incremental(client: SodaClient, args: argparse.Namespace) -> None:
    """Run an incremental sync and print a summary."""
    summary = incremental_mod.incremental_sync(
        client, overlap=timedelta(days=args.overlap_days)
    )
    log.info(
        "Incremental sync: pulled %s rows updated since %s",
        f"{summary['pulled']:,}", summary["cutoff"],
    )
    for year, s in sorted(summary["years"].items()):
        log.info("  %d: applied %s -> %s rows",
                 year, f"{s['applied']:,}", f"{s['partition_rows']:,}")
    if summary["skipped_unmanaged"]:
        log.info("  skipped %s rows for unmanaged years", f"{summary['skipped_unmanaged']:,}")
    log.info("  watermark now %s", summary["watermark"])


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the requested ingestion command.

    Args:
        argv: Argument list (defaults to ``sys.argv``). Used by tests.

    Raises:
        SodaError: Propagated from the underlying ingestion routines.
    """
    args = build_parser().parse_args(argv)
    load_dotenv()
    _configure_logging()

    with SodaClient() as client:
        if args.command == "backfill":
            _run_backfill(client, args)
        elif args.command == "incremental":
            _run_incremental(client, args)


if __name__ == "__main__":
    main()
