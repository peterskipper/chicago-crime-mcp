"""Historical backfill of the crime dataset to Hive-partitioned Parquet.

The dataset is pulled one year at a time. Each year is a bounded ``$where`` on
``date`` and is keyset-paginated by ``id``; the result is coerced, canonicalized,
and written to ``<base>/year=<YYYY>/part.parquet`` (a Hive layout DuckDB and
Postgres can read with partition pruning).

Checkpointing is at the year granularity: a re-run skips any year whose partition
already holds exactly the number of rows the API reports for it, so an
interrupted backfill resumes cheaply (each year is only a handful of 50k pages).

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from chicago_crime_mcp.ingest import schema
from chicago_crime_mcp.ingest.socrata import DEFAULT_PAGE_SIZE, SodaClient

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "parquet"

# Row groups are Parquet's unit of predicate pushdown (min/max skipping) and of
# DuckDB read parallelism. pyarrow writes one row group for a whole year (~265k
# rows) by default, which means a single scan thread and coarse skipping. Capping
# at ~100k rows yields a few row groups per year, so DuckDB can parallelize within
# a year's file and skip finer chunks - at no extra file count or storage cost.
ROW_GROUP_SIZE = 100_000


def year_where(year: int) -> str:
    """Build the ``$where`` predicate bounding a single calendar year.

    Args:
        year: Four-digit year.

    Returns:
        A SoQL predicate selecting ``date`` within ``[year, year+1)``.
    """
    return f"date >= '{year}-01-01T00:00:00' AND date < '{year + 1}-01-01T00:00:00'"


def partition_path(year: int, base: Path = PARQUET_DIR) -> Path:
    """Return the Parquet partition path for a year.

    Args:
        year: Four-digit year.
        base: Root directory of the partitioned dataset.

    Returns:
        ``<base>/year=<year>/part.parquet``.
    """
    return base / f"year={year}" / "part.parquet"


def _prepare(df: pd.DataFrame, reference: dict[str, str]) -> pd.DataFrame:
    """Canonicalize then coerce a raw page-concatenated frame for storage."""
    df = schema.add_canonical_primary_type(df, reference)
    return schema.coerce_types(df)


def backfill_year(
    client: SodaClient,
    year: int,
    reference: dict[str, str],
    base: Path = PARQUET_DIR,
    force: bool = False,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict:
    """Pull one year of incidents and write its Parquet partition.

    Args:
        client: An open :class:`SodaClient`.
        year: Four-digit year to pull.
        reference: An ``iucr -> primary_description`` map for canonicalization.
        base: Root directory of the partitioned dataset.
        force: Re-pull even if a complete partition already exists.
        page_size: Rows per keyset page.

    Returns:
        A summary dict: ``year``, ``rows`` written/present, ``expected`` (API
        count), and ``skipped`` (whether an existing complete partition was kept).

    Raises:
        SodaError: Propagated from the underlying requests.
    """
    where = year_where(year)
    expected = client.count(schema.CRIME_DATASET_ID, where)
    out_path = partition_path(year, base)

    if out_path.exists() and not force:
        existing = len(pd.read_parquet(out_path, columns=["id"]))
        if existing == expected:
            log.info("year %d: %d rows already complete, skipping", year, expected)
            return {"year": year, "rows": existing, "expected": expected, "skipped": True}
        log.warning(
            "year %d: partition has %d rows but API reports %d - re-pulling",
            year, existing, expected,
        )

    frames: list[pd.DataFrame] = []
    for page in client.paginate_keyset(
        schema.CRIME_DATASET_ID,
        select=",".join(schema.SELECT_FIELDS),
        where=where,
        page_size=page_size,
    ):
        frames.append(pd.DataFrame(page))
        log.info("year %d: %d/%d rows", year, sum(len(f) for f in frames), expected)

    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=schema.SELECT_FIELDS)
    )
    df = _prepare(df, reference)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, row_group_size=ROW_GROUP_SIZE)
    log.info("year %d: wrote %d rows -> %s", year, len(df), out_path)
    return {"year": year, "rows": len(df), "expected": expected, "skipped": False}


def backfill(
    client: SodaClient,
    start_year: int,
    end_year: int,
    base: Path = PARQUET_DIR,
    force: bool = False,
) -> list[dict]:
    """Backfill an inclusive range of years to partitioned Parquet.

    Loads the pinned IUCR reference once, then backfills each year in order.
    Safe to re-run: complete years are skipped.

    Args:
        client: An open :class:`SodaClient`.
        start_year: First year to pull (inclusive).
        end_year: Last year to pull (inclusive).
        base: Root directory of the partitioned dataset.
        force: Re-pull every year even if complete partitions exist.

    Returns:
        A list of per-year summary dicts (see :func:`backfill_year`).

    Raises:
        SodaError: Propagated from the underlying requests.
        FileNotFoundError: If the IUCR snapshot has not been created.
    """
    reference = schema.load_iucr_reference()
    results = []
    for year in range(start_year, end_year + 1):
        results.append(backfill_year(client, year, reference, base=base, force=force))
    total = sum(r["rows"] for r in results)
    log.info("backfill %d-%d complete: %d rows across %d years",
             start_year, end_year, total, len(results))
    return results
