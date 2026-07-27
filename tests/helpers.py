"""Shared test helpers for building Parquet fixtures.

The incident schema is defined once here rather than per test module: it mirrors
the 21 coerced columns ``ingest`` writes, and a second copy would drift the first
time a column is added.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Full incident schema, matching the columns/types ingest writes to Parquet.
SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("case_number", pa.string()),
        ("date", pa.timestamp("us")),
        ("block", pa.string()),
        ("iucr", pa.string()),
        ("primary_type", pa.string()),
        ("description", pa.string()),
        ("location_description", pa.string()),
        ("arrest", pa.bool_()),
        ("domestic", pa.bool_()),
        ("beat", pa.string()),
        ("district", pa.string()),
        ("ward", pa.int64()),
        ("community_area", pa.int64()),
        ("fbi_code", pa.string()),
        ("x_coordinate", pa.float64()),
        ("y_coordinate", pa.float64()),
        ("updated_on", pa.timestamp("us")),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("primary_type_canonical", pa.string()),
    ]
)


def row(**overrides) -> dict:
    """A complete incident row with sane defaults; override only what matters.

    Args:
        **overrides: Column values to replace in the default row.

    Returns:
        A dict covering every column in :data:`SCHEMA`.
    """
    base = dict(
        id=1,
        case_number="JF100001",
        date=datetime(2025, 1, 1, 3, 0),
        block="001XX N STATE ST",
        iucr="0486",
        primary_type="BATTERY",
        description="DOMESTIC BATTERY SIMPLE",
        location_description="APARTMENT",
        arrest=False,
        domestic=True,
        beat="1011",
        district="010",
        ward=1,
        community_area=29,
        fbi_code="08B",
        x_coordinate=1150000.0,
        y_coordinate=1900000.0,
        updated_on=datetime(2025, 1, 2, 0, 0),
        latitude=41.8781,
        longitude=-87.6298,
        primary_type_canonical="BATTERY",
    )
    base.update(overrides)
    return base


def write_partition(base: Path, year: int, rows: list[dict]) -> Path:
    """Write ``rows`` to ``base/year=<year>/part.parquet`` and return the path.

    Args:
        base: Root of the Hive-partitioned dataset.
        year: Partition year.
        rows: Incident dicts, e.g. from :func:`row`.

    Returns:
        The written partition path.
    """
    path = base / f"year={year}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)
    return path
