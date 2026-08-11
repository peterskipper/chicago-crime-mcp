"""Incremental sync of the crime dataset, keyed on the ``updated_on`` watermark.

Each run pulls rows with ``updated_on`` newer than the stored watermark minus a
small overlap (to catch late-arriving edits and boundary races), then merges them
into the affected year partitions: existing partition + new rows, deduplicated on
``id`` keeping the latest ``updated_on``, rewritten wholesale. Because a year
partition is only ~12 MB, a full rewrite is cheaper than appending delta files and
avoids any compaction machinery.

The pull is keyset-paginated by ``id`` (unique) with ``updated_on`` as a static
predicate. Only years that already have a partition are updated; updates to
unmanaged years are counted and skipped (they get picked up when that year is
backfilled). The watermark lives in a small JSON state file.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from chicago_crime_mcp.ingest import backfill, schema
from chicago_crime_mcp.ingest.socrata import SodaClient

log = logging.getLogger(__name__)

STATE_PATH = backfill.DATA_DIR / "ingest_state.json"
DEFAULT_OVERLAP = timedelta(days=1)


def load_state(path: Path = STATE_PATH) -> dict:
    """Load the ingest state file.

    Args:
        path: Path to the JSON state file.

    Returns:
        The parsed state dict, or an empty dict if the file does not exist.
    """
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    """Write the ingest state file.

    Args:
        state: State dict to persist (watermark, last run, counts).
        path: Path to the JSON state file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def managed_years(base: Path = backfill.PARQUET_DIR) -> set[int]:
    """Return the set of years that already have a Parquet partition.

    Args:
        base: Root directory of the partitioned dataset.

    Returns:
        A set of four-digit years discovered from ``year=<YYYY>`` directories.
    """
    if not base.exists():
        return set()
    years = set()
    for d in base.glob("year=*"):
        try:
            years.add(int(d.name.split("=", 1)[1]))
        except ValueError:
            continue
    return years


def derive_watermark(base: Path = backfill.PARQUET_DIR) -> pd.Timestamp | None:
    """Compute a bootstrap watermark from existing partitions.

    Used on the first incremental run, when no state file exists yet.

    Args:
        base: Root directory of the partitioned dataset.

    Returns:
        The maximum ``updated_on`` across all partitions, or ``None`` if there
        are no partitions.
    """
    mx: pd.Timestamp | None = None
    if not base.exists():
        return None
    for d in sorted(base.glob("year=*")):
        p = d / "part.parquet"
        if not p.exists():
            continue
        col = pd.read_parquet(p, columns=["updated_on"])["updated_on"]
        m = col.max()
        if pd.notna(m) and (mx is None or m > mx):
            mx = m
    return mx


def _merge_year(base: Path, year: int, new_rows: pd.DataFrame) -> int:
    """Merge new rows into a year partition, deduping on ``id``.

    Concatenates the existing partition (if any) with ``new_rows``, keeps the
    row with the latest ``updated_on`` per ``id``, sorts by ``id``, and rewrites
    the partition.

    Args:
        base: Root directory of the partitioned dataset.
        year: Partition year to merge into.
        new_rows: Coerced, canonicalized rows belonging to ``year``.

    Returns:
        The number of rows in the rewritten partition.
    """
    path = backfill.partition_path(year, base)
    if path.exists():
        combined = pd.concat([pd.read_parquet(path), new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined = (
        combined.sort_values("updated_on")
        .drop_duplicates("id", keep="last")
        .sort_values("id")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False, row_group_size=backfill.ROW_GROUP_SIZE)
    return len(combined)


def incremental_sync(
    client: SodaClient,
    base: Path = backfill.PARQUET_DIR,
    state_path: Path = STATE_PATH,
    overlap: timedelta = DEFAULT_OVERLAP,
) -> dict:
    """Pull rows updated since the watermark and merge them into partitions.

    Args:
        client: An open :class:`SodaClient`.
        base: Root directory of the partitioned dataset.
        state_path: Path to the JSON state file holding the watermark.
        overlap: How far before the watermark to re-pull, to catch late edits.

    Returns:
        A summary dict: ``pulled`` (rows fetched), ``cutoff`` (the ``updated_on``
        lower bound used), ``years`` (per-year applied/partition_rows),
        ``skipped_unmanaged`` (rows for years without a partition), and the new
        ``watermark``.

    Raises:
        RuntimeError: If there is no stored watermark and no existing partitions
            to derive one from (run a backfill first).
        SodaError: Propagated from the underlying requests.
    """
    state = load_state(state_path)
    watermark = (
        pd.Timestamp(state["watermark"]) if state.get("watermark") else derive_watermark(base)
    )
    if watermark is None:
        raise RuntimeError(
            "No watermark and no existing partitions - run a backfill before syncing."
        )

    cutoff = watermark - overlap
    where = f"updated_on > '{cutoff.strftime('%Y-%m-%dT%H:%M:%S')}'"
    log.info("incremental sync: pulling rows with %s", where)

    frames = [
        pd.DataFrame(page)
        for page in client.paginate_keyset(
            schema.CRIME_DATASET_ID, select=",".join(schema.SELECT_FIELDS), where=where
        )
    ]
    pulled = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=schema.SELECT_FIELDS)
    )

    summary: dict = {
        "pulled": len(pulled),
        "cutoff": cutoff.isoformat(),
        "years": {},
        "skipped_unmanaged": 0,
    }

    if len(pulled):
        reference = schema.load_iucr_reference()
        pulled = schema.add_canonical_primary_type(pulled, reference)
        pulled = schema.add_stable_category(pulled, schema.load_stable_category_map())
        pulled = schema.coerce_types(pulled)
        managed = managed_years(base)
        pyear = pulled["date"].dt.year

        for year, grp in pulled.groupby(pyear):
            year = int(year)
            if year not in managed:
                summary["skipped_unmanaged"] += len(grp)
                log.info("year %d unmanaged: %d updated rows skipped", year, len(grp))
                continue
            rows = _merge_year(base, year, grp)
            summary["years"][year] = {"applied": len(grp), "partition_rows": rows}
            log.info("year %d: applied %d updates -> %d rows", year, len(grp), rows)

        new_wm = pulled["updated_on"].max()
        if pd.notna(new_wm) and new_wm > watermark:
            watermark = new_wm

    save_state(
        {
            "watermark": watermark.isoformat(),
            "last_run": datetime.now(UTC).isoformat(),
            "last_pulled": int(len(pulled)),
        },
        state_path,
    )
    summary["watermark"] = watermark.isoformat()
    return summary
