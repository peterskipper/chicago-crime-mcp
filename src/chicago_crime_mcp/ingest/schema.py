"""Crime dataset schema, type coercion, and IUCR-based canonicalization.

Two Socrata datasets are involved:

* ``ijzp-q8t2`` - Crimes 2001 to Present (the incidents).
* ``c7ck-438e`` - the authoritative IUCR code reference (offense classification).

``primary_type`` in the incident feed is a human label that has drifted over
time (e.g. ``CRIM SEXUAL ASSAULT`` was relabeled ``CRIMINAL SEXUAL ASSAULT``).
The IUCR code is stable, so we derive a ``primary_type_canonical`` column purely
from the IUCR reference and keep the raw ``primary_type`` for provenance.

On-disk dtypes deliberately avoid pandas ``category``: category columns carry a
per-file dictionary, which would make Hive-partitioned Parquet files
schema-incompatible across years. Enum-like columns are stored as nullable
strings instead.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chicago_crime_mcp.ingest.socrata import SodaClient

CRIME_DATASET_ID = "ijzp-q8t2"
IUCR_DATASET_ID = "c7ck-438e"

# Pinned snapshot of the IUCR reference table, committed for reproducible,
# offline canonicalization. Refresh with ``refresh_iucr_snapshot``.
IUCR_REFERENCE_PATH = Path(__file__).parent / "data" / "iucr_codes.csv"

# Explicit column list for the incident pull. Excludes the nested ``location``
# object (redundant with latitude/longitude) and keeps payloads small/stable.
SELECT_FIELDS: list[str] = [
    "id",
    "case_number",
    "date",
    "block",
    "iucr",
    "primary_type",
    "description",
    "location_description",
    "arrest",
    "domestic",
    "beat",
    "district",
    "ward",
    "community_area",
    "fbi_code",
    "x_coordinate",
    "y_coordinate",
    # NOTE: the API also exposes a `year` field, but it is deliberately NOT
    # pulled. It is redundant with the Hive partition (`year=<YYYY>`), which is
    # derived from `date`; keeping an in-file `year` column would collide with
    # the partition value DuckDB/Postgres synthesize on read (duplicate column).
    "updated_on",
    "latitude",
    "longitude",
]

# On-disk dtype groups (see module docstring on why enums are strings, not category).
_STRING_COLS = [
    "case_number", "block", "iucr", "primary_type", "primary_type_canonical",
    "description", "location_description", "fbi_code",
    "beat", "district",  # kept as strings: zero-padded codes ("0111") used for joins
]
_DATETIME_COLS = ["date", "updated_on"]
_FLOAT_COLS = ["latitude", "longitude", "x_coordinate", "y_coordinate"]
# `year` is intentionally absent: it comes from the Hive partition, not the payload.
_INT_COLS = ["id", "ward", "community_area"]
_BOOL_COLS = ["arrest", "domestic"]
# SODA returns booleans as JSON true/false, but older exports use the strings
# "true"/"false" - map both, since "false" is otherwise truthy.
_BOOL_MAP = {True: True, False: False, "true": True, "false": False}


def normalize_iucr(code: object) -> str | None:
    """Normalize an IUCR code to its canonical 4-character, zero-padded form.

    Args:
        code: A raw IUCR value (str, number, ``None``, or NaN).

    Returns:
        The 4-character zero-padded code (e.g. ``"281"`` -> ``"0281"``), or
        ``None`` if the input is null/blank.
    """
    if pd.isna(code):  # handles None, float NaN, and pandas NA
        return None
    s = str(code).strip()
    return s.zfill(4) if s else None


def fetch_iucr_reference(client: SodaClient) -> pd.DataFrame:
    """Fetch the authoritative IUCR reference table from Socrata.

    Args:
        client: An open :class:`SodaClient`.

    Returns:
        A DataFrame with columns ``iucr`` (normalized), ``primary_description``,
        ``secondary_description``, ``index_code``, ``active``, sorted by ``iucr``.

    Raises:
        SodaError: Propagated from the underlying request.
    """
    rows = client.get(IUCR_DATASET_ID, {"$limit": 5000})
    df = pd.DataFrame(rows)
    df["iucr"] = df["iucr"].astype("string").map(normalize_iucr)
    keep = ["iucr", "primary_description", "secondary_description", "index_code", "active"]
    df = df[[c for c in keep if c in df.columns]]
    return df.sort_values("iucr").reset_index(drop=True)


def refresh_iucr_snapshot(
    client: SodaClient, path: Path = IUCR_REFERENCE_PATH
) -> dict[str, list[str]]:
    """Refresh the pinned IUCR snapshot from Socrata and report what changed.

    Fetches the current reference table, diffs it against the existing snapshot
    (if any), then overwrites the snapshot on disk.

    Args:
        client: An open :class:`SodaClient`.
        path: Where the snapshot CSV lives.

    Returns:
        A diff summary with keys ``added``, ``removed``, and ``relabeled`` (IUCR
        codes whose ``primary_description`` changed) - all lists of IUCR codes.

    Raises:
        SodaError: Propagated from the underlying request.
    """
    new = fetch_iucr_reference(client)
    added: list[str] = []
    removed: list[str] = []
    relabeled: list[str] = []
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        old_map = dict(zip(old["iucr"].str.zfill(4), old["primary_description"], strict=False))
        new_map = dict(zip(new["iucr"], new["primary_description"], strict=False))
        added = sorted(set(new_map) - set(old_map))
        removed = sorted(set(old_map) - set(new_map))
        relabeled = sorted(
            c for c in set(old_map) & set(new_map) if old_map[c] != new_map[c]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return {"added": added, "removed": removed, "relabeled": relabeled}


def load_iucr_reference(path: Path = IUCR_REFERENCE_PATH) -> dict[str, str]:
    """Load the pinned IUCR snapshot as an ``iucr -> primary_description`` map.

    Args:
        path: Path to the snapshot CSV.

    Returns:
        A dict mapping normalized 4-char IUCR codes to their official primary
        description.

    Raises:
        FileNotFoundError: If the snapshot has not been created yet.
    """
    df = pd.read_csv(path, dtype=str)
    df["iucr"] = df["iucr"].map(normalize_iucr)
    return dict(zip(df["iucr"], df["primary_description"], strict=False))


def add_canonical_primary_type(
    df: pd.DataFrame, reference: dict[str, str]
) -> pd.DataFrame:
    """Add a ``primary_type_canonical`` column derived from the IUCR code.

    The canonical value comes solely from the IUCR reference lookup, so drifted
    labels for the same code collapse to one value. Rows whose ``iucr`` is null
    or absent from the reference fall back to the raw ``primary_type``.

    Args:
        df: Incident rows including ``iucr`` and ``primary_type`` columns.
        reference: An ``iucr -> primary_description`` map from
            :func:`load_iucr_reference`.

    Returns:
        A copy of ``df`` with the added ``primary_type_canonical`` column.
    """
    iucr = df["iucr"].astype("string").map(normalize_iucr)
    canonical = iucr.map(reference)
    out = df.copy()
    out["primary_type_canonical"] = canonical.fillna(df["primary_type"]).astype("string")
    return out


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a raw SODA DataFrame to stable, partition-safe on-disk dtypes.

    SODA returns every value as a string. This casts each column group to a
    canonical nullable dtype (strings for enum-like/zero-padded codes, datetimes,
    floats, nullable ints, and booleans) so that Parquet partitions written
    across different years share an identical schema.

    Args:
        df: Raw incident rows (all-string values from SODA), optionally already
            carrying ``primary_type_canonical``.

    Returns:
        A copy of ``df`` with coerced dtypes. Columns absent from ``df`` are
        skipped.
    """
    out = df.copy()
    for col in _DATETIME_COLS:
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in _FLOAT_COLS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in _INT_COLS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in _BOOL_COLS:
        if col in out:
            out[col] = out[col].map(_BOOL_MAP).astype("boolean")
    for col in _STRING_COLS:
        if col in out:
            out[col] = out[col].astype("string")
    return out
