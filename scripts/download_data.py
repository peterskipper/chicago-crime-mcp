#!/usr/bin/env python3
"""
Fetch Chicago crime incidents from the Socrata (SODA) API.

Dataset: Crimes - 2001 to Present  (id: ijzp-q8t2)
Docs:    https://dev.socrata.com/foundry/data.cityofchicago.org/ijzp-q8t2

Usage
-----
  # 200-row peek, pretty-printed field inventory, nothing written
  python scripts/download_data.py peek

  # full year -> data/crimes_2025.parquet (paginated, checkpointed, resumable)
  python scripts/download_data.py year 2025

  # inspect what you downloaded
  python scripts/download_data.py inspect data/crimes_2025.parquet

An app token is optional but strongly recommended (shared throttle otherwise).
Register at https://data.cityofchicago.org/profile/edit/developer_settings
then:  export SOCRATA_APP_TOKEN=xxxxxxxx

Deps: pip install httpx pandas pyarrow
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

try:  # optional: load SOCRATA_APP_TOKEN from a local .env if python-dotenv is present
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

DOMAIN = "data.cityofchicago.org"
DATASET_ID = "ijzp-q8t2"
BASE_URL = f"https://{DOMAIN}/resource/{DATASET_ID}.json"

PAGE_SIZE = 50_000          # Socrata allows up to 50k per request
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"  # per-page JSON, so a failed run resumes cheaply

# Explicit column list. Socrata returns a nested `location` object and a few
# legacy columns we don't need; naming fields keeps payloads smaller and the
# schema stable if the portal adds columns later.
SELECT_FIELDS = [
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
    "year",
    "updated_on",
    "latitude",
    "longitude",
]


def client() -> httpx.Client:
    headers = {"Accept": "application/json"}
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token
    else:
        print(
            "  ! No SOCRATA_APP_TOKEN set - using the shared anonymous throttle.\n"
            "    Fine for a peek, expect 429s on a full-year pull.",
            file=sys.stderr,
        )
    return httpx.Client(headers=headers, timeout=60.0)


def get_page(c: httpx.Client, params: dict, attempt: int = 0) -> list[dict]:
    """One SODA request with naive exponential backoff on 429/5xx."""
    r = c.get(BASE_URL, params=params)
    if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
        wait = 2 ** attempt
        print(f"    {r.status_code} - retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
        return get_page(c, params, attempt + 1)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- peek


def peek(n: int = 200) -> None:
    """Pull a small sample and report the actual field shapes."""
    with client() as c:
        rows = get_page(
            c,
            {
                "$limit": n,
                "$order": "date DESC",
            },
        )

    print(f"\nPulled {len(rows)} rows (most recent first).\n")

    print("=" * 70)
    print("FIRST RECORD, VERBATIM")
    print("=" * 70)
    print(json.dumps(rows[0], indent=2))

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("FIELD INVENTORY")
    print("=" * 70)
    for col in sorted(df.columns):
        non_null = df[col].notna().sum()
        pct = 100 * non_null / len(df)
        sample = df[col].dropna().iloc[0] if non_null else "-"
        sample = str(sample)[:45]
        print(f"  {col:<22} {pct:5.1f}% populated   e.g. {sample}")

    print("\n" + "=" * 70)
    print("CARDINALITY OF LIKELY ENUM COLUMNS")
    print("=" * 70)
    for col in ("primary_type", "location_description", "fbi_code", "district"):
        if col in df.columns:
            print(f"\n  {col} - {df[col].nunique()} distinct in sample")
            print(df[col].value_counts().head(8).to_string())

    print("\n" + "=" * 70)
    print("NOTES FOR SCHEMA DESIGN")
    print("=" * 70)
    print(
        "  - Everything arrives as a JSON string, including numerics and bools.\n"
        "    Cast deliberately on load; don't trust pandas inference.\n"
        "  - lat/long are null for a meaningful slice of rows (redacted or\n"
        "    ungeocoded). Decide now whether spatial tools drop or flag these.\n"
        "  - `id` is the stable surrogate key; `case_number` is the CPD RD\n"
        "    number and is NOT unique (multi-victim/multi-offense incidents).\n"
        "  - `updated_on` is your incremental-sync watermark.\n"
        "  - `primary_type` is the natural enum for a constrained tool input;\n"
        "    pull the full distinct list from the API, not from this sample."
    )


# ---------------------------------------------------------------- full pull


def fetch_year(year: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    where = f"date >= '{year}-01-01T00:00:00' AND date < '{year + 1}-01-01T00:00:00'"

    with client() as c:
        count_rows = get_page(c, {"$select": "count(1) AS n", "$where": where})
        total = int(count_rows[0]["n"])
        print(f"\n{total:,} incidents in {year}\n")

        offset = 0
        page_no = 0
        while offset < total:
            page_path = RAW_DIR / f"{year}_{page_no:04d}.json"

            if page_path.exists():
                print(f"  page {page_no:>3}  cached")
            else:
                rows = get_page(
                    c,
                    {
                        "$select": ",".join(SELECT_FIELDS),
                        "$where": where,
                        # Stable sort is required for correct offset pagination.
                        "$order": "id",
                        "$limit": PAGE_SIZE,
                        "$offset": offset,
                    },
                )
                page_path.write_text(json.dumps(rows))
                print(f"  page {page_no:>3}  {len(rows):>6,} rows  "
                      f"({min(offset + PAGE_SIZE, total):,}/{total:,})")

            offset += PAGE_SIZE
            page_no += 1

    frames = [
        pd.DataFrame(json.loads(p.read_text()))
        for p in sorted(RAW_DIR.glob(f"{year}_*.json"))
    ]
    df = pd.concat(frames, ignore_index=True)
    df = coerce_types(df)

    out = DATA_DIR / f"crimes_{year}.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df):,} rows -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """SODA hands back strings for everything. Cast explicitly."""
    for col in ("date", "updated_on"):
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ("latitude", "longitude", "x_coordinate", "y_coordinate"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("year", "beat", "district", "ward", "community_area"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ("arrest", "domestic"):
        if col in df:
            df[col] = df[col].map({"true": True, "false": False, True: True, False: False})
            df[col] = df[col].astype("boolean")

    for col in ("primary_type", "location_description", "fbi_code", "iucr"):
        if col in df:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------- inspect


def inspect(path: str) -> None:
    df = pd.read_parquet(path)
    print(f"\n{len(df):,} rows x {len(df.columns)} cols\n")
    print(df.dtypes.to_string())

    print("\n--- null counts ---")
    nulls = df.isna().sum()
    print(nulls[nulls > 0].sort_values(ascending=False).to_string() or "  none")

    print("\n--- key integrity ---")
    print(f"  id unique:            {df['id'].is_unique}")
    print(f"  case_number unique:   {df['case_number'].is_unique}")
    dupes = df['case_number'].duplicated().sum()
    print(f"  case_number dupes:    {dupes:,}")

    print("\n--- date coverage ---")
    print(f"  {df['date'].min()}  ->  {df['date'].max()}")

    print("\n--- geo coverage ---")
    geo = df['latitude'].notna().sum()
    print(f"  {geo:,} / {len(df):,} rows geocoded ({100 * geo / len(df):.1f}%)")

    print("\n--- primary_type ---")
    print(df['primary_type'].value_counts().head(15).to_string())

    print("\n--- arrest rate by type (top 10 by volume) ---")
    top = df['primary_type'].value_counts().head(10).index
    rate = (
        df[df['primary_type'].isin(top)]
        .groupby('primary_type', observed=True)['arrest']
        .agg(['mean', 'size'])
        .sort_values('size', ascending=False)
    )
    rate.columns = ['arrest_rate', 'n']
    print(rate.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "peek"

    if cmd == "peek":
        peek(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    elif cmd == "year":
        fetch_year(int(sys.argv[2]))
    elif cmd == "inspect":
        inspect(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)