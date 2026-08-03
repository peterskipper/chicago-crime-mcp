"""Unit tests for the year-partitioned backfill.

A mock SODA backend serves both the ``count(1)`` query and keyset data pages
from a fixed in-memory row set, so we can exercise partition writing, the
year-granularity checkpoint (skip / re-pull), and canonicalization without
touching the network.
"""

from __future__ import annotations

import re

import httpx
import pandas as pd

from chicago_crime_mcp.ingest import backfill
from chicago_crime_mcp.ingest.socrata import SodaClient

# Two rows share IUCR 0281 under the two drifted labels; one is a plain THEFT.
ROWS = [
    {"id": "1", "iucr": "0810", "primary_type": "THEFT",
     "date": "2023-05-01T00:00:00.000", "arrest": "false", "beat": "0111"},
    {"id": "2", "iucr": "0281", "primary_type": "CRIM SEXUAL ASSAULT",
     "date": "2023-06-01T00:00:00.000", "arrest": "true", "beat": "0222"},
    {"id": "3", "iucr": "0281", "primary_type": "CRIMINAL SEXUAL ASSAULT",
     "date": "2023-07-01T00:00:00.000", "arrest": "false", "beat": "0333"},
]
REF = {"0810": "THEFT", "0281": "CRIMINAL SEXUAL ASSAULT"}
# No override for either code, so stable_category mirrors the canonical type.
CURATED: dict[str, str] = {}


def crime_client(rows, count_override=None):
    """A SodaClient backed by an in-memory row set honoring count + keyset."""

    def handler(request):
        params = request.url.params
        select = params.get("$select", "")
        where = params.get("$where", "")
        if "count(1)" in select:
            n = count_override if count_override is not None else len(rows)
            return httpx.Response(200, json=[{"n": str(n)}])
        limit = int(params["$limit"])
        m = re.search(r"id > (\d+)", where)
        lo = int(m.group(1)) if m else 0
        page = [r for r in rows if int(r["id"]) > lo][:limit]
        return httpx.Response(200, json=page)

    return SodaClient(transport=httpx.MockTransport(handler), app_token="T")


def test_partition_path_is_hive_layout(tmp_path):
    p = backfill.partition_path(2023, base=tmp_path)
    assert p == tmp_path / "year=2023" / "part.parquet"


def test_year_where_bounds_the_year():
    w = backfill.year_where(2023)
    assert "date >= '2023-01-01T00:00:00'" in w
    assert "date < '2024-01-01T00:00:00'" in w


def test_backfill_year_writes_partition_and_canonicalizes(tmp_path):
    client = crime_client(ROWS)
    # page_size=2 forces two keyset pages (2 rows, then 1).
    result = backfill.backfill_year(client, 2023, REF, CURATED, base=tmp_path, page_size=2)

    assert result == {"year": 2023, "rows": 3, "expected": 3, "skipped": False}
    out = backfill.partition_path(2023, base=tmp_path)
    assert out.exists()

    df = pd.read_parquet(out)
    assert len(df) == 3
    # both 0281 labels collapsed; raw preserved.
    assert df.sort_values("id")["primary_type_canonical"].tolist() == [
        "THEFT", "CRIMINAL SEXUAL ASSAULT", "CRIMINAL SEXUAL ASSAULT",
    ]
    assert "CRIM SEXUAL ASSAULT" in df["primary_type"].tolist()  # provenance kept
    # Both taxonomies are materialized at ingest, so no store has to derive one.
    assert df["stable_category"].tolist() == df["primary_type_canonical"].tolist()
    # dtypes were coerced and zero-padded beat survived.
    assert df["id"].dtype == "Int64"
    assert df["arrest"].dtype == "boolean"
    assert df.sort_values("id")["beat"].tolist() == ["0111", "0222", "0333"]


def test_backfill_year_skips_when_complete(tmp_path):
    client = crime_client(ROWS)
    backfill.backfill_year(client, 2023, REF, CURATED, base=tmp_path, page_size=2)
    # second run: partition already holds all 3 rows -> skipped, not re-pulled.
    again = backfill.backfill_year(client, 2023, REF, CURATED, base=tmp_path, page_size=2)
    assert again["skipped"] is True
    assert again["rows"] == 3


def test_backfill_year_force_repulls(tmp_path):
    client = crime_client(ROWS)
    backfill.backfill_year(client, 2023, REF, CURATED, base=tmp_path, page_size=2)
    forced = backfill.backfill_year(
        client, 2023, REF, CURATED, base=tmp_path, page_size=2, force=True
    )
    assert forced["skipped"] is False
    assert forced["rows"] == 3


def test_backfill_year_repulls_on_count_mismatch(tmp_path):
    # Pre-write an incomplete partition (only 1 row) at the Hive path.
    out = backfill.partition_path(2023, base=tmp_path)
    out.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(out, index=False)

    # API reports 3 -> mismatch with the 1-row partition -> re-pull to 3.
    client = crime_client(ROWS)
    result = backfill.backfill_year(client, 2023, REF, CURATED, base=tmp_path, page_size=2)
    assert result["skipped"] is False
    assert result["rows"] == 3
    assert len(pd.read_parquet(out)) == 3


def test_backfill_range_loops_years(tmp_path):
    client = crime_client(ROWS)
    results = backfill.backfill(client, 2022, 2023, base=tmp_path)
    assert [r["year"] for r in results] == [2022, 2023]
    assert all(r["rows"] == 3 for r in results)
    assert (tmp_path / "year=2022" / "part.parquet").exists()
    assert (tmp_path / "year=2023" / "part.parquet").exists()
