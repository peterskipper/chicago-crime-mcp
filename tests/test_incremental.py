"""Unit tests for the incremental (watermark) sync.

A mock SODA backend serves keyset data pages of "updated" rows; tests assert the
merge/dedupe behavior, watermark bootstrap and advance, the overlap cutoff in the
outgoing query, and that updates to unmanaged years are skipped.
"""

from __future__ import annotations

import re
from datetime import timedelta

import httpx
import pandas as pd

from chicago_crime_mcp.ingest import incremental, schema
from chicago_crime_mcp.ingest.socrata import SodaClient

REF = {"0810": "THEFT", "0281": "CRIMINAL SEXUAL ASSAULT"}

INIT_2023 = [
    {"id": "1", "iucr": "0810", "primary_type": "THEFT",
     "date": "2023-01-15T00:00:00.000", "updated_on": "2023-01-20T00:00:00.000",
     "arrest": "false"},
    {"id": "2", "iucr": "0281", "primary_type": "CRIM SEXUAL ASSAULT",
     "date": "2023-02-15T00:00:00.000", "updated_on": "2023-02-20T00:00:00.000",
     "arrest": "false"},
]


def write_partition(base, year, rows):
    """Seed a coerced/canonicalized partition, as the backfill would."""
    df = schema.coerce_types(schema.add_canonical_primary_type(pd.DataFrame(rows), REF))
    path = base / f"year={year}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def updates_client(rows, captured=None):
    """A SodaClient serving keyset pages from a fixed 'updated rows' set."""

    def handler(request):
        if captured is not None:
            captured.append(request)
        params = request.url.params
        limit = int(params["$limit"])
        m = re.search(r"id > (\d+)", params.get("$where", ""))
        lo = int(m.group(1)) if m else 0
        page = [r for r in rows if int(r["id"]) > lo][:limit]
        return httpx.Response(200, json=page)

    return SodaClient(transport=httpx.MockTransport(handler), app_token="T")


# -- state / helpers -------------------------------------------------------


def test_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    incremental.save_state({"watermark": "2023-06-01T00:00:00"}, p)
    assert incremental.load_state(p) == {"watermark": "2023-06-01T00:00:00"}


def test_load_state_missing_returns_empty(tmp_path):
    assert incremental.load_state(tmp_path / "nope.json") == {}


def test_managed_years_and_derive_watermark(tmp_path):
    write_partition(tmp_path, 2023, INIT_2023)
    assert incremental.managed_years(tmp_path) == {2023}
    assert incremental.derive_watermark(tmp_path) == pd.Timestamp("2023-02-20")


# -- incremental_sync ------------------------------------------------------


def test_sync_merges_updates_and_advances_watermark(tmp_path):
    base = tmp_path / "parquet"
    write_partition(base, 2023, INIT_2023)
    state = tmp_path / "state.json"

    updates = [
        # id 1 re-updated: arrest flips true, newer updated_on -> should replace.
        {"id": "1", "iucr": "0810", "primary_type": "THEFT",
         "date": "2023-01-15T00:00:00.000", "updated_on": "2023-06-01T00:00:00.000",
         "arrest": "true"},
        # id 3 brand new in 2023 -> should be added.
        {"id": "3", "iucr": "0281", "primary_type": "CRIMINAL SEXUAL ASSAULT",
         "date": "2023-07-10T00:00:00.000", "updated_on": "2023-06-02T00:00:00.000",
         "arrest": "false"},
    ]
    summary = incremental.incremental_sync(
        updates_client(updates), base=base, state_path=state
    )

    assert summary["pulled"] == 2
    assert summary["years"] == {2023: {"applied": 2, "partition_rows": 3}}
    assert summary["skipped_unmanaged"] == 0
    assert summary["watermark"].startswith("2023-06-02")

    df = pd.read_parquet(base / "year=2023" / "part.parquet").set_index("id")
    assert len(df) == 3
    assert df.loc[1, "arrest"]  # replaced row reflects the newer update
    assert 3 in df.index  # new row landed
    assert incremental.load_state(state)["watermark"].startswith("2023-06-02")


def test_sync_dedupes_keeping_latest_updated_on(tmp_path):
    base = tmp_path / "parquet"
    write_partition(base, 2023, INIT_2023)  # id 1 updated_on 2023-01-20
    # An older-updated copy of id 1 must NOT overwrite the newer stored row.
    stale = [
        {"id": "1", "iucr": "0810", "primary_type": "THEFT",
         "date": "2023-01-15T00:00:00.000", "updated_on": "2023-01-05T00:00:00.000",
         "arrest": "true"},
    ]
    incremental.incremental_sync(
        updates_client(stale), base=base, state_path=tmp_path / "s.json"
    )
    df = pd.read_parquet(base / "year=2023" / "part.parquet").set_index("id")
    assert not df.loc[1, "arrest"]  # kept the newer 2023-01-20 row (arrest false)


def test_sync_uses_overlap_cutoff_in_query(tmp_path):
    base = tmp_path / "parquet"
    write_partition(base, 2023, INIT_2023)  # watermark bootstraps to 2023-02-20
    captured = []
    incremental.incremental_sync(
        updates_client([], captured),
        base=base,
        state_path=tmp_path / "s.json",
        overlap=timedelta(days=1),
    )
    # cutoff = 2023-02-20 - 1 day = 2023-02-19
    assert "updated_on > '2023-02-19T00:00:00'" in captured[0].url.params["$where"]


def test_sync_skips_unmanaged_year(tmp_path):
    base = tmp_path / "parquet"
    write_partition(base, 2023, INIT_2023)  # only 2023 is managed
    updates = [
        {"id": "9", "iucr": "0810", "primary_type": "THEFT",
         "date": "2005-04-01T00:00:00.000", "updated_on": "2023-06-01T00:00:00.000",
         "arrest": "false"},  # a reclassified 2005 incident
    ]
    summary = incremental.incremental_sync(
        updates_client(updates), base=base, state_path=tmp_path / "s.json"
    )
    assert summary["skipped_unmanaged"] == 1
    assert summary["years"] == {}
    assert not (base / "year=2005").exists()  # no phantom partial partition


def test_sync_no_updates_keeps_watermark(tmp_path):
    base = tmp_path / "parquet"
    write_partition(base, 2023, INIT_2023)
    state = tmp_path / "s.json"
    summary = incremental.incremental_sync(updates_client([]), base=base, state_path=state)
    assert summary["pulled"] == 0
    assert summary["years"] == {}
    assert summary["watermark"].startswith("2023-02-20")  # unchanged
    assert incremental.load_state(state)["last_pulled"] == 0


def test_sync_without_watermark_or_partitions_raises(tmp_path):
    import pytest

    with pytest.raises(RuntimeError):
        incremental.incremental_sync(
            updates_client([]), base=tmp_path / "empty", state_path=tmp_path / "s.json"
        )
