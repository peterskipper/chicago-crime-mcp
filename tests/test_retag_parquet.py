"""Unit tests for the one-off `stable_category` Parquet migration.

The script rewrites local Parquet only, so these run entirely against tmp_path
fixtures -- no network, no database.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# scripts/ is not an importable package, so load the module by path.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retag_parquet.py"
_spec = importlib.util.spec_from_file_location("retag_parquet", _SCRIPT)
retag_parquet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retag_parquet)

CURATED = {"0760": "THEFT"}


def _partition(base: Path, year: int, frame: pd.DataFrame) -> Path:
    path = base / f"year={year}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def legacy(tmp_path):
    """A partition written before `stable_category` existed."""
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "iucr": ["0610", "0760", "9999"],
            "primary_type_canonical": ["BURGLARY", "BURGLARY", "OTHER OFFENSE"],
        }
    )
    return _partition(tmp_path, 2025, frame)


def test_retag_adds_the_column_and_applies_the_override(legacy):
    rows, moved = retag_parquet.retag_partition(legacy, CURATED)

    assert (rows, moved) == (3, 1)
    df = pd.read_parquet(legacy)
    assert df["stable_category"].tolist() == ["BURGLARY", "THEFT", "OTHER OFFENSE"]
    # The source taxonomy is not rewritten -- 0760 is still BURGLARY to the city.
    assert df["primary_type_canonical"].tolist() == [
        "BURGLARY", "BURGLARY", "OTHER OFFENSE",
    ]


def test_retag_is_idempotent(legacy):
    retag_parquet.retag_partition(legacy, CURATED)
    first = pd.read_parquet(legacy)
    retag_parquet.retag_partition(legacy, CURATED)
    second = pd.read_parquet(legacy)

    pd.testing.assert_frame_equal(first, second)


def test_retag_reapplies_a_changed_curation(legacy):
    """Re-running is how a curation change reaches partitions already on disk."""
    retag_parquet.retag_partition(legacy, CURATED)
    rows, moved = retag_parquet.retag_partition(legacy, {"0610": "TRESPASS"})

    assert (rows, moved) == (3, 1)
    df = pd.read_parquet(legacy)
    # The old override is gone, not layered on top of the new one.
    assert df["stable_category"].tolist() == ["TRESPASS", "BURGLARY", "OTHER OFFENSE"]


def test_dry_run_reports_without_writing(legacy):
    before = pd.read_parquet(legacy)
    rows, moved = retag_parquet.retag_partition(legacy, CURATED, dry_run=True)

    assert (rows, moved) == (3, 1)
    pd.testing.assert_frame_equal(pd.read_parquet(legacy), before)
    assert "stable_category" not in pd.read_parquet(legacy).columns


def test_retag_leaves_no_temp_file_behind(legacy):
    retag_parquet.retag_partition(legacy, CURATED)
    assert list(legacy.parent.glob("*.tmp")) == []


def test_main_walks_every_partition(tmp_path, caplog):
    for year in (2024, 2025):
        _partition(
            tmp_path,
            year,
            pd.DataFrame(
                {"id": [1], "iucr": ["0760"], "primary_type_canonical": ["BURGLARY"]}
            ),
        )
    retag_parquet.main(["--base", str(tmp_path)])

    for year in (2024, 2025):
        df = pd.read_parquet(tmp_path / f"year={year}" / "part.parquet")
        assert df["stable_category"].tolist() == ["THEFT"]


def test_main_warns_when_there_is_nothing_to_do(tmp_path, caplog):
    retag_parquet.main(["--base", str(tmp_path / "empty")])
    assert "no partitions found" in caplog.text
