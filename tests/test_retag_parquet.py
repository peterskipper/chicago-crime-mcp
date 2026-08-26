"""Unit tests for the local Parquet migration.

The script rewrites local Parquet only, so these run entirely against tmp_path
fixtures -- no network, no database. The neighborhood polygons come in as a stub
locator so the suite stays off DuckDB's spatial extension; the real one is
exercised by the ``spatial`` tests in test_geo_boundaries.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tests.helpers import StubLocator

# scripts/ is not an importable package, so load the module by path.
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retag_parquet.py"
_spec = importlib.util.spec_from_file_location("retag_parquet", _SCRIPT)
assert _spec is not None and _spec.loader is not None, f"cannot load {_SCRIPT}"
retag_parquet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retag_parquet)

REFERENCE = {"0610": "BURGLARY", "0760": "BURGLARY", "9999": "OTHER OFFENSE"}
CURATED = {"0760": "THEFT"}

WICKER_PARK = (41.9088, -87.6796)
LOOP = (41.8781, -87.6298)
LAKE = (41.8800, -87.5500)


def locator() -> StubLocator:
    """Two points that locate and, by omission, one that does not."""
    return StubLocator({WICKER_PARK: "Wicker Park", LOOP: "Loop"})


def retag(path, curated=CURATED, boundaries=None, **kwargs) -> dict:
    """Call the script with the fixtures' reference data."""
    return retag_parquet.retag_partition(
        path, REFERENCE, curated, boundaries or locator(), **kwargs
    )


def _partition(base: Path, year: int, frame: pd.DataFrame) -> Path:
    path = base / f"year={year}" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _frame(**overrides) -> pd.DataFrame:
    """Three rows: two locatable and distinct, one in the lake."""
    base = {
        "id": [1, 2, 3],
        "iucr": ["0610", "0760", "9999"],
        "primary_type": ["BURGLARY", "BURGLARY", "OTHER OFFENSE"],
        "primary_type_canonical": ["BURGLARY", "BURGLARY", "OTHER OFFENSE"],
        "latitude": [WICKER_PARK[0], LOOP[0], LAKE[0]],
        "longitude": [WICKER_PARK[1], LOOP[1], LAKE[1]],
    }
    base.update(overrides)
    return pd.DataFrame(base)


@pytest.fixture
def legacy(tmp_path):
    """A partition written before `neighborhood` existed."""
    return _partition(tmp_path, 2025, _frame())


# --- adding a column --------------------------------------------------------


def test_retag_adds_the_neighborhood_column(legacy):
    summary = retag(legacy)

    assert summary["rows"] == 3
    assert summary["located"] == 2, "the lake row must not be located"
    assert summary["changed"]["neighborhood"] == 3, "a new column changes every row"

    df = pd.read_parquet(legacy)
    assert df["neighborhood"].tolist()[:2] == ["Wicker Park", "Loop"]
    assert df["neighborhood"].isna().tolist() == [False, False, True]


def test_retag_still_applies_the_offense_curation(legacy):
    """The migration this script was written for has to keep working."""
    retag(legacy)
    df = pd.read_parquet(legacy)
    assert df["stable_category"].tolist() == ["BURGLARY", "THEFT", "OTHER OFFENSE"]
    # The source taxonomy is not rewritten -- 0760 is still BURGLARY to the city.
    assert df["primary_type_canonical"].tolist() == [
        "BURGLARY", "BURGLARY", "OTHER OFFENSE",
    ]


def test_retag_preserves_columns_it_does_not_derive(legacy):
    """Running the whole pipeline must not disturb the rest of the row."""
    before = pd.read_parquet(legacy)
    retag(legacy)
    after = pd.read_parquet(legacy)
    for column in ("id", "iucr", "primary_type", "latitude", "longitude"):
        assert after[column].tolist() == before[column].tolist(), column


# --- idempotence ------------------------------------------------------------


def test_retag_is_idempotent(legacy):
    retag(legacy)
    first = pd.read_parquet(legacy)
    summary = retag(legacy)
    second = pd.read_parquet(legacy)

    pd.testing.assert_frame_equal(first, second)
    assert summary["changed"] == {
        "primary_type_canonical": 0,
        "stable_category": 0,
        "neighborhood": 0,
    }, "a second run must report nothing to do"


def test_unlocatable_rows_do_not_read_as_churn(legacy):
    """Null-to-null is unchanged, or the 1.8% would look like work every run."""
    retag(legacy)
    assert pd.read_parquet(legacy)["neighborhood"].isna().sum() == 1
    assert retag(legacy)["changed"]["neighborhood"] == 0


def test_retag_reapplies_a_changed_curation(legacy):
    """Re-running is how a curation change reaches partitions already on disk."""
    retag(legacy)
    summary = retag(legacy, curated={"0610": "TRESPASS"})

    assert summary["changed"]["stable_category"] == 2, "one code gained it, one lost it"
    df = pd.read_parquet(legacy)
    # The old override is gone, not layered on top of the new one.
    assert df["stable_category"].tolist() == ["TRESPASS", "BURGLARY", "OTHER OFFENSE"]


def test_retag_reapplies_moved_boundaries(legacy):
    """The same mechanism carries a boundary correction, not just a curation one."""
    retag(legacy)
    summary = retag(legacy, boundaries=StubLocator({WICKER_PARK: "Bucktown"}))

    assert summary["changed"]["neighborhood"] == 2, "one row moved, one lost its tag"
    assert pd.read_parquet(legacy)["neighborhood"].tolist()[0] == "Bucktown"


# --- writing safely ---------------------------------------------------------


def test_dry_run_reports_without_writing(legacy):
    before = pd.read_parquet(legacy)
    summary = retag(legacy, dry_run=True)

    assert summary["rows"] == 3
    assert summary["changed"]["neighborhood"] == 3
    pd.testing.assert_frame_equal(pd.read_parquet(legacy), before)
    assert "neighborhood" not in pd.read_parquet(legacy).columns


def test_retag_leaves_no_temp_file_behind(legacy):
    retag(legacy)
    assert list(legacy.parent.glob("*.tmp")) == []


# --- the whole run ----------------------------------------------------------


def test_main_walks_every_partition(tmp_path, monkeypatch):
    stub = locator()
    monkeypatch.setattr(retag_parquet.NeighborhoodBoundaries, "load", lambda: stub)
    monkeypatch.setattr(retag_parquet.schema, "load_iucr_reference", lambda: REFERENCE)
    monkeypatch.setattr(retag_parquet.schema, "load_stable_category_map", lambda: CURATED)
    for year in (2024, 2025):
        _partition(tmp_path, year, _frame())

    retag_parquet.main(["--base", str(tmp_path)])

    for year in (2024, 2025):
        df = pd.read_parquet(tmp_path / f"year={year}" / "part.parquet")
        assert df["stable_category"].tolist() == ["BURGLARY", "THEFT", "OTHER OFFENSE"]
        assert df["neighborhood"].tolist()[:2] == ["Wicker Park", "Loop"]
    assert stub.closed, "the polygons it opened must be released"


def test_main_warns_when_there_is_nothing_to_do(tmp_path, caplog):
    retag_parquet.main(["--base", str(tmp_path / "empty")])
    assert "no partitions found" in caplog.text
