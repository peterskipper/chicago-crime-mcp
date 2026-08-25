"""Shared test helpers for building Parquet fixtures.

The incident schema is defined once here rather than per test module: it mirrors
the 22 coerced columns ``ingest`` writes, and a second copy would drift the first
time a column is added.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
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
        ("stable_category", pa.string()),
        # Nullable, unlike the other derived columns: ~1.8% of real rows have no
        # coordinates or fall outside every polygon.
        ("neighborhood", pa.string()),
    ]
)


def row(**overrides) -> dict:
    """A complete incident row with sane defaults; override only what matters.

    ``stable_category`` defaults to whatever ``primary_type_canonical`` ends up
    being, mirroring the real derivation (an uncurated code falls through to its
    canonical type). So a test that overrides only the canonical type gets a
    consistent row, and one exercising the curated remap sets both explicitly.

    .. warning::
       **Every geography default here is a single constant** -- the same beat,
       district, ward and community area on every row. A fixture that does not
       override them puts all its rows in one place, so a test filtering on a
       geography has nothing to exclude: it passes identically whether the
       predicate is applied or dropped. This has already produced one vacuous
       test.

       So when writing a filter test, give the geography **at least two distinct
       values across the fixture**, and assert against the unfiltered result
       (``filtered < unfiltered``, or an exact id set) rather than an absolute
       count -- an absolute count cannot distinguish "the filter matched
       everything" from "the filter was never applied". The same reasoning
       applies to any low-variety default: ``arrest``, ``domestic`` and the
       offense columns are constants here too.

       ``neighborhood`` is the newest such default and carries an extra trap:
       it is the only geography that can be **null**, so a fixture of nothing but
       the default never exercises the unlocatable rows that a
       neighborhood-grouped aggregate must not silently drop. Give at least one
       fixture row ``neighborhood=None``.

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
        # Consistent with the default coordinates above, which are State and
        # Madison. Change one and change the other.
        neighborhood="Loop",
    )
    base.update(overrides)
    base.setdefault("stable_category", base["primary_type_canonical"])
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


class StubLocator:
    """A :class:`~chicago_crime_mcp.ingest.schema.PointLocator` with no geometry.

    Lets a test exercise the ingest pipeline without DuckDB's spatial extension,
    which downloads on first use and would drag the whole ingest suite behind the
    ``spatial`` marker. Points not in the lookup come back null, which is how a
    test reaches the unlocatable path deliberately.

    Attributes:
        calls: How many times :meth:`locate` ran, so a test can assert the
            pipeline actually invoked it rather than passing by omission.
        closed: Whether :meth:`close` was called. The real locator holds a DuckDB
            connection, so who closes it -- the caller who injected one, or the
            code that loaded one -- is behaviour worth pinning down.
    """

    def __init__(self, mapping: dict[tuple[float, float], str] | None = None) -> None:
        """Build a locator over an explicit ``(latitude, longitude) -> name`` map.

        Args:
            mapping: Coordinates to neighborhood names. Empty means everything
                is unlocatable.
        """
        self._mapping = mapping or {}
        self.calls = 0
        self.closed = False
        #: The distinct names this locator can produce, mirroring
        #: ``NeighborhoodBoundaries.names``.
        self.names = tuple(sorted(set(self._mapping.values())))

    def locate(self, latitude: pd.Series, longitude: pd.Series) -> pd.Series:
        """Look each point up in the mapping, null where it is absent."""
        self.calls += 1
        lat = pd.to_numeric(latitude, errors="coerce")
        lon = pd.to_numeric(longitude, errors="coerce")
        return pd.Series(
            [self._mapping.get((a, b)) for a, b in zip(lat, lon, strict=True)],
            index=latitude.index,
            dtype="string",
        )

    def close(self) -> None:
        """Record that the owner released this locator."""
        self.closed = True

    def __enter__(self) -> StubLocator:
        """Enter a context manager, returning ``self``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the context manager, closing the locator."""
        self.close()


class RecordingSink:
    """A telemetry sink that keeps records in memory instead of on disk.

    Stands in for :class:`~chicago_crime_mcp.telemetry.sink.TelemetrySink`
    wherever a test cares about *what* was recorded rather than how it was
    written. The file format is
    :mod:`tests.test_telemetry_sink`'s subject, not everyone else's.

    Attributes:
        records: Every record written, in order.
    """

    def __init__(self) -> None:
        """Create an empty sink."""
        self.records: list[Any] = []

    def write(self, record: Any) -> None:
        """Keep one record.

        Args:
            record: The :class:`~chicago_crime_mcp.telemetry.record.CallRecord`.
        """
        self.records.append(record)

    @property
    def only(self) -> Any:
        """The single record written.

        Returns:
            The one record.

        Raises:
            AssertionError: If the count is not exactly one -- which is itself
                the assertion most callers want, since a middleware that logs
                twice or not at all is a bug either way.
        """
        assert len(self.records) == 1, f"expected 1 record, got {len(self.records)}"
        return self.records[0]
