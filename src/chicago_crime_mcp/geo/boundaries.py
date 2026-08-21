"""Point-in-polygon tagging: which of the 98 neighborhoods does an incident sit in?

This is the ingest half of ``geo``. The city does not put a neighborhood on an
incident, so the column is derived here, once, and landed in Parquet -- the same
shape as ``stable_category``, and for the same reason: a rule that lives in one
place cannot drift between the two stores that read it.

**Nothing in the serving path imports this module.** Queries answer from the
tagged column, so the server never loads a polygon, never links DuckDB's spatial
extension, and never needs the network that ``INSTALL spatial`` reaches for on
first use. The name-resolution half of ``geo`` lives in :mod:`geo.resolve` and
is pure table lookup for exactly that reason.

**The column is nullable**, unlike ``stable_category``: about 1.8% of rows get no
neighborhood -- roughly 1.55% have no coordinates at all, and a further ~0.3% are
geocoded but fall outside every polygon (the lake, the airport, the city edge).
That is a fact about the data, not a failure to handle, and a NOT NULL here would
reject rows that are legitimately unlocatable.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import duckdb
import pandas as pd

from chicago_crime_mcp.reference import NEIGHBORHOOD_BOUNDARIES_PATH

#: Column names this module reads off an incident frame.
LATITUDE_COLUMN = "latitude"
LONGITUDE_COLUMN = "longitude"

#: The property carrying the neighborhood name in the vendored GeoJSON.
NAME_PROPERTY = "pri_neigh"


class NeighborhoodBoundaries:
    """The 98 neighborhood polygons, loaded once and reused for every batch.

    Loading costs a DuckDB connection, the spatial extension and a 2.3 MB parse,
    so a backfill builds one of these and threads it through every partition
    rather than paying that per file. Use it as a context manager, or call
    :meth:`close`.

    Attributes:
        names: The 98 neighborhood names, sorted, in their stored spelling.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Wrap a connection that already holds the ``neighborhoods`` table.

        Prefer :meth:`load`; this exists so a test can inject a connection
        holding hand-built polygons instead of the vendored file.

        Args:
            conn: A DuckDB connection with ``spatial`` loaded and a
                ``neighborhoods(neighborhood, geom)`` table populated.
        """
        self._conn = conn
        self.names: tuple[str, ...] = tuple(
            row[0]
            for row in conn.execute(
                "SELECT neighborhood FROM neighborhoods ORDER BY neighborhood"
            ).fetchall()
        )

    @classmethod
    def load(cls, path: Path = NEIGHBORHOOD_BOUNDARIES_PATH) -> Self:
        """Load the vendored boundary file into a fresh in-memory DuckDB.

        Args:
            path: Path to the GeoJSON boundary file.

        Returns:
            A ready-to-use instance.

        Raises:
            duckdb.IOException: If the file is missing or unreadable.
            duckdb.Error: If the ``spatial`` extension cannot be installed --
                which on a cold machine means the network was unreachable, since
                it downloads before caching to ``~/.duckdb/extensions``.
        """
        conn = duckdb.connect()
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute(
            f"""
            CREATE TABLE neighborhoods AS
            SELECT {NAME_PROPERTY} AS neighborhood, geom FROM ST_Read(?)
            """,
            [str(path)],
        )
        return cls(conn)

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()

    def __enter__(self) -> Self:
        """Enter a context manager, returning ``self``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the context manager, closing the connection."""
        self.close()

    def locate(self, latitude: pd.Series, longitude: pd.Series) -> pd.Series:
        """Find the neighborhood containing each point.

        Uses ``ST_Contains``, which excludes the boundary itself. The 98
        polygons are disjoint, so that costs nothing on real data -- no incident
        in the dataset falls in two -- and it keeps the result single-valued by
        construction rather than by picking a winner.

        Args:
            latitude: Point latitudes. Nulls are allowed and yield null.
            longitude: Point longitudes, aligned to ``latitude``.

        Returns:
            A nullable-string Series of neighborhood names on ``latitude``'s
            index, holding ``pd.NA`` where the point is missing coordinates or
            falls outside every polygon.

        Raises:
            ValueError: If the join returned a different number of rows than it
                was given, which would mean the polygons had come to overlap and
                the column was no longer single-valued.
        """
        points = pd.DataFrame(
            {
                "i": range(len(latitude)),
                "lat": pd.to_numeric(latitude, errors="coerce").to_numpy(dtype="float64"),
                "lon": pd.to_numeric(longitude, errors="coerce").to_numpy(dtype="float64"),
            }
        )
        # Registered explicitly rather than left to DuckDB's replacement scan,
        # which would resolve the name off this frame's locals -- convenient, and
        # silently shadowed the moment a real table shares the name.
        self._conn.register("_points", points)
        try:
            located = self._conn.execute(
                """
                SELECT p.i, n.neighborhood
                FROM _points AS p
                LEFT JOIN neighborhoods AS n
                  ON ST_Contains(n.geom, ST_Point(p.lon, p.lat))
                ORDER BY p.i
                """
            ).fetchdf()
        finally:
            self._conn.unregister("_points")

        if len(located) != len(points):
            raise ValueError(
                f"point-in-polygon returned {len(located)} rows for {len(points)} points: "
                "the neighborhood polygons are no longer disjoint"
            )
        return pd.Series(
            located["neighborhood"].to_numpy(), index=latitude.index, dtype="string"
        )
