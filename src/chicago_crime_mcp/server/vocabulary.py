"""The closed sets the tools validate against, read from the data once.

Every value a tool will reject comes from the dataset rather than from a
constant: a ``Literal`` frozen at import time goes stale the moment CPD mints a
code or retires a beat, and a server that rejects a value the data contains is
worse than one that never checked.

**Why this is validated eagerly rather than inferred from an empty result.** It
is tempting to skip the check, run the query, and only look up the valid values
when nothing comes back -- the enumeration would then cost nothing on the happy
path. It is also wrong: ``types=["BURGLARY", "BURGLERY"]`` compiles to an ``IN``
list, matches the real category, and returns a confident non-empty page with the
typo silently dropped. The caller cannot see that half its filter was ignored.
So the values are checked before the query, and the cost is removed by caching
instead.

**The cache is per rollup build, which is what makes it free.** These sets change
when the nightly rebuild lands and at no other time, and
:class:`~chicago_crime_mcp.server.context.ServerContext` already notices that --
it reopens the DuckDB connection when the file is swapped underneath it. So the
vocabulary is loaded on first use after each open and reused until the next one.
This is also the first thing that belongs in a shared cache when one exists: it
is the same eight queries in every process, answered identically until the
rollups rebuild.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chicago_crime_mcp.store.normalize import GEO_COLUMN, TYPE_COLUMN, Geography, Taxonomy

if TYPE_CHECKING:  # pragma: no cover - import-time only, for annotations
    import duckdb

    from chicago_crime_mcp.store.duckdb.queries import DatasetMeta, GeocodeCoverage


@dataclass(frozen=True)
class Vocabulary:
    """Every valid value, plus the dataset facts each tool reports.

    Attributes:
        categories: Valid offense categories, per taxonomy.
        geographies: Valid values, per geography dimension. ``citywide`` maps to
            an empty tuple: it has no column, so it has no values.
        dataset: The rollup build's provenance -- the window, the row count, when
            it was built. Every tool puts this in its envelope, including the
            Postgres-backed ones, so provenance does not get thinner depending
            on which store happened to answer.
        geocoding: Geocoding completeness over the whole window, by category.
    """

    categories: dict[Taxonomy, tuple[str, ...]]
    geographies: dict[Geography, tuple[str | int, ...]]
    dataset: DatasetMeta
    geocoding: GeocodeCoverage

    def categories_for(self, taxonomy: Taxonomy) -> tuple[str, ...]:
        """Return the valid offense categories under one taxonomy.

        Args:
            taxonomy: Which taxonomy.

        Returns:
            The categories, sorted.
        """
        return self.categories[taxonomy]

    def values_for(self, geography: Geography) -> tuple[str | int, ...]:
        """Return the valid values of one geography dimension.

        Args:
            geography: Which dimension.

        Returns:
            The values, sorted, in the column's storage type. Empty for
            ``citywide``.
        """
        return self.geographies[geography]


def load(conn: duckdb.DuckDBPyConnection) -> Vocabulary:
    """Read every closed set from a built rollup database.

    Eight small queries against relations the context has already warmed, run
    once per rollup build.

    Args:
        conn: An open connection to a built rollup database.

    Returns:
        The populated vocabulary.
    """
    from chicago_crime_mcp.store.duckdb import queries

    return Vocabulary(
        categories={t: queries.categories(conn, t) for t in TYPE_COLUMN},
        geographies={g: queries.geography_values(conn, g) for g in GEO_COLUMN},
        dataset=queries.dataset_meta(conn),
        geocoding=queries.geocode_coverage(conn),
    )
