"""Storage-layer configuration: connection DSNs and local data paths.

Every engine is reached purely through an environment variable - a DSN (Data
Source Name: a single connection string, e.g.
``postgresql://user:pass@host:5432/db``) for Postgres, and a filesystem path for
DuckDB/Parquet. Local development and the Railway deployment therefore differ
only in what those variables hold; no code branches on environment.

Defaults line up with ``docker-compose.yml`` so a fresh local checkout works
against the local stack with no ``.env`` at all. Production (Railway) injects
``DATABASE_URL``.

There is deliberately no ``redis_url``. A cache-aside tier was planned and cut
after measurement; see "Why there is no cache" in the README. Should that
decision be revisited, this is where the DSN belongs.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Mirror docker-compose.yml so StoreConfig.from_env() connects to the local
# stack out of the box.
DEFAULT_DATABASE_URL = "postgresql://crime:crime@localhost:5432/chicago_crime"
DEFAULT_PARQUET_ROOT = Path("data/parquet")
DEFAULT_DUCKDB_PATH = Path("data/duckdb/crime.duckdb")


@dataclass(frozen=True)
class StoreConfig:
    """Immutable connection + path settings for the storage layer.

    Attributes:
        database_url: libpq/psycopg DSN for the PostGIS database.
        parquet_root: Root of the Hive-partitioned Parquet dataset that DuckDB
            reads (``year=<YYYY>/part.parquet`` underneath).
        duckdb_path: Path to the persistent DuckDB database file (rollup tables +
            the Parquet-backed view). Its parent directory is created on connect
            by the DuckDB store, not here.
    """

    database_url: str = DEFAULT_DATABASE_URL
    parquet_root: Path = DEFAULT_PARQUET_ROOT
    duckdb_path: Path = DEFAULT_DUCKDB_PATH

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> StoreConfig:
        """Build a config from environment variables, falling back to local defaults.

        Args:
            environ: Environment mapping to read (defaults to ``os.environ``).
                Injectable so tests can supply a controlled environment without
                mutating the process globals.

        Returns:
            A populated :class:`StoreConfig`.
        """
        env = os.environ if environ is None else environ
        return cls(
            database_url=env.get("DATABASE_URL", DEFAULT_DATABASE_URL),
            parquet_root=Path(env.get("PARQUET_ROOT", str(DEFAULT_PARQUET_ROOT))),
            duckdb_path=Path(env.get("DUCKDB_PATH", str(DEFAULT_DUCKDB_PATH))),
        )
