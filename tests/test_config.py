"""Unit tests for the storage-layer configuration.

Pure logic (no DB), so these always run - not gated behind the ``integration``
marker. The environment is injected rather than mutated via ``os.environ`` so
tests stay isolated and order-independent.
"""

from __future__ import annotations

import re
from pathlib import Path

from chicago_crime_mcp.store.config import (
    DEFAULT_DATABASE_URL,
    DEFAULT_DUCKDB_PATH,
    DEFAULT_PARQUET_ROOT,
    DEFAULT_REDIS_URL,
    StoreConfig,
)

COMPOSE_FILE = Path(__file__).resolve().parents[1] / "docker-compose.yml"


def _compose_default(name: str) -> str:
    """Return the fallback of a ``${NAME:-default}`` interpolation in the compose file.

    The container boots with these defaults when no ``.env`` overrides them, so
    they are the source of truth the local config defaults must mirror.

    Args:
        name: The environment variable name inside the interpolation.

    Returns:
        The literal default string (the part after ``:-``).
    """
    text = COMPOSE_FILE.read_text()
    match = re.search(rf"\$\{{{name}:-([^}}]+)\}}", text)
    assert match is not None, f"no ${{{name}:-default}} found in {COMPOSE_FILE.name}"
    return match.group(1)


def test_from_env_uses_defaults_when_empty():
    cfg = StoreConfig.from_env({})
    assert cfg.database_url == DEFAULT_DATABASE_URL
    assert cfg.redis_url == DEFAULT_REDIS_URL
    assert cfg.parquet_root == DEFAULT_PARQUET_ROOT
    assert cfg.duckdb_path == DEFAULT_DUCKDB_PATH


def test_from_env_reads_overrides():
    env = {
        "DATABASE_URL": "postgresql://u:p@db.internal:5432/prod",
        "REDIS_URL": "redis://cache.internal:6379/1",
        "PARQUET_ROOT": "/mnt/volume/parquet",
        "DUCKDB_PATH": "/mnt/volume/crime.duckdb",
    }
    cfg = StoreConfig.from_env(env)
    assert cfg.database_url == env["DATABASE_URL"]
    assert cfg.redis_url == env["REDIS_URL"]
    assert cfg.parquet_root == Path("/mnt/volume/parquet")
    assert cfg.duckdb_path == Path("/mnt/volume/crime.duckdb")


def test_path_fields_are_paths_not_strings():
    # A production DATABASE_URL should not drag the path fields off their
    # defaults; and those defaults must be Path, not str, for downstream joins.
    cfg = StoreConfig.from_env({"DATABASE_URL": "postgresql://x@y/z"})
    assert isinstance(cfg.parquet_root, Path)
    assert isinstance(cfg.duckdb_path, Path)


def test_default_database_url_matches_compose():
    # Guard against config.py and docker-compose.yml drifting apart: a fresh
    # checkout with no .env must connect to the container the compose file boots.
    user = _compose_default("POSTGRES_USER")
    password = _compose_default("POSTGRES_PASSWORD")
    db = _compose_default("POSTGRES_DB")
    port = _compose_default("POSTGRES_PORT")
    expected = f"postgresql://{user}:{password}@localhost:{port}/{db}"
    assert DEFAULT_DATABASE_URL == expected, (
        f"DEFAULT_DATABASE_URL ({DEFAULT_DATABASE_URL!r}) does not match the "
        f"docker-compose.yml defaults ({expected!r}); update whichever is wrong."
    )


def test_default_redis_url_matches_compose():
    # The Redis DB index (/0) is a client-side choice not expressed in compose;
    # only the port is shared, so that is all we cross-check.
    port = _compose_default("REDIS_PORT")
    assert f":{port}/" in DEFAULT_REDIS_URL, (
        f"DEFAULT_REDIS_URL ({DEFAULT_REDIS_URL!r}) does not use the "
        f"docker-compose.yml Redis port ({port})."
    )


def test_config_is_frozen():
    cfg = StoreConfig()
    try:
        cfg.database_url = "mutated"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("StoreConfig should be immutable")
