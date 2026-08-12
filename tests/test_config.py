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
    assert cfg.parquet_root == DEFAULT_PARQUET_ROOT
    assert cfg.duckdb_path == DEFAULT_DUCKDB_PATH


def test_from_env_reads_overrides():
    env = {
        "DATABASE_URL": "postgresql://u:p@db.internal:5432/prod",
        "PARQUET_ROOT": "/mnt/volume/parquet",
        "DUCKDB_PATH": "/mnt/volume/crime.duckdb",
    }
    cfg = StoreConfig.from_env(env)
    assert cfg.database_url == env["DATABASE_URL"]
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


def test_there_is_no_cache_tier():
    # The Redis cache was planned, measured, and cut -- see "Why there is no
    # cache" in the README. This guards the decision rather than an
    # implementation: the failure it prevents is half-reverting the cut, leaving
    # a config field or a compose service for a tier nothing uses, which is the
    # state the cut existed to clean up. Re-adding a cache should mean deleting
    # this test on purpose, not discovering it went red.
    assert not hasattr(StoreConfig(), "redis_url"), (
        "StoreConfig has a redis_url again; if the cache decision was revisited, "
        "update the README section and remove this test deliberately."
    )
    # Matches a service declaration (two-space-indented key under `services:`)
    # or an image line, never prose -- the compose header explains the cut and
    # names Redis on purpose.
    compose = COMPOSE_FILE.read_text()
    assert not re.search(r"^\s{2}redis:\s*$", compose, re.MULTILINE), (
        "docker-compose.yml declares a redis service again; nothing in the "
        "codebase connects to one."
    )
    assert not re.search(r"^\s*image:.*redis", compose, re.MULTILINE | re.IGNORECASE), (
        "docker-compose.yml runs a redis image again; nothing connects to it."
    )


def test_config_is_frozen():
    cfg = StoreConfig()
    try:
        cfg.database_url = "mutated"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("StoreConfig should be immutable")
