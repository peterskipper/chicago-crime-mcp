"""Shared pytest fixtures.

Integration tests run against a **dedicated test database**, never the dev
database, so running ``pytest -m integration`` can never clobber locally loaded
data. The test database name defaults to the dev database name with a ``_test``
suffix (override the whole DSN with ``TEST_DATABASE_URL``) and is created on
demand if it does not exist. Each test still drops/recreates ``incidents`` for a
hermetic starting state -- safe now that the target is a throwaway database.

Postgres/psycopg imports are lazy (inside fixtures) so that unit-only runs, which
never touch these fixtures, don't require the ``store`` dependencies.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

import pytest


def _derive_dsns() -> tuple[str, str, str]:
    """Resolve the DSNs used to provision and reach the test database.

    The test DSN reuses the dev connection (host/port/credentials) but targets a
    separate database. The maintenance DSN targets the server's standard
    ``postgres`` database and is used only to ``CREATE DATABASE`` the test one.

    Returns:
        A ``(maintenance_dsn, test_dsn, test_db_name)`` tuple.
    """
    from chicago_crime_mcp.store.config import StoreConfig

    override = os.environ.get("TEST_DATABASE_URL")
    base = urlparse(override or StoreConfig.from_env().database_url)
    dev_db = base.path.lstrip("/")
    test_db = dev_db if override else f"{dev_db}_test"
    test_dsn = urlunparse(base._replace(path=f"/{test_db}"))
    maintenance_dsn = urlunparse(base._replace(path="/postgres"))
    return maintenance_dsn, test_dsn, test_db


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Ensure a dedicated test database exists and return its DSN.

    Connects to the maintenance database in autocommit mode (``CREATE DATABASE``
    cannot run inside a transaction) and creates the test database if absent. The
    database is left in place between runs; per-test cleanup happens at the table
    level in :func:`pg_conn`.

    Returns:
        The DSN of the dedicated test database.

    Raises:
        Skips the whole integration suite if Postgres is unreachable.
    """
    import psycopg
    from psycopg import sql

    maintenance_dsn, test_dsn, test_db = _derive_dsns()

    try:
        conn = psycopg.connect(maintenance_dsn, connect_timeout=2, autocommit=True)
    except psycopg.OperationalError as exc:  # pragma: no cover - env dependent
        pytest.skip(f"no Postgres reachable: {exc}")
    try:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (test_db,)
        ).fetchone()
        if not exists:
            # Identifier can't be parameterized; test_db is derived from our own
            # config, never from external input.
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_db)))
    finally:
        conn.close()
    return test_dsn


@pytest.fixture
def pg_conn(test_database_url):
    """An open connection to the test database with a clean ``incidents`` table.

    Drops ``incidents`` before and after each test so every test starts from a
    known-empty state. Safe because ``test_database_url`` is a throwaway database,
    not the dev one.
    """
    import psycopg

    conn = psycopg.connect(test_database_url, connect_timeout=2)
    conn.execute("DROP TABLE IF EXISTS incidents CASCADE")
    conn.commit()
    yield conn
    conn.execute("DROP TABLE IF EXISTS incidents CASCADE")
    conn.commit()
    conn.close()
