"""Process-wide connection lifecycle for the MCP server.

The tools are stateless functions; everything that has to survive between calls
lives here. Three things make this more than a pair of connection handles, and
each of them was measured rather than assumed:

**Postgres is pooled, and pooling must not lose a setting.**
``store.postgres.queries`` depends on ``prepare_threshold=None``: without it,
psycopg prepares a statement after a handful of identical calls and Postgres may
switch to a generic plan that is orders of magnitude slower for a selective
filter. A pool that opened connections its own way would drop that silently, so
it is passed the module's own :data:`~chicago_crime_mcp.store.postgres.queries.CONNECT_KWARGS`.
A single connection is not safe to share across concurrent HTTP requests, which
is what the deployed transport does.

**DuckDB is opened read-only, once, and re-opened when the file moves.** The
rollup build cannot write the live database while a server holds it open --
DuckDB takes an exclusive lock and the nightly rebuild would fail every night --
so ``chicago-crime-rollup`` builds to a temporary file and ``os.replace()``\\ s it
over the live path. That leaves this process holding the *old inode*: correct
data, quietly stale, indefinitely. So the path is ``stat()``\\ ed on each
acquisition and the connection reopened when the inode changes. Read-only is
also what makes concurrent readers legal and makes it structurally impossible
for a tool to mutate the rollups.

Reopening is not merely calling ``connect`` again. DuckDB caches database
instances **by path**, so a second ``connect`` to the same path while the first
connection is still open hands back the same instance and the same stale data.
The old connection has to be closed first, which invalidates any cursor taken
from it, which is why :meth:`ServerContext.duckdb` counts readers and drains
them before swapping. See its docstring for the measurements.

**A freshly opened DuckDB connection is cold, and only the real query path warms
it.** Measured in fresh processes: with no warm-up the first aggregate costs
488 ms; ``SELECT 1`` brings that to 265 ms and a ``count(*)`` over every relation
to 261 ms, because both absorb the same one-time engine start-up and nothing
else. Executing one real, tiny aggregate costs 262 ms and takes the first real
aggregate to **3.6 ms**. The total is the same; what changes is that it is spent
at startup instead of on a caller's first question, and again after each nightly
swap. So the warm-up is a query, not a table scan -- see
:meth:`ServerContext._warm`.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from chicago_crime_mcp.server import vocabulary as vocabulary_module
from chicago_crime_mcp.server.errors import DataUnavailableError
from chicago_crime_mcp.server.vocabulary import Vocabulary
from chicago_crime_mcp.store.config import StoreConfig
from chicago_crime_mcp.store.duckdb.rollups import (
    CODE_MONTH_TABLE,
    COVERAGE_TABLE,
    ROLLUP_TABLES,
)

if TYPE_CHECKING:  # pragma: no cover - import-time only, for annotations
    import duckdb
    import psycopg

log = logging.getLogger(__name__)

#: Relations checked at open, so a half-built database fails at startup rather
#: than at the first tool call that happens to need the missing one. Derived from
#: the builder's own constants, so a rollup table added there is checked here
#: without a second list to remember.
#:
#: This is an existence check and **not** a warm-up, however much it looks like
#: one. ``SELECT count(*)`` is answered from row-group metadata without reading
#: the data, and measured in a fresh process it leaves the first real aggregate
#: at 261 ms against 488 ms for no warm-up at all -- all of that saving being a
#: one-time engine initialisation that any statement at all, ``SELECT 1``
#: included, absorbs just as well. See :func:`ServerContext._warm` for what
#: actually works.
CHECKED_RELATIONS: tuple[str, ...] = (
    *ROLLUP_TABLES,
    CODE_MONTH_TABLE,
    COVERAGE_TABLE,
    "rollup_meta",
)

#: Pool bounds. Small on purpose: the queries are indexed and short, so a deep
#: pool would buy queueing rather than throughput, and Postgres charges real
#: memory per backend.
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 8

#: How long a rollup-file swap waits for in-flight DuckDB cursors to finish
#: before giving up and retrying on the next call. Requests are single-digit
#: milliseconds and a swap happens once a night, so this is never reached in
#: practice; it exists so a wedged request degrades into brief staleness rather
#: than into a stuck server.
SWAP_DRAIN_TIMEOUT = 5.0


class ServerContext:
    """Open connections to both stores, shared by every tool call.

    Not thread-safe to *construct* concurrently, but safe to use once opened:
    the Postgres pool hands out one connection per caller, and the DuckDB
    connection is only ever handed out as a cursor.

    Attributes:
        config: The storage configuration this context was opened against.
    """

    def __init__(self, config: StoreConfig | None = None) -> None:
        """Prepare a context without connecting to anything.

        Args:
            config: Storage settings. Defaults to reading the environment,
                which is what the server does; tests pass one explicitly.
        """
        self.config = config or StoreConfig.from_env()
        self._pool: object | None = None
        self._duck: duckdb.DuckDBPyConnection | None = None
        self._duck_inode: int | None = None
        # Requests currently holding a cursor, and the condition the swap waits
        # on. See `duckdb()` for why the old connection has to be closed and
        # therefore drained rather than simply dropped.
        self._readers = 0
        self._lock = threading.Condition()
        # Valid values and dataset facts, loaded on first use after each DuckDB
        # open and dropped by the swap that made them stale. Guarded by its own
        # lock: building it goes through `duckdb()`, which takes `_lock`.
        self._vocabulary: Vocabulary | None = None
        self._vocab_lock = threading.Lock()

    def open(self) -> None:
        """Connect to both stores and warm the DuckDB page cache.

        Raises:
            DataUnavailableError: If the rollup database is missing or was never
                built. Raised at startup rather than on the first tool call: a
                server that cannot answer anything should fail where an operator
                sees it, not where a model does.
        """
        from psycopg_pool import ConnectionPool

        from chicago_crime_mcp.store.postgres.queries import CONNECT_KWARGS

        pool = ConnectionPool(
            self.config.database_url,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            kwargs=CONNECT_KWARGS,
            # Hands out a connection only after checking it is still alive, so a
            # Postgres restart or an idle-timeout reaper surfaces as a one-off
            # reconnect instead of as a failed tool call.
            check=ConnectionPool.check_connection,
            open=False,
        )
        pool.open()
        self._pool = pool

        self._open_duckdb()
        # Load the vocabulary here rather than on the first call that needs it.
        # Safe from `open()` and only from here: the reopen path runs inside
        # `_lock`, which `vocabulary()` would deadlock on -- after a swap it is
        # rebuilt lazily instead, costing a few milliseconds once a night.
        self.vocabulary()
        log.info(
            "server context open: postgres=%s duckdb=%s",
            self.config.database_url.rsplit("@", 1)[-1],
            self.config.duckdb_path,
        )

    def close(self) -> None:
        """Close both stores. Safe to call more than once."""
        if self._pool is not None:
            self._pool.close()  # type: ignore[attr-defined]
            self._pool = None
        if self._duck is not None:
            self._duck.close()
        self._duck = None
        self._duck_inode = None
        self._vocabulary = None

    @contextmanager
    def postgres(self) -> Iterator[psycopg.Connection]:
        """Borrow a pooled Postgres connection for the duration of a call.

        Yields:
            A live connection, returned to the pool on exit.

        Raises:
            DataUnavailableError: If the context was never opened.
        """
        if self._pool is None:
            raise DataUnavailableError(
                "the server is not connected to Postgres",
                hint="This is a server-side problem, not a problem with the request.",
            )
        with self._pool.connection() as conn:  # type: ignore[attr-defined]
            yield conn

    @contextmanager
    def duckdb(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Borrow a DuckDB cursor, reopening first if the file was swapped.

        A cursor rather than the connection itself: cursors over one DuckDB
        connection are independent, so concurrent tool calls do not interleave
        on a shared one.

        **The swap has to close the old connection, and that is why this
        counts readers.** DuckDB caches database instances *by path* for the
        life of the process: while any connection to the path is open, opening
        it again returns the same instance, still bound to the old inode, so a
        reopen that does not close first silently keeps serving yesterday's
        data. (Verified, along with the fact that passing a different config to
        force a fresh instance does not work either -- DuckDB rejects it with
        "Can't open a connection to same database file with a different
        configuration".) Closing the old connection invalidates cursors taken
        from it, so the swap waits for in-flight requests to finish first.

        Should a request somehow outlast :data:`SWAP_DRAIN_TIMEOUT`, the swap is
        abandoned for this call and retried on the next one: serving correct but
        slightly stale rollups for a few more milliseconds is strictly better
        than closing a connection out from under a running query.

        Yields:
            A cursor onto the current rollup database.

        Raises:
            DataUnavailableError: If the context was never opened.
        """
        with self._lock:
            if self._duck is None:
                raise DataUnavailableError(
                    "the server is not connected to the rollup database",
                    hint="This is a server-side problem, not a problem with the request.",
                )
            if self._current_inode() != self._duck_inode:
                drained = self._lock.wait_for(lambda: self._readers == 0, SWAP_DRAIN_TIMEOUT)
                # Another thread may have done the swap while this one waited.
                if not drained:
                    log.warning(
                        "rollup database was replaced but readers did not drain in %ss; "
                        "serving the previous build and retrying on the next call",
                        SWAP_DRAIN_TIMEOUT,
                    )
                elif self._current_inode() != self._duck_inode:
                    log.info("rollup database was replaced; reopening")
                    self._duck.close()
                    self._duck = None
                    self._open_duckdb()
            conn = self._duck
            self._readers += 1
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()
            with self._lock:
                self._readers -= 1
                self._lock.notify_all()

    def vocabulary(self) -> Vocabulary:
        """Return the valid values and dataset facts for the current build.

        Loaded once per DuckDB open and reused until the nightly rebuild swaps
        the file, which is exactly the lifetime over which the answers cannot
        change. Acquiring the connection first is deliberate: that call is what
        detects the swap and clears the stale cache, so the check below can
        never hand back a vocabulary from a previous build.

        Returns:
            The cached vocabulary, building it if this is the first use.
        """
        with self.duckdb() as conn:
            with self._vocab_lock:
                if self._vocabulary is None:
                    self._vocabulary = vocabulary_module.load(conn)
                return self._vocabulary

    def _open_duckdb(self) -> None:
        """Open the rollup database read-only, warm it, and record its inode.

        Raises:
            DataUnavailableError: If the file is absent or holds no build.
        """
        import duckdb as duckdb_module

        from chicago_crime_mcp.store.duckdb import rollups

        path = Path(self.config.duckdb_path)
        if not path.exists():
            raise DataUnavailableError(
                f"no rollup database at {path}",
                hint="Run `chicago-crime-rollup` to build it.",
            )
        try:
            conn = rollups.connect(path, read_only=True)
        except duckdb_module.Error as exc:
            raise DataUnavailableError(
                f"could not open the rollup database at {path}: {exc}",
                hint="Run `chicago-crime-rollup` to rebuild it.",
            ) from exc

        try:
            self._warm(conn)
        except duckdb_module.Error as exc:
            conn.close()
            raise DataUnavailableError(
                f"the rollup database at {path} is missing its tables: {exc}",
                hint="Run `chicago-crime-rollup` to build it.",
            ) from exc

        self._duck = conn
        self._duck_inode = self._current_inode()
        # Anything derived from the previous build is now stale by definition.
        self._vocabulary = None

    def _warm(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Check the relations exist, then absorb the first-aggregate cost.

        The warm-up is **one real, tiny aggregate**, and nothing cheaper works.
        Measured in fresh processes, with the first full-span aggregate timed
        after each strategy:

        =========================  ==========  ==================
        warm-up                    costs       first aggregate
        =========================  ==========  ==================
        none                       --          488 ms
        ``SELECT 1``               0.3 ms      265 ms
        ``count(*)`` per relation  2.1 ms      261 ms
        ``sum()`` per relation     4.9 ms      258 ms
        a real tiny ``aggregate``  262 ms      **3.6 ms**
        =========================  ==========  ==================

        So the cost splits in two. About 225 ms is DuckDB starting up and any
        statement absorbs it. The remaining ~260 ms belongs to the aggregate
        code path itself and only executing that path pays it off -- touching
        the same relations with a different query does not, which is why the
        obvious warm-up loop is worthless here. Note the total is the same
        either way; the point is *where* it is spent. Paid at startup, no one is
        waiting for it. Left unpaid, it lands on a caller's first question, and
        again after every nightly rebuild swaps the file.

        The citywide warm-up query is enough to cover a later query against any
        other geography's table, so this does not need to be repeated per
        relation.

        Args:
            conn: The freshly opened connection.
        """
        from chicago_crime_mcp.store.duckdb import queries

        for relation in CHECKED_RELATIONS:
            conn.execute(f"SELECT count(*) FROM {relation}").fetchone()

        # Scoped to the dataset's own first month rather than a hardcoded date,
        # so the warm-up query returns real rows and exercises the coverage and
        # routing paths a caller's query will take.
        meta = queries.dataset_meta(conn)
        first = meta.min_date.date().replace(day=1)
        queries.aggregate(
            conn,
            queries.AggregateQuery(
                start=first,
                end=first.replace(day=28),
                geography="citywide",
                limit=1,
            ),
        )

    def _current_inode(self) -> int | None:
        """Return the inode of the rollup database file, or None if it is gone.

        Returns:
            The inode number, or None when the path does not resolve -- which is
            the window during a rebuild's ``os.replace()`` and is treated as
            "unchanged" so a request in that window keeps the connection it has
            rather than failing.
        """
        try:
            return Path(self.config.duckdb_path).stat().st_ino
        except OSError:
            return self._duck_inode


#: The context the tools read. Module-level rather than threaded through every
#: tool signature because a tool's parameters are its published schema, and an
#: injected connection handle has no business appearing in what the model reads.
#: Tests replace it through :func:`use_context`.
_context: ServerContext | None = None


def set_context(context: ServerContext | None) -> None:
    """Install the context the tools will use.

    Args:
        context: The open context, or None to clear it.
    """
    global _context
    _context = context


def get_context() -> ServerContext:
    """Return the installed context.

    Returns:
        The context set by the application lifespan.

    Raises:
        DataUnavailableError: If no context has been installed, which means the
            server's startup did not complete.
    """
    if _context is None:
        raise DataUnavailableError(
            "the server has no open store connections",
            hint="This is a server-side problem, not a problem with the request.",
        )
    return _context


@contextmanager
def use_context(context: ServerContext) -> Iterator[ServerContext]:
    """Temporarily install a context, restoring the previous one on exit.

    Args:
        context: The context to install.

    Yields:
        The installed context.
    """
    previous = _context
    set_context(context)
    try:
        yield context
    finally:
        set_context(previous)
