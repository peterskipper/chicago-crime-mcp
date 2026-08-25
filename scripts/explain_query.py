"""Measure what a query actually costs, and whether an index would change that.

This is the instrument behind every "we measured it" claim in the schema
comments. It exists so those comments can say *why* a thing is fast or slow
without freezing a table of milliseconds into a file where they will silently
rot: when you want the numbers, run this and get today's numbers on your machine.

Run it with no arguments for a worked example -- the question that decided
``incidents_hood_date_idx``:

    python scripts/explain_query.py

Or ask your own:

    python scripts/explain_query.py \\
        --sql "SELECT id FROM incidents WHERE ward = %s ORDER BY date DESC LIMIT 51" \\
        --param 42 \\
        --index "(ward, date DESC, id DESC)"


How to read a measurement
-------------------------

**Buffers matter more than milliseconds.** A buffer is one 8 KB page the
database touched. That number is a property of the plan and barely moves between
runs; milliseconds depend on your cache, your disk and what else the machine is
doing. Two queries at the same wall time touching 200 and 68,000 pages are not
equally good -- the second one is fine only while the data fits in RAM, and it is
the one that falls over on a small production box. Judge a plan by buffers, use
milliseconds as a sanity check.

``hit`` is pages found in Postgres's own cache; ``read`` is pages it had to go
get. A warm database reports almost all hits, which is why a slow query can look
fast right after you have run it five times.

**`rows_removed_by_filter` is the tell for a missing index.** It counts rows the
plan fetched and then threw away. A query returning 51 rows that discarded 77,000
to find them is reading the table and filtering in memory: the index it needs
does not exist, or the planner decided not to use it.

**Repeat, and take the median.** The first execution is cold. ``EXPLAIN ANALYZE``
also adds its own per-row instrumentation overhead, so absolute times run high --
another reason to compare buffers between variants rather than trusting the clock.

**Parameters are not the same as literals.** Postgres can pick a different plan
for ``WHERE x = 'Loop'`` than for ``WHERE x = $1``, because with a literal it
knows how selective the value is. Pass real parameters (``--param``) so you
measure the shape your application actually sends. Related: this connects with
``prepare_threshold=None`` -- after five executions psycopg starts reusing a
*generic* plan chosen without knowing your values, which has been measured on
this project to be catastrophically slower. See ``store.postgres.queries``.


How to decide whether to build an index
---------------------------------------

1. **Use the span people actually ask for.** The single biggest trap. An index
   can look worthless over eleven years and be worth two orders of magnitude over
   one -- because what decides it is how many rows match *inside the date window*,
   not how many distinct values the column has. Measure several spans.
2. **Prefer a composite that also delivers the sort order.** For a
   ``WHERE x = ? ORDER BY date DESC, id DESC LIMIT n`` shape, ``(x, date DESC, id
   DESC)`` lets the planner stop after n rows. A bare ``(x)`` only supports a
   bitmap scan plus a sort, so it reads every matching row in the span first --
   often much worse despite being much smaller.
3. **Check the planner is stable.** Create, ``ANALYZE``, measure; drop and repeat.
   If the plan flips between runs the costs are near a crossover and the index
   will help unpredictably. ``--trials`` does this for you.
4. **Weigh the size.** Index size is reported. An index the planner ignores is
   pure cost.

Needs a live Postgres with data in it. Creating a candidate index on a large
table is real work (seconds to minutes) and real disk; every candidate is dropped
again in a ``finally``, including on Ctrl-C.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import psycopg

from chicago_crime_mcp.store.config import StoreConfig

#: Name every candidate index is built under, so a crashed run leaves at most one
#: identifiable object behind rather than a scatter of guesses.
CANDIDATE_INDEX = "explain_candidate_idx"

#: Executions per measurement. Enough for a median to mean something without
#: making a slow query tedious.
DEFAULT_REPEAT = 7


@dataclass(frozen=True)
class Measurement:
    """What one plan cost, over several executions.

    Attributes:
        label: How this variant was described, e.g. ``"no index"``.
        median_ms: Median execution time. The least trustworthy field here.
        min_ms: Fastest execution, i.e. the fully warm case.
        max_ms: Slowest execution, usually the first.
        hit: Pages served from Postgres's cache on the final run.
        read: Pages fetched from the operating system on the final run. Non-zero
            means the data did not fit in ``shared_buffers``.
        rows_removed_by_filter: Rows fetched and discarded, summed over the plan.
            Large values relative to the rows returned mean a missing index.
        scans: Human-readable scan nodes, e.g. ``"Index Scan on incidents_date_idx"``.
        index_size: Size of the candidate index, if this variant built one.
        plan: The raw plan JSON, for when the summary is not enough.
    """

    label: str
    median_ms: float
    min_ms: float
    max_ms: float
    hit: int
    read: int
    rows_removed_by_filter: int
    scans: tuple[str, ...]
    index_size: str | None = None
    plan: dict = field(default_factory=dict, repr=False)

    @property
    def buffers(self) -> int:
        """Total pages touched -- the number to compare between variants."""
        return self.hit + self.read


def _walk(node: dict):
    """Yield every node in a plan tree, depth first."""
    yield node
    for child in node.get("Plans", ()):
        yield from _walk(child)


def _summarize(label: str, plans: list[dict], times: list[float]) -> Measurement:
    """Fold several executions of one plan into a :class:`Measurement`.

    Buffer and filter counts come from the last execution rather than an average:
    they are plan properties, so they are identical across runs once warm, and an
    average would only blur the cold first run into them.

    Buffers are read off the **root node only**. Postgres accumulates them up the
    tree, so each parent already includes everything its children touched, and
    summing the tree would count the same pages several times over. Rows removed
    by filter are the opposite -- reported per node -- so those do get summed.

    Args:
        label: How to describe this variant.
        plans: The ``Plan`` object from each execution.
        times: ``Execution Time`` from each execution, in milliseconds.

    Returns:
        The folded measurement.
    """
    last = plans[-1]
    nodes = list(_walk(last))
    scans = tuple(
        f"{n['Node Type']} on {n['Index Name']}" if n.get("Index Name") else n["Node Type"]
        for n in nodes
        if "Scan" in n["Node Type"]
    )
    return Measurement(
        label=label,
        median_ms=statistics.median(times),
        min_ms=min(times),
        max_ms=max(times),
        hit=last.get("Shared Hit Blocks", 0),
        read=last.get("Shared Read Blocks", 0),
        rows_removed_by_filter=sum(int(n.get("Rows Removed by Filter", 0)) for n in nodes),
        scans=scans,
        plan=last,
    )


def measure(
    conn: psycopg.Connection,
    sql: str,
    params: tuple = (),
    label: str = "as-is",
    repeat: int = DEFAULT_REPEAT,
) -> Measurement:
    """Run one query under ``EXPLAIN (ANALYZE, BUFFERS)`` several times.

    ``ANALYZE`` means the query is genuinely executed, not just planned -- so do
    not point this at anything that writes.

    Args:
        conn: An open connection. Should have ``prepare_threshold=None``; see
            :func:`connect`.
        sql: The statement, with ``%s`` placeholders for any parameters.
        params: Values for those placeholders.
        label: How to describe this variant in the output.
        repeat: How many times to execute it.

    Returns:
        The folded :class:`Measurement`.

    Raises:
        psycopg.Error: Propagated from the database, e.g. on invalid SQL.
    """
    plans, times = [], []
    for _ in range(repeat):
        result = conn.execute(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params
        ).fetchone()[0][0]
        plans.append(result["Plan"])
        times.append(result["Execution Time"])
    return _summarize(label, plans, times)


def compare(
    conn: psycopg.Connection,
    sql: str,
    params: tuple = (),
    candidates: tuple[str, ...] = (),
    table: str = "incidents",
    repeat: int = DEFAULT_REPEAT,
) -> list[Measurement]:
    """Measure the query as it stands, then once per candidate index.

    Each candidate is created, ``ANALYZE``d so the planner has statistics for it,
    measured, and dropped again -- in a ``finally``, so an interrupted run does
    not leave an index behind.

    Args:
        conn: An open connection, in autocommit mode (``CREATE INDEX`` wants it).
        sql: The statement to measure.
        params: Values for its placeholders.
        candidates: Index definitions to try, each the parenthesized column list
            of a ``CREATE INDEX``, e.g. ``"(neighborhood, date DESC, id DESC)"``.
        table: The table the candidates are built on.
        repeat: Executions per measurement.

    Returns:
        One :class:`Measurement` per variant, baseline first.
    """
    results = [measure(conn, sql, params, "no candidate index", repeat)]
    for definition in candidates:
        try:
            conn.execute(f"CREATE INDEX {CANDIDATE_INDEX} ON {table} {definition}")
            conn.execute(f"ANALYZE {table}")
            size = conn.execute(
                "SELECT pg_size_pretty(pg_relation_size(%s::regclass))", (CANDIDATE_INDEX,)
            ).fetchone()[0]
            found = measure(conn, sql, params, definition, repeat)
            results.append(
                Measurement(**{**found.__dict__, "index_size": size})
            )
        finally:
            conn.execute(f"DROP INDEX IF EXISTS {CANDIDATE_INDEX}")
    return results


@contextmanager
def without_indexes(conn: psycopg.Connection, names: tuple[str, ...]) -> Iterator[None]:
    """Temporarily drop existing indexes, then put them back exactly as they were.

    The question "would this index be worth building" is easy to answer before
    you build it and awkward afterwards, because the baseline you want to compare
    against no longer exists. This recovers it: the original ``CREATE INDEX``
    statements are read back out of ``pg_indexes``, so what is restored is what
    was there, not a guess.

    Rebuilding costs whatever the index costs to build -- seconds on a few
    million rows. Restoration is in a ``finally``, so an interrupted run still
    puts them back.

    Args:
        conn: An open connection in autocommit mode.
        names: Index names to drop for the duration.

    Yields:
        None, with those indexes absent.

    Raises:
        ValueError: If a named index does not exist, rather than silently
            measuring a baseline that was already the baseline.
    """
    saved = {}
    for name in names:
        row = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname = %s", (name,)
        ).fetchone()
        if row is None:
            raise ValueError(f"no such index: {name}")
        saved[name] = row[0]
    try:
        for name in names:
            conn.execute(f"DROP INDEX {name}")
        yield
    finally:
        for definition in saved.values():
            conn.execute(definition)


def stability(
    conn: psycopg.Connection,
    sql: str,
    params: tuple = (),
    definition: str = "",
    table: str = "incidents",
    trials: int = 3,
    repeat: int = DEFAULT_REPEAT,
) -> list[Measurement]:
    """Build, measure and drop one candidate several times over.

    The planner chooses from cost estimates built by ``ANALYZE``, which samples
    the table at random. When two plans cost about the same, consecutive builds
    can pick differently -- and an index that helps only sometimes is a poor
    reason to pay for one. Identical buffer counts across trials mean the choice
    is settled.

    Args:
        conn: An open connection in autocommit mode.
        sql: The statement to measure.
        params: Values for its placeholders.
        definition: The candidate index's parenthesized column list.
        table: The table to build it on.
        trials: How many build/measure/drop cycles to run.
        repeat: Executions per measurement.

    Returns:
        One :class:`Measurement` per trial.
    """
    return [
        compare(conn, sql, params, (definition,), table, repeat)[1]
        for _ in range(trials)
    ]


def render(measurements: list[Measurement]) -> str:
    """Format measurements as a table, widest column first.

    Args:
        measurements: What to render.

    Returns:
        A printable multi-line string.
    """
    width = max(len(m.label) for m in measurements)
    lines = [
        f"  {'variant'.ljust(width)}  {'median':>9}  {'buffers':>10}  "
        f"{'filtered':>9}  {'size':>8}  plan"
    ]
    for m in measurements:
        reads = f" ({m.read:,} read)" if m.read else ""
        lines.append(
            f"  {m.label.ljust(width)}  {m.median_ms:8.2f}ms  {m.buffers:>10,}  "
            f"{m.rows_removed_by_filter:>9,}  {(m.index_size or '-'):>8}  "
            f"{m.scans[0] if m.scans else '?'}{reads}"
        )
    return "\n".join(lines)


def connect(autocommit: bool = True) -> psycopg.Connection:
    """Open a connection configured the way the application's own queries are.

    ``prepare_threshold=None`` disables psycopg's automatic prepared statements.
    That matters here for the same reason it matters in production: once psycopg
    prepares a statement it may switch to a plan chosen without knowing the
    parameter values, and measuring the wrong plan is worse than not measuring.

    Args:
        autocommit: Needed for ``CREATE INDEX``; harmless otherwise.

    Returns:
        An open connection to ``StoreConfig.database_url``.
    """
    return psycopg.connect(
        StoreConfig.from_env().database_url,
        prepare_threshold=None,
        autocommit=autocommit,
    )


# -- the worked example ------------------------------------------------------

DEMO_SQL = """
SELECT id, date, primary_type_canonical
FROM incidents
WHERE neighborhood = ANY(%s) AND date >= %s AND date < %s
ORDER BY date DESC, id DESC
LIMIT 51
"""

#: The same question asked over three windows. The wide span is the one that
#: makes the index look pointless, and it is the one nobody asks.
DEMO_SPANS = (
    ("all 11 years", "2015-01-01", "2027-01-01"),
    ("one year", "2025-01-01", "2026-01-01"),
    ("last 90 days", "2026-05-25", "2026-08-23"),
)

DEMO_CANDIDATES = ("(neighborhood)", "(neighborhood, date DESC, id DESC)")

#: The index this comparison originally decided on. It ships in schema.sql now,
#: so reproducing the decision means taking it away again for the baseline.
DEMO_SHIPPED_INDEX = "incidents_hood_date_idx"


def demo(conn: psycopg.Connection, repeat: int) -> None:
    """Reproduce the comparison that decided ``incidents_hood_date_idx``.

    Args:
        conn: An open connection in autocommit mode.
        repeat: Executions per measurement.
    """
    hood = conn.execute(
        """SELECT neighborhood FROM incidents WHERE neighborhood IS NOT NULL
           GROUP BY 1 ORDER BY count(*) LIMIT 1"""
    ).fetchone()[0]
    n = conn.execute(
        "SELECT count(*) FROM incidents WHERE neighborhood = %s", (hood,)
    ).fetchone()[0]
    print(
        f"search_incidents shape, filtered to the rarest neighborhood:\n"
        f"  {hood} -- {n:,} rows.\n"
        f"  Selective filters are where an index earns its keep; a common value\n"
        f"  matches often enough that scanning date order finds a page quickly.\n"
    )
    print(
        f"  ({DEMO_SHIPPED_INDEX} ships in schema.sql, so it is dropped for the\n"
        f"  duration and rebuilt afterwards -- otherwise the baseline already has it.)\n"
    )
    with without_indexes(conn, (DEMO_SHIPPED_INDEX,)):
        for label, lo, hi in DEMO_SPANS:
            print(f"{label}:")
            print(render(
                compare(conn, DEMO_SQL, ([hood], lo, hi), DEMO_CANDIDATES, repeat=repeat)
            ))
            print()
    print(
        "Read the buffers column, not the clock. Over the widest span the composite\n"
        "barely matters -- which is what made it look not worth building. Over the\n"
        "spans people actually ask for it wins, because it satisfies the filter AND\n"
        "the sort, so the planner stops at 51 rows instead of discarding thousands\n"
        "to find them. A bare (neighborhood) index cannot do that: it supports a\n"
        "bitmap scan plus a sort, so it reads every matching row in the span first."
    )


def _parse_param(raw: str) -> Any:
    """Interpret a ``--param`` value, as JSON when it looks like JSON.

    So ``42`` arrives as an int and ``["Loop"]`` as a list for ``= ANY(%s)``,
    while a bare ``Loop`` stays a string.

    Args:
        raw: The value as typed on the command line.

    Returns:
        The parsed value.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run either the worked example or a supplied query.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sql", help="Statement to measure. Omit to run the worked example.")
    parser.add_argument(
        "--param", action="append", default=[], type=_parse_param,
        help="A parameter value, repeatable. JSON is parsed: 42, [\"Loop\"].",
    )
    parser.add_argument(
        "--index", action="append", default=[],
        help='Candidate index columns, repeatable, e.g. "(ward, date DESC)".',
    )
    parser.add_argument("--table", default="incidents", help="Table candidates are built on.")
    parser.add_argument(
        "--drop", action="append", default=[],
        help="Existing index to measure without, repeatable. Rebuilt afterwards.",
    )
    parser.add_argument(
        "--repeat", type=int, default=DEFAULT_REPEAT, help="Executions per measurement."
    )
    parser.add_argument(
        "--trials", type=int, default=0,
        help="Rebuild the first --index this many times to check the planner is stable.",
    )
    args = parser.parse_args(argv)

    with connect() as conn:
        if not args.sql:
            demo(conn, args.repeat)
            return
        params = tuple(args.param)
        with without_indexes(conn, tuple(args.drop)):
            if args.trials:
                print(render(stability(
                    conn, args.sql, params, args.index[0], args.table, args.trials, args.repeat
                )))
            else:
                print(render(compare(
                    conn, args.sql, params, tuple(args.index), args.table, args.repeat
                )))


if __name__ == "__main__":
    main()
