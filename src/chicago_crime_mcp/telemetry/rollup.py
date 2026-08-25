"""The batch job: what the logs say about how the tool surface is being used.

**Failures are the product signal, and that is forced by the protocol, not
chosen for elegance.** MCP hands a server structured arguments and nothing else
-- no prompt, no conversation, no user. So "what do people ask about" is not a
question this data can answer, and any attempt to infer it would be inventing.
What the arguments *do* show is where the surface let a model down: a category
it invented, a neighborhood nobody taught us, a filter combination that is
always empty. Every report here is a variation on that.

**Read with a declared schema.** ``read_json`` is given explicit ``columns``
rather than left to infer. Inference over a glob would derive a slightly
different schema per day -- a day with no errors has no error values to type --
and then fail to unify them. Declaring the schema also means a record field that
gets renamed breaks here loudly instead of silently reading as NULL.

**DuckDB and nothing else.** No Postgres, no server, no fastmcp. This is a cron
job over files; it should stay runnable against a directory of logs copied off a
box.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from chicago_crime_mcp.telemetry.sink import TelemetryConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: The record's fields, as DuckDB types. Mirrors
#: :class:`~chicago_crime_mcp.telemetry.record.CallRecord`; the two are kept in
#: step by a test that compares them field for field, because the failure mode
#: otherwise is a column that silently reads NULL forever.
COLUMNS: dict[str, str] = {
    "ts": "VARCHAR",
    "call_id": "VARCHAR",
    "session_id": "VARCHAR",
    "client_id": "VARCHAR",
    "transport": "VARCHAR",
    "tool": "VARCHAR",
    "duration_ms": "DOUBLE",
    "args": "JSON",
    "outcome": "VARCHAR",
    "route_store": "VARCHAR",
    "route_tier": "VARCHAR",
    "route_table": "VARCHAR",
    "route_reason": "VARCHAR",
    "route_elapsed_ms": "DOUBLE",
    "row_count": "BIGINT",
    "truncated": "BOOLEAN",
    "cursor_issued": "BOOLEAN",
    "taxonomy_mode": "VARCHAR",
    "warning_codes": "VARCHAR[]",
    "result_bytes": "BIGINT",
    "resolution_kind": "VARCHAR",
    "error_code": "VARCHAR",
    "error_field": "VARCHAR",
    "error_received": "VARCHAR",
    "error_nearest_match": "VARCHAR",
}

#: Arguments that are mechanism rather than filter. Excluded from the "filter
#: combination" grouping: a page size does not explain why nothing matched, and
#: leaving it in would split one real combination across several rows.
MECHANISM_ARGS = ("limit", "cursor", "offset", "taxonomy")

#: Columns of ``reference/neighborhood_aliases.csv``, so the backlog is emitted
#: as a file that can be diffed against the real one rather than a report
#: somebody has to retype.
ALIAS_COLUMNS = (
    "alias",
    "match_kind",
    "target_geography",
    "target_value",
    "target_label",
    "note",
)


@dataclass(frozen=True)
class Report:
    """One named table of findings.

    Attributes:
        title: What the table answers.
        note: Why it is worth looking at, in a sentence -- the reports are read
            by whoever is on rotation, not only by whoever wrote them.
        rows: The findings, most important first.
    """

    title: str
    note: str
    rows: list[dict[str, Any]]


def connect(paths: Sequence[Path] | Path) -> duckdb.DuckDBPyConnection:
    """Open an in-memory database with the log files exposed as ``calls``.

    In memory because there is nothing to keep: the files are the durable
    artifact and the rollup is derived. A run that wants to keep its output
    writes the reports, not the database.

    Args:
        paths: A directory of daily files, or explicit file paths.

    Returns:
        A connection with a ``calls`` view over every record found.

    Raises:
        FileNotFoundError: If no log files match.
    """
    if isinstance(paths, Path):
        files = sorted(paths.glob("calls-*.jsonl")) if paths.is_dir() else [paths]
    else:
        files = list(paths)
    if not files:
        raise FileNotFoundError(f"no telemetry log files found in {paths}")

    conn = duckdb.connect()
    columns = ", ".join(f"'{name}': '{kind}'" for name, kind in COLUMNS.items())
    # Inlined rather than bound: DuckDB cannot prepare a CREATE VIEW, so the
    # paths are escaped for a SQL string literal instead. They come from a CLI
    # argument, which is why the doubling is not optional.
    sources = ", ".join("'" + str(path).replace("'", "''") + "'" for path in files)
    conn.execute(
        f"""
        CREATE VIEW calls AS
        SELECT *, CAST(ts AS TIMESTAMP) AS at
        FROM read_json(
            [{sources}],
            format = 'newline_delimited',
            columns = {{{columns}}}
        )
        """
    )
    return conn


def _rows(conn: duckdb.DuckDBPyConnection, sql: str, params: Sequence[Any] = ()) -> list[dict]:
    """Run a query and return dict rows.

    Args:
        conn: The open connection.
        sql: The query.
        params: Bound parameters.

    Returns:
        One dict per row, keyed by column name.
    """
    cursor = conn.execute(sql, list(params))
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


#: The arg keys that count as a filter combination, as a SQL expression. Built
#: from a list literal rather than a Python tuple's repr: the repr of a
#: one-element tuple carries a trailing comma and is not valid SQL, which is a
#: trap that would spring the day somebody shortens MECHANISM_ARGS.
_FILTER_KEYS = (
    "list_sort(list_filter(json_keys(args), k -> NOT list_contains("
    + "[" + ", ".join(f"'{name}'" for name in MECHANISM_ARGS) + "], k)))"
)


def overview(conn: duckdb.DuckDBPyConnection) -> Report:
    """Calls, outcomes and latency per tool.

    Args:
        conn: A connection from :func:`connect`.

    Returns:
        The report, busiest tool first.
    """
    return Report(
        title="Calls by tool",
        note=(
            "Where the traffic is, and how often each tool disappoints. A high "
            "empty rate is a vocabulary problem; a high error rate is an "
            "under-specified tool description."
        ),
        rows=_rows(
            conn,
            """
            SELECT tool,
                   count(*)                                        AS calls,
                   count(*) FILTER (outcome = 'empty')             AS empty,
                   count(*) FILTER (outcome = 'error')             AS errors,
                   round(count(*) FILTER (outcome = 'empty')  / count(*), 4) AS empty_rate,
                   round(count(*) FILTER (outcome = 'error')  / count(*), 4) AS error_rate,
                   round(quantile_cont(duration_ms, 0.5), 1)       AS p50_ms,
                   round(quantile_cont(duration_ms, 0.95), 1)      AS p95_ms,
                   round(quantile_cont(duration_ms, 0.99), 1)      AS p99_ms
            FROM calls
            GROUP BY tool
            ORDER BY calls DESC
            """,
        ),
    )


def empty_by_filters(conn: duckdb.DuckDBPyConnection, *, min_calls: int = 1) -> Report:
    """Which filter combinations come back empty, and how reliably.

    A combination that is *always* empty is the interesting one: it means the
    surface accepts a question it can never answer, which is a description
    problem rather than a data problem.

    Args:
        conn: A connection from :func:`connect`.
        min_calls: Ignore combinations seen fewer times than this.

    Returns:
        The report, worst rate first.
    """
    return Report(
        title="Empty results by filter combination",
        note=(
            "Valid filters that matched nothing. A combination at 100% has never "
            "once returned a row -- treat that as a tool description that invites "
            "a question the data cannot answer."
        ),
        rows=_rows(
            conn,
            f"""
            SELECT tool,
                   {_FILTER_KEYS} AS filters,
                   count(*)                            AS calls,
                   count(*) FILTER (outcome = 'empty') AS empty,
                   round(count(*) FILTER (outcome = 'empty') / count(*), 4) AS empty_rate
            FROM calls
            WHERE outcome IN ('ok', 'empty')
            GROUP BY tool, filters
            HAVING count(*) >= ?
               AND count(*) FILTER (outcome = 'empty') > 0
            ORDER BY empty_rate DESC, empty DESC
            """,
            [min_calls],
        ),
    )


def resolution_misses(conn: duckdb.DuckDBPyConnection) -> Report:
    """The names ``resolve_neighborhood`` could not resolve, ranked by demand.

    This is the alias backlog, and the reason the record carries
    ``resolution_kind`` as a column of its own. A soft miss (``suggestion``)
    looks like a successful call in every other report -- it returned candidates,
    it raised nothing -- so counting errors would miss most of it.

    There is deliberately no "what the server guessed instead" column. For this
    one tool the error path passes ``suggest_nearest=False`` -- difflib answers
    "Bronzeville" with "Andersonville", 19 km away and perfectly fluent -- so
    such a column would be empty by construction rather than merely often empty.

    Args:
        conn: A connection from :func:`connect`.

    Returns:
        The report, most-asked name first.
    """
    return Report(
        title="resolve_neighborhood misses (the alias backlog)",
        note=(
            "Names people asked for that this server could not resolve. 'suggestion' "
            "means it returned guesses the model was told not to filter on; 'none' "
            "means it raised. Both are places somebody wanted and did not get. "
            "Curate the top of this list into reference/neighborhood_aliases.csv."
        ),
        rows=_rows(
            conn,
            """
            SELECT json_extract_string(args, '$.name')     AS name,
                   count(*)                                AS asked,
                   count(DISTINCT session_id)              AS sessions,
                   list_distinct(list(resolution_kind))    AS kinds
            FROM calls
            WHERE tool = 'resolve_neighborhood'
              AND resolution_kind IN ('suggestion', 'none')
              AND json_extract_string(args, '$.name') IS NOT NULL
            GROUP BY name
            ORDER BY asked DESC, name
            """,
        ),
    )


def arg_failures(conn: duckdb.DuckDBPyConnection) -> Report:
    """Which arguments the model gets wrong, and what it invents instead.

    Args:
        conn: A connection from :func:`connect`.

    Returns:
        The report, most frequent first.
    """
    return Report(
        title="Malformed arguments by field",
        note=(
            "Each row is a teaching error the model had to recover from. The "
            "invented values are the useful part: a value several callers reach "
            "for is a synonym the tool description should have accepted or named. "
            "error_code='unhandled' is not a teaching error -- it is a bug."
        ),
        rows=_rows(
            conn,
            """
            SELECT tool,
                   error_code,
                   coalesce(error_field, '(none)')     AS field,
                   count(*)                            AS errors,
                   list_distinct(
                       list(error_received) FILTER (error_received IS NOT NULL)
                   )[:8]                               AS invented_values,
                   count(*) FILTER (error_nearest_match IS NOT NULL) AS suggested
            FROM calls
            WHERE outcome = 'error'
            GROUP BY tool, error_code, field
            ORDER BY errors DESC
            """,
        ),
    )


def latency_by_route(conn: duckdb.DuckDBPyConnection) -> Report:
    """Where the time goes, split by the route the query actually took.

    ``overhead_ms`` is the gap between the whole call and the query inside it:
    validation, mapping and envelope construction. Reported because a slow
    serializer and a slow database want opposite fixes, and the tool-level
    number alone cannot tell them apart.

    Args:
        conn: A connection from :func:`connect`.

    Returns:
        The report, slowest route first.
    """
    return Report(
        title="Latency and response size by route",
        note=(
            "overhead_ms is this server's own time -- the whole call minus the "
            "query it contains. If it dominates, the database is not the problem."
        ),
        rows=_rows(
            conn,
            """
            SELECT coalesce(route_store, '(none)')                  AS store,
                   coalesce(route_tier, '-')                        AS tier,
                   count(*)                                         AS calls,
                   round(quantile_cont(duration_ms, 0.5), 1)        AS p50_ms,
                   round(quantile_cont(duration_ms, 0.95), 1)       AS p95_ms,
                   round(quantile_cont(
                       duration_ms - coalesce(route_elapsed_ms, 0), 0.5), 1) AS overhead_ms,
                   round(quantile_cont(result_bytes, 0.5))          AS p50_bytes,
                   max(result_bytes)                                AS max_bytes,
                   count(*) FILTER (truncated)                      AS truncated
            FROM calls
            WHERE outcome IN ('ok', 'empty')
            GROUP BY store, tier
            ORDER BY p95_ms DESC
            """,
        ),
    )


def warning_frequency(conn: duckdb.DuckDBPyConnection) -> Report:
    """How often each qualification is attached to an answer.

    A warning that fires on nearly every call has stopped carrying information,
    which is a reason to narrow its trigger rather than to leave it.

    Args:
        conn: A connection from :func:`connect`.

    Returns:
        The report, most frequent first.
    """
    return Report(
        title="Warnings attached to answers",
        note=(
            "A warning on almost every call trains its reader to ignore it. "
            "Check the share, not only the count."
        ),
        rows=_rows(
            conn,
            """
            WITH exploded AS (
                SELECT unnest(warning_codes) AS code FROM calls
            )
            SELECT code,
                   count(*) AS attached,
                   round(count(*) / (SELECT count(*) FROM calls), 4) AS share_of_calls
            FROM exploded
            GROUP BY code
            ORDER BY attached DESC
            """,
        ),
    )


#: The reports a plain run produces, in reading order.
REPORTS = (
    overview,
    empty_by_filters,
    resolution_misses,
    arg_failures,
    latency_by_route,
    warning_frequency,
)


def alias_backlog_rows(misses: Report, *, limit: int = 25) -> list[dict[str, str]]:
    """Turn the miss report into rows shaped like the alias reference file.

    The target columns are left blank on purpose. Deciding that "Bronzeville"
    means Douglas-and-Grand-Boulevard is a judgement about Chicago, and this job
    has no business guessing at it -- what it can do is spare somebody the
    typing and put the demand count in the note.

    Args:
        misses: The report from :func:`resolution_misses`.
        limit: How many of the most-asked names to emit.

    Returns:
        Rows ready for :mod:`csv`, most-asked first.
    """
    return [
        {
            "alias": row["name"],
            "match_kind": "",
            "target_geography": "",
            "target_value": "",
            "target_label": "",
            "note": f"TODO: asked {row['asked']}x across {row['sessions']} session(s).",
        }
        for row in misses.rows[:limit]
    ]


def render(reports: Sequence[Report]) -> str:
    """Format the reports as plain text.

    Args:
        reports: What to render.

    Returns:
        The whole report as one string.
    """
    out: list[str] = []
    for report in reports:
        out.append(f"\n{report.title}\n{'=' * len(report.title)}")
        out.append(f"{report.note}\n")
        if not report.rows:
            out.append("  (nothing)\n")
            continue
        headers = list(report.rows[0])
        widths = {
            h: max(len(h), *(len(_cell(r[h])) for r in report.rows)) for h in headers
        }
        out.append("  " + "  ".join(h.ljust(widths[h]) for h in headers))
        out.append("  " + "  ".join("-" * widths[h] for h in headers))
        for row in report.rows:
            out.append("  " + "  ".join(_cell(row[h]).ljust(widths[h]) for h in headers))
        out.append("")
    return "\n".join(out)


def _cell(value: Any) -> str:
    """Render one cell.

    Args:
        value: The value.

    Returns:
        Its display form; ``-`` for NULL, comma-joined for a list.
    """
    if value is None:
        return "-"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def main(argv: list[str] | None = None) -> int:
    """Run the rollup over a directory of logs and print the reports.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 on success, 1 when there is nothing to read.
    """
    parser = argparse.ArgumentParser(
        description="Roll up the MCP server's per-call telemetry logs.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Log directory or a single .jsonl file (default: TELEMETRY_LOG_DIR).",
    )
    parser.add_argument(
        "--min-calls",
        type=int,
        default=1,
        help="Ignore filter combinations seen fewer times than this.",
    )
    parser.add_argument(
        "--alias-backlog",
        type=Path,
        default=None,
        help="Write the resolve_neighborhood misses as an alias-file CSV stub.",
    )
    args = parser.parse_args(argv)

    path = args.path or TelemetryConfig.from_env().log_dir
    try:
        conn = connect(path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        reports = [
            empty_by_filters(conn, min_calls=args.min_calls) if fn is empty_by_filters else fn(conn)
            for fn in REPORTS
        ]
        total = conn.execute("SELECT count(*) FROM calls").fetchone()[0]
        print(f"{total} tool call(s) from {path}")
        print(render(reports))

        if args.alias_backlog:
            misses = next(r for r in reports if r.title.startswith("resolve_neighborhood"))
            rows = alias_backlog_rows(misses)
            args.alias_backlog.parent.mkdir(parents=True, exist_ok=True)
            with args.alias_backlog.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ALIAS_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote {len(rows)} backlog row(s) to {args.alias_backlog}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALIAS_COLUMNS",
    "COLUMNS",
    "MECHANISM_ARGS",
    "REPORTS",
    "Report",
    "alias_backlog_rows",
    "arg_failures",
    "connect",
    "empty_by_filters",
    "latency_by_route",
    "main",
    "overview",
    "render",
    "resolution_misses",
    "warning_frequency",
]
