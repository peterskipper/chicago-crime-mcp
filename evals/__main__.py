"""``python -m evals`` -- run the suite and print a scorecard.

Needs an Anthropic API key and the stores the server itself needs: a loaded
Postgres and a built rollup database. It makes real API calls and costs real
money, which is why it is not in ``make ci`` and never will be.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from evals.harness import CaseResult, load_cases, preflight, resolve_model, run_suite

#: Width of the progress bar, in characters.
BAR_WIDTH = 24


def progress(index: int, total: int, result: CaseResult) -> None:
    """Print one line as each case finishes, to stderr.

    A bar rather than a spinner because the work is countable, and a line per
    case rather than one rewritten in place because the interesting part is
    *which* case is slow or failing -- a suite run takes minutes and is usually
    watched intermittently. stderr keeps the scorecard on stdout pipeable.

    Args:
        index: 1-based position of the case that just finished.
        total: How many cases are running.
        result: Its graded outcome.
    """
    filled = round(BAR_WIDTH * index / total)
    bar = "#" * filled + "." * (BAR_WIDTH - filled)
    if result.passed:
        mark = "pass"
    else:
        mark = "known-hard" if result.case.expected_failure else "FAIL"
    held = sum(1 for c in result.checks if c.passed)
    print(
        f"[{bar}] {index:>2}/{total}  {result.case.id:<46} "
        f"{held}/{len(result.checks)}  {result.transcript.turns} turns  {mark}",
        file=sys.stderr,
        flush=True,
    )


def scorecard(results: list[CaseResult]) -> str:
    """Render the results as a table plus a per-affordance summary.

    Args:
        results: Every graded case.

    Returns:
        The scorecard.
    """
    width = max((len(r.case.id) for r in results), default=4) + 2
    header = f"{'Case':<{width}} Checks  Turns  Result"
    # The model is part of the result, not a footnote: a scorecard only means
    # something next to another one from the same instrument.
    lines = ["", f"model: {resolve_model()}", "", header, "-" * len(header)]
    for result in results:
        held = sum(1 for c in result.checks if c.passed)
        if result.passed:
            mark = "pass"
        else:
            mark = "known-hard" if result.case.expected_failure else "FAIL"
        lines.append(
            f"{result.case.id:<{width}} {held:>2}/{len(result.checks):<2}  "
            f"{result.transcript.turns:>4}   {mark}"
        )
        for check in result.checks:
            if not check.passed:
                lines.append(f"{'':<{width}}   - {check.name}: {check.detail}")

    by_affordance: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_affordance[result.case.affordance].append(result.passed)

    lines += ["", "By affordance", "-" * len(header)]
    for affordance, outcomes in sorted(by_affordance.items()):
        lines.append(f"  {affordance:<24} {sum(outcomes)}/{len(outcomes)}")

    checks = [c for r in results for c in r.checks]
    passed = [r for r in results if r.passed]
    hard_failures = [r for r in results if not r.passed and not r.case.expected_failure]
    tokens = sum(r.transcript.usage.get("output_tokens", 0) for r in results)
    lines += [
        "",
        f"{len(passed)}/{len(results)} cases, "
        f"{sum(1 for c in checks if c.passed)}/{len(checks)} checks, "
        f"{len(hard_failures)} unexpected failure(s), "
        f"{tokens} output tokens",
        "",
    ]
    return "\n".join(lines)


def as_json(results: list[CaseResult]) -> str:
    """Render the results as JSON, for keeping a run or diffing two.

    Args:
        results: Every graded case.

    Returns:
        The serialized results.
    """
    return json.dumps(
        [
            {
                "model": resolve_model(),
                "id": r.case.id,
                "affordance": r.case.affordance,
                "question": r.case.question,
                "passed": r.passed,
                "expected_failure": r.case.expected_failure,
                "turns": r.transcript.turns,
                "usage": r.transcript.usage,
                "answer": r.transcript.answer,
                "calls": [
                    {"tool": c.name, "arguments": c.arguments, "ok": c.ok, "error": c.error}
                    for c in r.transcript.calls
                ],
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks
                ],
            }
            for r in results
        ],
        indent=2,
        default=str,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the suite.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        0 if every case that was expected to pass did, else 1.
    """
    parser = argparse.ArgumentParser(description="Run the MCP tool-surface evals.")
    parser.add_argument("--only", action="append", help="Run just this case id (repeatable).")
    parser.add_argument("--affordance", help="Run only cases for this affordance.")
    parser.add_argument("--json", type=Path, help="Also write the full results here.")
    parser.add_argument("--list", action="store_true", help="List the cases and exit.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check the server, stores and cases without calling the model.",
    )
    args = parser.parse_args(argv)

    if args.preflight:
        findings = asyncio.run(preflight())
        for line in findings:
            print(line)
        return 1 if any(line.startswith("FAIL") for line in findings) else 0

    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    if args.affordance:
        cases = [c for c in cases if c.affordance == args.affordance]
    if not cases:
        print("no cases matched", file=sys.stderr)
        return 1

    if args.list:
        for case in cases:
            print(f"{case.id:<45} {case.affordance:<20} {case.question}")
        return 0

    print(
        f"running {len(cases)} case(s) on {resolve_model()}",
        file=sys.stderr,
        flush=True,
    )
    results = asyncio.run(run_suite(cases, on_progress=progress))
    print(scorecard(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(as_json(results), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if any(not r.passed and not r.case.expected_failure for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
