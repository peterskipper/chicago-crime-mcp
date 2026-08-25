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

from evals.harness import CaseResult, load_cases, run_suite


def scorecard(results: list[CaseResult]) -> str:
    """Render the results as a table plus a per-affordance summary.

    Args:
        results: Every graded case.

    Returns:
        The scorecard.
    """
    lines = ["", "Case                                     Checks   Turns  Result", "-" * 72]
    for result in results:
        held = sum(1 for c in result.checks if c.passed)
        if result.passed:
            mark = "pass"
        else:
            mark = "known-hard" if result.case.expected_failure else "FAIL"
        lines.append(
            f"{result.case.id:<40} {held:>2}/{len(result.checks):<2}   "
            f"{result.transcript.turns:>3}    {mark}"
        )
        for check in result.checks:
            if not check.passed:
                lines.append(f"{'':<40}   - {check.name}: {check.detail}")

    by_affordance: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_affordance[result.case.affordance].append(result.passed)

    lines += ["", "By affordance", "-" * 72]
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
    args = parser.parse_args(argv)

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

    results = asyncio.run(run_suite(cases))
    print(scorecard(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(as_json(results), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if any(not r.passed and not r.case.expected_failure for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
