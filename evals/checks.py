"""The check vocabulary: what a case is allowed to assert about a run.

Deliberately small and deliberately mechanical. Every check here reads the
*transcript* -- which tools were called, with what, in what order, and what came
back -- rather than grading the prose, because tool-call behaviour is the thing
this server can actually influence. Two checks do look at the answer text, and
they are the narrow kind: a substring that must or must not appear, and whether
the numbers in the answer came from somewhere.

Nothing here imports the Anthropic SDK. Checks are pure functions over a
transcript, which is what lets the whole vocabulary be unit-tested offline while
only the run itself needs an API key.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: Numbers below this many digits are ignored by :func:`grounded_numbers`.
#: "the top 5 categories" and "3 of them" are the model counting its own list,
#: not citing a figure, and demanding a source for those would fail every
#: correct answer.
MIN_GROUNDED_DIGITS = 3

#: Longest series :func:`_derivable` will scan for windows or pairs. Sized to a
#: real period series -- two years of monthly buckets -- rather than to a
#: category cross-product, which for one ward-year is already 236 rows. Measured:
#: allowing those scans on a 236-row series accepted 100% of random three-digit
#: figures. A whole-series *total* is exempt from this limit, being one specific
#: value rather than tens of thousands.
MAX_SERIES = 24


@dataclass(frozen=True)
class ToolCall:
    """One tool call as the harness observed it.

    Attributes:
        name: The tool called.
        arguments: The arguments the model chose.
        ok: Whether the server returned rather than raised.
        error: The teaching message, when it raised.
        result: The structured response, when it returned.
    """

    name: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None = None
    result: dict[str, Any] | None = None


@dataclass
class Transcript:
    """Everything one question produced.

    Attributes:
        case_id: The case that generated it.
        question: The question as asked.
        calls: Tool calls in the order they were made.
        answer: The model's final text.
        turns: How many model turns it took.
        usage: Token counts, for cost reporting.
    """

    case_id: str
    question: str
    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    turns: int = 0
    usage: dict[str, int] = field(default_factory=dict)

    def named(self, tool: str) -> list[ToolCall]:
        """Return the calls made to one tool.

        Args:
            tool: The tool name.

        Returns:
            Its calls, in order.
        """
        return [call for call in self.calls if call.name == tool]

    def first_index(self, tool: str) -> int | None:
        """Return the position of the first call to a tool.

        Args:
            tool: The tool name.

        Returns:
            Its index in :attr:`calls`, or None if never called.
        """
        for index, call in enumerate(self.calls):
            if call.name == tool:
                return index
        return None

    def results_blob(self) -> str:
        """Return every successful result serialized into one string.

        Returns:
            The concatenated JSON of every successful result.
        """
        return json.dumps([c.result for c in self.calls if c.ok], default=str)

    def result_numbers(self) -> tuple[set[float], list[list[float]]]:
        """Return the figures a model could have read, and how they group.

        Two shapes, because grounding needs both. The flat set answers "was this
        number handed to the model at all". The series answer "could the model
        have *computed* it" -- summing a column of buckets into a total is the
        commonest thing a correct answer does, and is indistinguishable from
        invention unless the series is kept intact.

        **Series are per call, not merged across calls**, which was a real bug:
        a question about ward 3 in 2016 and 2024 is two calls, and merging their
        monthly buckets made the total 13,880 rather than the 7,052 and 6,828
        the model correctly reported.

        Only successful calls contribute: a figure cited out of an error message
        is not a sourced figure.

        Returns:
            A ``(all_values, series)`` pair, where ``series`` holds one list per
            repeated key per call.
        """
        flat: set[float] = set()
        all_series: list[list[float]] = []

        for call in self.calls:
            if not call.ok:
                continue
            series: dict[str, list[float]] = {}

            def walk(node: Any, key: str, series: dict[str, list[float]] = series) -> None:
                if isinstance(node, bool):
                    return
                if isinstance(node, (int, float)):
                    flat.add(float(node))
                    series.setdefault(key, []).append(float(node))
                elif isinstance(node, str):
                    # Every digit run inside a string counts as handed to the
                    # model, not just wholly-numeric strings. Three kinds arrive
                    # this way and all are legitimately quotable: zero-padded
                    # districts and IUCR codes ("018", "0325"), the year inside
                    # a period ("2016-01-01"), and numbers written into a
                    # warning message, which the model read with everything else.
                    for run in _digits(node):
                        flat.add(float(run))
                elif isinstance(node, dict):
                    for k, v in node.items():
                        walk(v, k, series)
                elif isinstance(node, list):
                    for item in node:
                        walk(item, key, series)

            walk(call.result, "")
            all_series.extend(series.values())
        return flat, all_series


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check.

    Attributes:
        name: The check that ran.
        passed: Whether it held.
        detail: What was seen, phrased so a failure is diagnosable without
            re-reading the transcript.
    """

    name: str
    passed: bool
    detail: str


def calls_tool(transcript: Transcript, expected: list[str]) -> list[CheckResult]:
    """Require that each named tool was called at least once."""
    return [
        CheckResult(
            name=f"calls_tool[{tool}]",
            passed=bool(transcript.named(tool)),
            detail=f"called {len(transcript.named(tool))}x",
        )
        for tool in expected
    ]


def avoids_tool(transcript: Transcript, forbidden: list[str]) -> list[CheckResult]:
    """Require that none of the named tools was called.

    The negative half of tool choice, and the more informative half: reaching
    for ``search_incidents`` to answer "how many" means listing rows to count
    them, which is exactly the behaviour the aggregate path exists to prevent.
    """
    return [
        CheckResult(
            name=f"avoids_tool[{tool}]",
            passed=not transcript.named(tool),
            detail=f"called {len(transcript.named(tool))}x",
        )
        for tool in forbidden
    ]


def calls_before(transcript: Transcript, pair: list[str]) -> list[CheckResult]:
    """Require that one tool was first called before another was.

    How "look before you guess" is checked: a model that filters on a
    neighborhood name before resolving it got lucky rather than right.
    """
    first, second = pair
    a, b = transcript.first_index(first), transcript.first_index(second)
    passed = a is not None and b is not None and a < b
    return [
        CheckResult(
            name=f"calls_before[{first} -> {second}]",
            passed=passed,
            detail=f"{first} at {a}, {second} at {b}",
        )
    ]


def tool_args(transcript: Transcript, spec: dict[str, dict[str, Any]]) -> list[CheckResult]:
    """Require that a tool was called at least once with given argument values.

    Checked against *some* call rather than every call: a model that gets it
    wrong, reads the teaching error and retries correctly has done the right
    thing, and a check that demanded every attempt be correct would punish
    exactly the recovery this surface is built to produce.
    """
    results = []
    for tool, wanted in spec.items():
        calls = transcript.named(tool)
        matched = [c for c in calls if all(c.arguments.get(k) == v for k, v in wanted.items())]
        results.append(
            CheckResult(
                name=f"tool_args[{tool}]",
                passed=bool(matched),
                detail=(
                    f"wanted {wanted}; saw {[c.arguments for c in calls] or 'no calls'}"
                    if not matched
                    else f"matched on {len(matched)} of {len(calls)} call(s)"
                ),
            )
        )
    return results


def result_filters(transcript: Transcript, spec: dict[str, dict[str, Any]]) -> list[CheckResult]:
    """Require that a tool *applied* given filter values, as the envelope echoes them.

    Distinct from :func:`tool_args`, and the difference is the point. That one
    reads what the model typed; this reads ``filters_applied``, which is the
    value after normalization. A model that sends ``geography_values: [18]`` for
    district 18 is not wrong -- the server zero-pads it to ``"018"`` and answers
    correctly -- so asserting the wire form tests a spelling the surface does not
    require. Assert the interpreted value and the check follows the contract the
    envelope actually makes.

    Args:
        transcript: The run.
        spec: ``{tool: {field: value}}``, compared as strings so ``18`` and
            ``"018"`` are not held apart by type alone.

    Returns:
        One result per tool named.
    """
    results = []
    for tool, wanted in spec.items():
        seen = []
        matched = False
        for call in transcript.named(tool):
            applied = (call.result or {}).get("filters_applied") or {}
            seen.append(applied)
            if all(_as_text(applied.get(k)) == _as_text(v) for k, v in wanted.items()):
                matched = True
        results.append(
            CheckResult(
                name=f"result_filters[{tool}]",
                passed=matched,
                detail=(
                    f"matched on {len(transcript.named(tool))} call(s)"
                    if matched
                    else f"wanted {wanted}; applied {seen or 'no calls'}"
                ),
            )
        )
    return results


def _as_text(value: Any) -> Any:
    """Render a filter value for comparison, ignoring numeric-vs-string typing.

    Zero-padding is preserved -- ``"018"`` and ``"18"`` stay different, because
    the district really is three characters. What is collapsed is only the
    difference between the number 18 and the string "18".

    Args:
        value: A filter value, possibly a list.

    Returns:
        The value as a string, or a list of strings.
    """
    if isinstance(value, list):
        return [_as_text(v) for v in value]
    return str(value)


def answer_contains_any(transcript: Transcript, wanted: list[str]) -> list[CheckResult]:
    """Require that the answer contains at least one of several phrasings.

    For a fact with more than one natural wording. "the feed excludes the most
    recent 7 days" is a real thing to check for and the model may write it as
    "7-day", "seven days" or "a week"; a single substring makes the check about
    phrasing rather than about whether the caveat landed.

    Args:
        transcript: The run.
        wanted: Acceptable substrings, matched case-insensitively.

    Returns:
        One result.
    """
    answer = transcript.answer.lower()
    hit = [needle for needle in wanted if needle.lower() in answer]
    return [
        CheckResult(
            name=f"answer_contains_any[{'|'.join(wanted)}]",
            passed=bool(hit),
            detail=f"matched {hit}" if hit else "none of them appeared",
        )
    ]


def recovers_from_error(transcript: Transcript, _spec: bool) -> list[CheckResult]:
    """Require that a call failed and a later call to the same tool succeeded.

    This is the teaching-error affordance stated as an observable event. Note
    what it does *not* require: that the model never erred. An error here is the
    designed path, and the claim being checked is that the message was enough to
    act on.
    """
    failed = {c.name for c in transcript.calls if not c.ok}
    recovered = {
        call.name
        for index, call in enumerate(transcript.calls)
        if call.ok and call.name in failed and _errored_before(transcript, call.name, index)
    }
    return [
        CheckResult(
            name="recovers_from_error",
            passed=bool(recovered),
            detail=(
                f"errored on {sorted(failed) or 'nothing'}, recovered on {sorted(recovered)}"
            ),
        )
    ]


def _errored_before(transcript: Transcript, tool: str, index: int) -> bool:
    """Return whether a tool failed at any point before a given position.

    Args:
        transcript: The run.
        tool: The tool name.
        index: The position of the successful call.

    Returns:
        True if the same tool errored earlier.
    """
    return any(not c.ok and c.name == tool for c in transcript.calls[:index])


def no_error(transcript: Transcript, _spec: bool) -> list[CheckResult]:
    """Require that nothing failed.

    For the cases where a well-described tool should be usable first time.
    """
    failed = [f"{c.name}: {c.error}" for c in transcript.calls if not c.ok]
    return [
        CheckResult(
            name="no_error",
            passed=not failed,
            detail="; ".join(failed) if failed else "clean",
        )
    ]


def answer_contains(transcript: Transcript, wanted: list[str]) -> list[CheckResult]:
    """Require substrings in the final answer, case-insensitively.

    Kept to facts the answer is *wrong* without -- that a figure covers a wider
    area than was asked about, that a period is incomplete. Not used to grade
    phrasing.
    """
    answer = transcript.answer.lower()
    return [
        CheckResult(
            name=f"answer_contains[{needle}]",
            passed=needle.lower() in answer,
            detail="present" if needle.lower() in answer else "absent",
        )
        for needle in wanted
    ]


def answer_omits(transcript: Transcript, forbidden: list[str]) -> list[CheckResult]:
    """Require that substrings do *not* appear in the final answer.

    The check that catches a confident wrong answer: naming a neighborhood at
    the other end of the city, or reporting a decline from a month that is still
    filling.
    """
    answer = transcript.answer.lower()
    return [
        CheckResult(
            name=f"answer_omits[{needle}]",
            passed=needle.lower() not in answer,
            detail="absent" if needle.lower() not in answer else "present",
        )
        for needle in forbidden
    ]


def grounded_numbers(transcript: Transcript, _spec: bool) -> list[CheckResult]:
    """Require that every substantial figure in the answer is sourced or derivable.

    A heuristic, and the first version of it was a bad one: it demanded that
    every number appear *verbatim* in a tool result, which failed a correct
    answer for summing six monthly buckets into a total. Measured against a real
    run, that produced eight of nine failures and buried every other signal in
    the suite. Three things were being conflated, and only the first is a defect:

    * **invention** -- a figure with no relationship to the data;
    * **derivation** -- a total, a difference, a percentage. Correct arithmetic,
      and exactly what a useful answer does;
    * **identifiers** -- IUCR codes, district numbers, years. Not figures at all.

    So a number passes if it was handed to the model, or if it is the sum of one
    of the result's own series, or a difference or percentage relating two values
    within one series. What is left over is the thing worth failing on.

    It still cannot catch a real figure that is misattributed, and does not try.
    It catches invention, which is the failure that makes a grounded data tool
    worthless.

    Args:
        transcript: The run.
        _spec: Unused; the check is enabled by being listed.

    Returns:
        One result naming any figure that is neither sourced nor derivable.
    """
    flat, series = transcript.result_numbers()
    asked = {float(n) for n in _digits(transcript.question) if n}
    ungrounded = []
    for number in re.findall(r"\d[\d,]*", transcript.answer):
        bare = number.replace(",", "")
        if len(bare) < MIN_GROUNDED_DIGITS:
            continue
        value = float(bare)
        if value in flat or value in asked or _derivable(value, series):
            continue
        ungrounded.append(bare)
    return [
        CheckResult(
            name="grounded_numbers",
            passed=not ungrounded,
            detail=(
                f"neither returned nor derivable from a returned series: {sorted(set(ungrounded))}"
                if ungrounded
                else "every figure sourced or derivable"
            ),
        )
    ]


def _derivable(target: float, series: Sequence[Sequence[float]]) -> bool:
    """Return whether a figure follows from one of the result's own series.

    Covers what a data answer actually computes: a total, a total over part of a
    period series, a change between two totals, and a change as a percentage.

    **The limits here are the whole design, and they were set by measurement.**
    An earlier version allowed contiguous-window sums over any series; fired at a
    real 236-bucket result it accepted *100% of random three-digit figures and
    65% of four-digit ones*. A window scan over a long series produces tens of
    thousands of candidate values and blankets the range, which turns the check
    into decoration. So:

    * a **whole-series total** is allowed on any length -- it is one candidate
      value per series, which is a specific claim;
    * **totals are then related to each other** by difference and percentage,
      which is how "7,052 in 2016 and 6,828 in 2024, a fall of 224" is reached
      across two calls, and there are only a handful of totals;
    * **window sums and pairwise scans over raw values** are restricted to short
      series, the length of a real period series rather than a category
      cross-product.

    Args:
        target: The figure from the answer.
        series: One list of values per repeated key per call.

    Returns:
        True if some series reaches it by one of those routes.
    """
    totals = [sum(values) for values in series if values]
    if any(_close(total, target) for total in totals):
        return True
    if _related(target, totals):
        return True
    for values in series:
        if len(values) < 2 or len(values) > MAX_SERIES:
            continue
        if _window_sum(values, target):
            return True
        if _related(target, values):
            return True
    return False


def _related(target: float, values: Sequence[float]) -> bool:
    """Return whether two of the values differ by the figure, or relate as a percentage.

    Args:
        target: The figure from the answer.
        values: Candidate values.

    Returns:
        True if some pair reaches it.
    """
    if len(values) > MAX_SERIES:
        return False
    for a in values:
        for b in values:
            if _close(abs(a - b), target):
                return True
            if b and (
                _close(round(abs(a - b) / b * 100), target)
                or _close(round(a / b * 100), target)
            ):
                return True
    return False


def _window_sum(values: Sequence[float], target: float) -> bool:
    """Return whether any run of consecutive values totals the figure.

    "Thefts over the last six months" out of a twelve-month series, or one year
    out of two. Runs of length one are skipped -- a single value is already
    covered by the exact-match set, and allowing them here would let any
    returned number satisfy any other check by accident.

    Args:
        values: One series, in the order returned.
        target: The figure from the answer.

    Returns:
        True if a contiguous run sums to it.
    """
    for start in range(len(values)):
        running = 0.0
        for end in range(start, len(values)):
            running += values[end]
            if end > start and _close(running, target):
                return True
    return False


def _close(value: float, target: float) -> bool:
    """Return whether two figures agree once rounding is allowed.

    A percentage the model rounded, or a total it reported to the nearest whole
    number, should not read as invention.

    Args:
        value: The computed figure.
        target: The figure from the answer.

    Returns:
        True if they differ by at most one.
    """
    return abs(value - target) <= 1


def _digits(text: str) -> set[str]:
    """Return every digit run in a string, commas stripped.

    Args:
        text: The text to scan.

    Returns:
        The bare digit strings it contains.
    """
    return {match.replace(",", "") for match in re.findall(r"\d[\d,]*", text)}


#: The whole vocabulary. A case using a key not in here is rejected when the
#: case file loads, rather than silently asserting nothing -- an eval that
#: quietly checks less than it claims is worse than no eval.
CHECKS = {
    "calls_tool": calls_tool,
    "avoids_tool": avoids_tool,
    "calls_before": calls_before,
    "tool_args": tool_args,
    "result_filters": result_filters,
    "answer_contains_any": answer_contains_any,
    "recovers_from_error": recovers_from_error,
    "no_error": no_error,
    "answer_contains": answer_contains,
    "answer_omits": answer_omits,
    "grounded_numbers": grounded_numbers,
}


def run_checks(transcript: Transcript, expect: dict[str, Any]) -> list[CheckResult]:
    """Run every check a case declares.

    Args:
        transcript: The run to grade.
        expect: The case's ``expect`` block.

    Returns:
        Every check result, in declaration order.

    Raises:
        KeyError: If the case names a check that does not exist.
    """
    results: list[CheckResult] = []
    for name, spec in expect.items():
        if name not in CHECKS:
            raise KeyError(f"unknown check {name!r}; known: {sorted(CHECKS)}")
        results.extend(CHECKS[name](transcript, spec))
    return results


__all__ = [
    "CHECKS",
    "MAX_SERIES",
    "MIN_GROUNDED_DIGITS",
    "CheckResult",
    "ToolCall",
    "Transcript",
    "run_checks",
]
