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
from typing import Any

#: Numbers below this many digits are ignored by :func:`grounded_numbers`.
#: "the top 5 categories" and "3 of them" are the model counting its own list,
#: not citing a figure, and demanding a source for those would fail every
#: correct answer.
MIN_GROUNDED_DIGITS = 3


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

        Used by :func:`grounded_numbers` to ask whether a figure in the answer
        appeared anywhere the model could have read it.

        Returns:
            The concatenated JSON of every successful result.
        """
        return json.dumps([c.result for c in self.calls if c.ok], default=str)


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
    """Require that every substantial figure in the answer came from a result.

    A heuristic, and worth being honest about what it can and cannot do. It
    ignores numbers of fewer than :data:`MIN_GROUNDED_DIGITS` digits (list
    positions and small counts the model derived itself) and numbers already
    present in the question. What is left is the class this is for: a plausible
    four-digit total that appears in no tool result.

    It cannot catch a figure that is real but misattributed, and it will not try
    to. It catches invention, which is the failure that makes a grounded data
    tool worthless.

    Args:
        transcript: The run.
        _spec: Unused; the check is enabled by being listed.

    Returns:
        One result naming any ungrounded figures.
    """
    blob = _digits(transcript.results_blob())
    asked = _digits(transcript.question)
    ungrounded = []
    for number in re.findall(r"\d[\d,]*", transcript.answer):
        bare = number.replace(",", "")
        if len(bare) < MIN_GROUNDED_DIGITS or bare in blob or bare in asked:
            continue
        ungrounded.append(bare)
    return [
        CheckResult(
            name="grounded_numbers",
            passed=not ungrounded,
            detail=(
                f"not found in any tool result: {sorted(set(ungrounded))}"
                if ungrounded
                else "every figure traced to a result"
            ),
        )
    ]


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
    "MIN_GROUNDED_DIGITS",
    "CheckResult",
    "ToolCall",
    "Transcript",
    "run_checks",
]
