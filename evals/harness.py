"""Drive a real model against the real server, and grade what it did.

**A manual tool-use loop, not the SDK's tool runner.** The runner would be less
code and would hide the thing being measured: this harness has to observe every
tool call, its arguments, and whether the server raised, because that transcript
*is* the grade. Owning the loop is the point.

**The server is the real one, in-process.** ``create_app()`` with FastMCP's
in-memory client -- no HTTP, no deployment, no separate process -- so a run
exercises the same tools, the same envelopes and the same errors that a
deployed client would meet. It does need the stores: a loaded Postgres and a
built rollup database, exactly as the server itself does.

**Errors are handed back to the model, not raised.** A teaching error arrives as
a ``tool_result`` with ``is_error: true``, which is what MCP clients do and what
gives the self-correcting loop somewhere to happen. A harness that aborted on
the first error would make the teaching-error affordance untestable.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from evals.checks import CheckResult, ToolCall, Transcript, run_checks

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: Where the cases live.
CASES_PATH = Path(__file__).parent / "cases.yaml"

#: The model under evaluation, overridable with ``EVAL_MODEL``.
#:
#: This measures the *tool surface*, so the model is an instrument rather than
#: the subject. Two consequences worth stating, because a switchable instrument
#: invites both mistakes:
#:
#: * **A scorecard is only comparable to another on the same model.** Which one
#:   ran is recorded in the header and in the JSON for exactly that reason.
#: * **Thinking is left unconfigured**, which is not the same thing on every
#:   model. Opus 5 and Sonnet 5 run adaptive thinking when the parameter is
#:   omitted; Haiku 4.5 predates that and runs with no thinking at all unless
#:   given an explicit ``budget_tokens``. A Haiku run is therefore measuring
#:   something meaningfully different, not just something cheaper.
DEFAULT_MODEL = "claude-opus-5"

#: Cap on model turns per question. Reached only by a model that is looping;
#: the longest healthy run in this suite is a resolve, a schema read, a retry
#: after a teaching error, and the query.
MAX_TURNS = 12

#: Enough for a data answer with its caveats, and not so much that a runaway
#: case is expensive.
MAX_TOKENS = 4096

#: No system prompt beyond what the server itself publishes. The server's
#: ``INSTRUCTIONS`` and its tool descriptions are the thing under test; adding
#: guidance here would grade a prompt this project does not ship.
SYSTEM = (
    "You are answering questions about Chicago crime data using the tools "
    "provided. Answer the question that was asked, and state any qualification "
    "the tool results attach to the answer."
)


@dataclass(frozen=True)
class Case:
    """One question and what a correct run looks like.

    Attributes:
        id: Stable identifier, used in the scorecard.
        question: The question, phrased as a person would ask it.
        expect: The checks to run, keyed by name.
        affordance: Which of the five this case is trying to break.
        expected_failure: Kept because it is hard, not because it passes.
            Reported separately and never fails the run.
    """

    id: str
    question: str
    expect: dict[str, Any]
    affordance: str = "unspecified"
    expected_failure: bool = False


@dataclass
class CaseResult:
    """One case, run and graded.

    Attributes:
        case: The case.
        transcript: What the model did.
        checks: Every check result.
    """

    case: Case
    transcript: Transcript
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        """Whether every check held."""
        return all(check.passed for check in self.checks)


def load_cases(path: Path | None = None) -> list[Case]:
    """Read and validate the case file.

    Validation is not a formality. A case naming a check that does not exist
    would otherwise assert nothing at all and report a pass, which is worse than
    having no eval: it is a green light for an untested claim.

    Args:
        path: The case file. Defaults to :data:`CASES_PATH`.

    Returns:
        The cases, in file order.

    Raises:
        ValueError: If an id is duplicated or a case declares no checks.
        KeyError: If a case names a check outside the vocabulary.
    """
    from evals.checks import CHECKS

    raw = yaml.safe_load((path or CASES_PATH).read_text(encoding="utf-8"))
    cases = [Case(**entry) for entry in raw]

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id!r}")
        seen.add(case.id)
        if not case.expect:
            raise ValueError(f"case {case.id!r} declares no checks")
        for name in case.expect:
            if name not in CHECKS:
                raise KeyError(f"case {case.id!r} uses unknown check {name!r}")
    return cases


def to_anthropic_tools(mcp_tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert the server's MCP tool listing into Anthropic tool definitions.

    A near-identity mapping, and that is the interesting part: MCP publishes a
    name, a description and a JSON Schema, which is exactly what the Messages
    API wants. The descriptions the model reads here are the tool docstrings --
    written for a model to act on, which is why nothing is rewritten in between.

    Args:
        mcp_tools: What ``Client.list_tools()`` returned.

    Returns:
        Tool definitions for the Messages API.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }
        for tool in mcp_tools
    ]


def resolve_model() -> str:
    """Return the model this run will use.

    One function so the scorecard header and the request cannot disagree about
    which model produced the results being reported.

    Returns:
        ``EVAL_MODEL`` if set, else :data:`DEFAULT_MODEL`.
    """
    return os.environ.get("EVAL_MODEL", DEFAULT_MODEL)


async def run_case(client: Any, mcp: Any, tools: list[dict[str, Any]], case: Case) -> Transcript:
    """Run one question to completion and return the transcript.

    Args:
        client: An ``anthropic.AsyncAnthropic``.
        mcp: An open FastMCP ``Client`` for the server under test.
        tools: The tool definitions from :func:`to_anthropic_tools`.
        case: The case to run.

    Returns:
        Everything the run produced.
    """
    transcript = Transcript(case_id=case.id, question=case.question)
    messages: list[dict[str, Any]] = [{"role": "user", "content": case.question}]
    usage = {"input_tokens": 0, "output_tokens": 0}

    while transcript.turns < MAX_TURNS:
        response = await client.messages.create(
            model=resolve_model(),
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )
        transcript.turns += 1
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            transcript.answer = "\n".join(
                block.text for block in response.content if block.type == "text"
            )
            break

        messages.append({"role": "assistant", "content": response.content})
        # Every tool_result for one assistant turn goes back in a single user
        # message. Splitting them teaches the model to stop calling in parallel.
        results = []
        for block in (b for b in response.content if b.type == "tool_use"):
            call, payload = await _invoke(mcp, block.name, block.input)
            transcript.calls.append(call)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": payload,
                    "is_error": not call.ok,
                }
            )
        messages.append({"role": "user", "content": results})

    transcript.usage = usage
    return transcript


async def _invoke(mcp: Any, name: str, arguments: dict[str, Any]) -> tuple[ToolCall, str]:
    """Call one tool, capturing the outcome rather than letting it propagate.

    A teaching error is a result to hand back, not a failure to abort on -- the
    model is supposed to read it and retry, and that recovery is what several
    cases are checking for.

    Args:
        mcp: The open FastMCP client.
        name: The tool to call.
        arguments: The model's arguments.

    Returns:
        The observed call, and the text to return to the model.
    """
    try:
        result = await mcp.call_tool(name, arguments)
    except Exception as exc:
        return ToolCall(name=name, arguments=arguments, ok=False, error=str(exc)), str(exc)

    body = getattr(result, "structured_content", None) or {}
    payload = json.dumps(body, default=str)
    return ToolCall(name=name, arguments=arguments, ok=True, result=body), payload


async def preflight() -> list[str]:
    """Check everything the suite needs except the model, and report.

    Worth its own mode because every prerequisite here fails in a way that looks
    like a bad eval result rather than a broken setup: an unloaded database
    yields empty answers, a missing tool description yields poor tool choice.
    Costs nothing and makes no API call.

    Returns:
        One line per finding, problems prefixed ``FAIL``.
    """
    from fastmcp import Client

    from chicago_crime_mcp.server.app import create_app

    findings: list[str] = []
    cases = load_cases()
    findings.append(f"ok    {len(cases)} case(s) load and validate")

    async with Client(create_app()) as mcp:
        tools = to_anthropic_tools(await mcp.list_tools())
        findings.append(f"ok    {len(tools)} tool(s): {', '.join(t['name'] for t in tools)}")
        for tool in tools:
            # Only the description is required of every tool. An empty
            # properties map is correct for a tool that takes no arguments --
            # describe_schema is exactly that -- so a missing schema is only
            # wrong when the schema itself is malformed.
            if not tool["description"]:
                findings.append(f"FAIL  {tool['name']} has no description for the model to read")
            if tool["input_schema"].get("type") != "object":
                findings.append(f"FAIL  {tool['name']} publishes a malformed argument schema")

        # The stores answer, and the teaching-error path still teaches. Both are
        # prerequisites several cases silently depend on, and both fail in ways
        # that look like a poor eval result rather than a broken setup.
        call, _ = await _invoke(mcp, "describe_schema", {})
        if not call.ok:
            findings.append(f"FAIL  describe_schema raised: {call.error}")
        else:
            coverage = (call.result or {}).get("provenance", {})
            findings.append(
                f"ok    stores answered; data covers "
                f"{coverage.get('coverage_start', '?')} to {coverage.get('coverage_end', '?')} "
                f"({coverage.get('rows', '?')} rows)"
            )
        probe = {"types": ["BATERY"], "start": "2024-01-01", "end": "2024-02-01"}
        bad, message = await _invoke(mcp, "search_incidents", probe)
        if bad.ok:
            findings.append("FAIL  an invented category was accepted instead of taught")
        elif "BATTERY" in (message or ""):
            findings.append("ok    teaching errors reach the caller with a suggestion")
        else:
            findings.append(f"FAIL  error carried no suggestion: {message[:80]}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        findings.append(
            "note  ANTHROPIC_API_KEY is unset; the SDK will fall back to an "
            "`ant auth login` profile if one exists"
        )
    return findings


async def run_suite(cases: Sequence[Case]) -> list[CaseResult]:
    """Run every case against a freshly built server.

    Args:
        cases: What to run.

    Returns:
        One graded result per case, in order.
    """
    import anthropic
    from fastmcp import Client

    from chicago_crime_mcp.server.app import create_app

    client = anthropic.AsyncAnthropic()
    app = create_app()
    results: list[CaseResult] = []

    async with Client(app) as mcp:
        tools = to_anthropic_tools(await mcp.list_tools())
        for case in cases:
            transcript = await run_case(client, mcp, tools, case)
            results.append(
                CaseResult(
                    case=case,
                    transcript=transcript,
                    checks=run_checks(transcript, case.expect),
                )
            )
    return results


__all__ = [
    "CASES_PATH",
    "DEFAULT_MODEL",
    "MAX_TOKENS",
    "MAX_TURNS",
    "SYSTEM",
    "Case",
    "CaseResult",
    "load_cases",
    "preflight",
    "resolve_model",
    "run_case",
    "run_suite",
    "to_anthropic_tools",
]
