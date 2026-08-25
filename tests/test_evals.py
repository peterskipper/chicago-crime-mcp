"""Tests for the eval harness that need no API key.

The live run costs money and needs a loaded database, so it is not a test. But
the two things most likely to be quietly wrong -- a check that does not check
what it says, and a case that asserts nothing -- are pure functions over a
transcript, and those are tested here in full.

The case-file test is the important one. A case naming a check that does not
exist would assert nothing and report a pass, which is worse than having no eval
at all: it is a green light for an untested claim.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import pytest
from evals import checks
from evals.checks import ToolCall, Transcript, run_checks
from evals.harness import Case, load_cases, to_anthropic_tools


def _transcript(calls=(), answer="", question="") -> Transcript:
    """Build a transcript from a terse call spec."""
    return Transcript(
        case_id="t",
        question=question,
        calls=[
            ToolCall(
                name=name,
                arguments=args or {},
                ok=ok,
                error=None if ok else "teaching message",
                # Kept even on a failed call, deliberately: the filtering is
                # Transcript.results_blob's job, and nulling it here would make
                # the "an error message is not a source" test pass vacuously.
                result=result,
            )
            for name, args, ok, result in calls
        ],
        answer=answer,
    )


def _passed(results) -> bool:
    """Whether every check in a list held."""
    return all(r.passed for r in results)


# --- tool choice -------------------------------------------------------------


def test_calls_tool_holds_when_called_and_fails_when_not():
    transcript = _transcript([("aggregate_incidents", {}, True, {})])
    assert _passed(checks.calls_tool(transcript, ["aggregate_incidents"]))
    assert not _passed(checks.calls_tool(transcript, ["search_incidents"]))


def test_avoids_tool_is_the_mirror_of_calls_tool():
    transcript = _transcript([("search_incidents", {}, True, {})])
    assert _passed(checks.avoids_tool(transcript, ["aggregate_incidents"]))
    assert not _passed(checks.avoids_tool(transcript, ["search_incidents"]))


def test_calls_before_requires_the_order_not_just_the_presence():
    """Both called is not enough -- resolving after filtering is luck."""
    resolve = ("resolve_neighborhood", {}, True, {})
    aggregate = ("aggregate_incidents", {}, True, {})
    right = _transcript([resolve, aggregate])
    wrong = _transcript([aggregate, resolve])
    pair = ["resolve_neighborhood", "aggregate_incidents"]
    assert _passed(checks.calls_before(right, pair))
    assert not _passed(checks.calls_before(wrong, pair))


def test_calls_before_fails_when_either_tool_is_missing():
    """The negative half: an absent tool must not vacuously satisfy an ordering."""
    only_one = _transcript([("resolve_neighborhood", {}, True, {})])
    assert not _passed(
        checks.calls_before(only_one, ["resolve_neighborhood", "aggregate_incidents"])
    )


# --- arguments ---------------------------------------------------------------


def test_tool_args_matches_on_any_call_not_every_call():
    """A model that errs, reads the message and retries has done the right thing."""
    transcript = _transcript(
        [
            ("search_incidents", {"district": "1"}, False, None),
            ("search_incidents", {"district": "018"}, True, {}),
        ]
    )
    assert _passed(checks.tool_args(transcript, {"search_incidents": {"district": "018"}}))


def test_tool_args_fails_when_no_call_matches():
    transcript = _transcript([("search_incidents", {"district": "007"}, True, {})])
    assert not _passed(checks.tool_args(transcript, {"search_incidents": {"district": "018"}}))


def test_tool_args_fails_when_the_tool_was_never_called():
    assert not _passed(
        checks.tool_args(_transcript(), {"search_incidents": {"district": "018"}})
    )


# --- errors and recovery -----------------------------------------------------


def test_recovers_from_error_needs_a_later_success_on_the_same_tool():
    recovered = _transcript(
        [("aggregate_incidents", {}, False, None), ("aggregate_incidents", {}, True, {})]
    )
    assert _passed(checks.recovers_from_error(recovered, True))


def test_recovery_on_a_different_tool_is_not_recovery():
    """Giving up and asking something else is not the loop this claims to test."""
    sidestepped = _transcript(
        [("aggregate_incidents", {}, False, None), ("search_incidents", {}, True, {})]
    )
    assert not _passed(checks.recovers_from_error(sidestepped, True))


def test_a_success_before_the_error_does_not_count_as_recovery():
    """Order matters: the successful call has to follow the failure."""
    wrong_order = _transcript(
        [("aggregate_incidents", {}, True, {}), ("aggregate_incidents", {}, False, None)]
    )
    assert not _passed(checks.recovers_from_error(wrong_order, True))


def test_recovers_from_error_fails_when_nothing_ever_failed():
    """It asserts that the teaching loop ran, not merely that things went well."""
    clean = _transcript([("aggregate_incidents", {}, True, {})])
    assert not _passed(checks.recovers_from_error(clean, True))


def test_no_error_is_the_opposite_check():
    clean = _transcript([("aggregate_incidents", {}, True, {})])
    dirty = _transcript([("aggregate_incidents", {}, False, None)])
    assert _passed(checks.no_error(clean, True))
    assert not _passed(checks.no_error(dirty, True))


# --- the answer text ---------------------------------------------------------


def test_answer_contains_is_case_insensitive():
    transcript = _transcript(answer="This is the Community Area of Douglas.")
    assert _passed(checks.answer_contains(transcript, ["community area"]))


def test_answer_omits_catches_the_confident_wrong_name():
    """The Bronzeville failure: a fluent answer about somewhere 19 km away."""
    bad = _transcript(answer="Andersonville saw 1,200 offenses.")
    good = _transcript(answer="Bronzeville has no boundary of its own.")
    assert not _passed(checks.answer_omits(bad, ["Andersonville"]))
    assert _passed(checks.answer_omits(good, ["Andersonville"]))


# --- grounding ---------------------------------------------------------------


def test_grounded_numbers_passes_when_the_figure_is_in_a_result():
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, True, {"data": {"count": 4821}})],
        answer="There were 4,821 offenses.",
    )
    assert _passed(checks.grounded_numbers(transcript, True))


def test_grounded_numbers_catches_an_invented_figure():
    """The failure that makes a grounded data tool worthless."""
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, True, {"data": {"count": 4821}})],
        answer="There were 7,300 offenses.",
    )
    results = checks.grounded_numbers(transcript, True)
    assert not _passed(results)
    assert "7300" in results[0].detail


def test_grounded_numbers_ignores_small_numbers():
    """'the top 5 categories' is the model counting its own list, not citing."""
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, True, {"rows": []})],
        answer="The top 5 categories, of which 3 are violent.",
    )
    assert _passed(checks.grounded_numbers(transcript, True))


def test_grounded_numbers_accepts_a_figure_taken_from_the_question():
    """A model echoing back coordinates it was given has invented nothing."""
    transcript = _transcript(
        calls=[("nearby_incidents", {}, True, {"rows": []})],
        question="within 500 metres of latitude 41.8781",
        answer="Within 500 metres of 41.8781 there were none.",
    )
    assert _passed(checks.grounded_numbers(transcript, True))


def test_grounded_numbers_does_not_read_a_failed_calls_payload():
    """An error message is not a source; a figure cited from one is unsourced."""
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, False, {"count": 4821})],
        answer="There were 4,821 offenses.",
    )
    assert not _passed(checks.grounded_numbers(transcript, True))


def test_result_filters_reads_the_normalized_value_not_the_typed_one():
    """The whole reason this check exists, taken from a real run.

    A model sent ``geography_values: [18]``; the server zero-padded it to "018"
    and answered correctly. Asserting the wire form failed a run that did
    everything right.
    """
    transcript = _transcript(
        [(
            "search_incidents",
            {"geography": "district", "geography_values": [18]},
            True,
            {"filters_applied": {"geography": "district", "geography_values": ["018"]}},
        )]
    )
    spec = {"search_incidents": {"geography": "district", "geography_values": ["018"]}}
    assert _passed(checks.result_filters(transcript, spec))


def test_result_filters_still_fails_on_the_wrong_district():
    """The negative half: type-tolerance must not become value-tolerance."""
    transcript = _transcript(
        [("search_incidents", {}, True, {"filters_applied": {"geography_values": ["007"]}})]
    )
    assert not _passed(
        checks.result_filters(transcript, {"search_incidents": {"geography_values": ["018"]}})
    )


def test_result_filters_does_not_collapse_zero_padding():
    """'18' and '018' are genuinely different; only int-vs-str is ignored."""
    transcript = _transcript(
        [("search_incidents", {}, True, {"filters_applied": {"geography_values": ["18"]}})]
    )
    assert not _passed(
        checks.result_filters(transcript, {"search_incidents": {"geography_values": ["018"]}})
    )


def test_result_filters_fails_when_the_envelope_echoed_nothing():
    transcript = _transcript([("search_incidents", {}, True, {})])
    assert not _passed(
        checks.result_filters(transcript, {"search_incidents": {"geography": "district"}})
    )


def test_answer_contains_any_accepts_alternative_phrasings():
    """A fact with several natural wordings should not be graded on phrasing."""
    for wording in ("the most recent 7 days", "a 7-day lag", "seven days behind", "about a week"):
        transcript = _transcript(answer=f"The feed excludes {wording}.")
        assert _passed(
            checks.answer_contains_any(
                transcript, ["7 day", "7-day", "seven day", "week", "most recent"]
            )
        ), wording


def test_answer_contains_any_fails_when_none_appear():
    transcript = _transcript(answer="There were 12 robberies.")
    assert not _passed(checks.answer_contains_any(transcript, ["7 day", "week"]))


# --- grounding: derivation is not invention ----------------------------------


def _buckets(values, start_year=2016):
    """A result shaped like an aggregate response.

    Periods are real dates, because that is what the tool returns and it matters
    here: a year mentioned in an answer is a four-digit number like any other,
    and it is grounded precisely when the results (or the question) contain it.
    """
    return {
        "data": {
            "buckets": [
                {"period": f"{start_year + i}-01-01", "incidents": v}
                for i, v in enumerate(values)
            ]
        }
    }


def test_a_year_named_in_the_answer_is_grounded_by_the_periods_returned():
    """A year is a four-digit number; it is sourced when the buckets carry it."""
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, True, _buckets([7052, 6828]))],
        answer="In 2016 there were 7,052 and in 2017 there were 6,828.",
    )
    assert _passed(checks.grounded_numbers(transcript, True))


def test_a_total_summed_from_buckets_is_grounded():
    """The commonest correct thing an answer does, and the original false positive."""
    values = [4081, 4210, 4055, 3990, 4150, 4000]
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, True, _buckets(values))],
        answer=f"Thefts totalled {sum(values):,} over the six months.",
    )
    assert _passed(checks.grounded_numbers(transcript, True))


def test_a_difference_between_two_periods_is_grounded():
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, True, _buckets([7052, 6828]))],
        answer="Ward 3 fell by 224 offenses.",
    )
    assert _passed(checks.grounded_numbers(transcript, True))


def test_a_rounded_percentage_is_grounded():
    """Rounding to a whole percent must not read as invention."""
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, True, _buckets([7052, 6828]))],
        answer="That is 97% of the 2016 level.",
    )
    assert _passed(checks.grounded_numbers(transcript, True))


def test_a_zero_padded_code_returned_as_a_string_is_grounded():
    """IUCR codes arrive as strings; they are figures the model was handed."""
    transcript = _transcript(
        calls=[("describe_schema", {}, True, {"codes": [{"iucr": "0325"}]})],
        answer="IUCR 0325 covers aggravated vehicular hijacking.",
    )
    assert _passed(checks.grounded_numbers(transcript, True))


def test_invention_is_still_caught_among_real_figures():
    """The point of the check survives the loosening."""
    values = [4081, 4210, 4055]
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, True, _buckets(values))],
        answer=f"There were {sum(values):,} thefts, of which 9,912 were on the North Side.",
    )
    results = checks.grounded_numbers(transcript, True)
    assert not _passed(results)
    assert "9912" in results[0].detail


def test_a_long_series_is_not_scanned_pairwise():
    """Guards the loosening from becoming vacuous.

    With enough values almost any number is reachable by some pair, so the
    pairwise scan is capped -- otherwise the check would pass everything.
    """
    long_series = list(range(checks.MAX_SERIES + 20))
    transcript = _transcript(
        calls=[("aggregate_incidents", {}, True, _buckets(long_series))],
        answer="There were 8,675,309 offenses.",
    )
    assert not _passed(checks.grounded_numbers(transcript, True))


# --- the case file -----------------------------------------------------------


def test_the_shipped_cases_load_and_validate():
    """Loading is the validation: unknown checks and empty expects both raise."""
    cases = load_cases()
    assert len(cases) >= 15
    assert all(case.expect for case in cases)


def test_every_shipped_case_runs_against_an_empty_transcript():
    """No case may crash the grader -- a failing check must report, not raise.

    An empty transcript is the worst input a check will ever see: no calls, no
    answer. Every case should grade it (mostly as failures) rather than throw.
    """
    empty = _transcript()
    for case in load_cases():
        results = run_checks(empty, case.expect)
        assert results, f"case {case.id!r} produced no check results"


def test_the_shipped_cases_cover_every_affordance():
    """The suite exists to exercise the five claims, so it must reach all five."""
    affordances = {case.affordance for case in load_cases()}
    assert affordances >= {
        "schema discovery",
        "teaching errors",
        "entity resolution",
        "result envelopes",
        "bounded results",
    }


def test_an_unknown_check_is_rejected_rather_than_silently_ignored(tmp_path):
    """The failure this guards: an eval that quietly checks less than it claims."""
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: x\n  question: q\n  expect:\n    invented_check: true\n", encoding="utf-8"
    )
    with pytest.raises(KeyError, match="invented_check"):
        load_cases(path)


def test_a_case_with_no_checks_is_rejected(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("- id: x\n  question: q\n  expect: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no checks"):
        load_cases(path)


def test_duplicate_case_ids_are_rejected(tmp_path):
    """Ids key the scorecard; two rows with one name is not a report."""
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: x\n  question: q\n  expect:\n    no_error: true\n"
        "- id: x\n  question: r\n  expect:\n    no_error: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_cases(path)


def test_run_checks_rejects_an_unknown_check_at_grading_time_too():
    with pytest.raises(KeyError):
        run_checks(_transcript(), {"nope": True})


# --- the MCP -> Anthropic tool mapping ---------------------------------------


def test_tool_definitions_carry_the_docstring_the_model_reads():
    """Nothing is rewritten in between: the tool docstring *is* the description."""

    class _Tool:
        name = "search_incidents"
        description = "Find offenses matching filters."
        inputSchema = {"type": "object", "properties": {"district": {"type": "string"}}}

    converted = to_anthropic_tools([_Tool()])
    assert converted == [
        {
            "name": "search_incidents",
            "description": "Find offenses matching filters.",
            "input_schema": {"type": "object", "properties": {"district": {"type": "string"}}},
        }
    ]


def test_a_tool_without_a_description_converts_to_an_empty_string():
    """The Messages API requires the key; None would be rejected."""

    class _Tool:
        name = "x"
        description = None
        inputSchema = {"type": "object"}

    assert to_anthropic_tools([_Tool()])[0]["description"] == ""


def test_a_case_is_not_an_expected_failure_by_default():
    """expected_failure must be opted into, or a typo would hide a real break."""
    assert Case(id="x", question="q", expect={"no_error": True}).expected_failure is False
