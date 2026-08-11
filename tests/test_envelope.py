"""Tests for the result envelope and the warning wording layer.

The store dataclasses are used directly rather than stand-ins: the whole point
of this layer is that it maps onto *those* shapes, and a hand-rolled fake would
keep passing after a field was renamed underneath it.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel

from chicago_crime_mcp.server.envelope import (
    COVERAGE_WARNING_THRESHOLD,
    Provenance,
    ResultWarning,
    RouteInfo,
    ToolResult,
    boundary_warning,
    coverage_warning,
    empty_result_warning,
    multiple_matches_warning,
    partial_period_warning,
    provisional_warning,
    truncated_warning,
    ungeocoded_warning,
)
from chicago_crime_mcp.store.duckdb.queries import (
    CodeCoverage,
    CoverageReport,
    DatasetMeta,
    Route,
)
from chicago_crime_mcp.store.postgres.queries import Timing


def _coverage(*, affected: int, total: int, codes: int = 1) -> CoverageReport:
    code = CodeCoverage(
        iucr="0760",
        description="BURGLARY FROM MOTOR VEHICLE",
        category="BURGLARY",
        first_month=date(2021, 11, 1),
        last_month=date(2026, 7, 1),
        incidents=affected,
        enters=True,
        exits=False,
    )
    return CoverageReport(
        codes=(code,), code_count=codes, affected_incidents=affected, total_incidents=total
    )


class _Filters(BaseModel):
    kind: str


# --- routing -----------------------------------------------------------------


def test_a_duckdb_route_keeps_its_tier_and_table():
    route = Route(
        tier="rollup", table="rollup_ward", reason="span is month-aligned", elapsed_ms=0.7
    )
    info = RouteInfo.from_store(route, store="duckdb")
    assert (info.store, info.tier, info.table) == ("duckdb", "rollup", "rollup_ward")
    assert info.reason == "span is month-aligned"


def test_a_postgres_timing_leaves_the_narrower_fields_empty_rather_than_invented():
    timing = Timing(store="postgres", reason="radius query", elapsed_ms=28.1234)
    info = RouteInfo.from_store(timing, store="postgres")
    assert (info.tier, info.table) == (None, None)
    assert info.reason == "radius query"


def test_elapsed_is_rounded_so_the_envelope_does_not_imply_nanosecond_precision():
    info = RouteInfo.from_store(Timing(store="postgres", reason="x", elapsed_ms=1.23456789),
                                store="postgres")
    assert info.elapsed_ms == 1.235


# --- warnings that decide whether to fire ------------------------------------


def test_boundary_warning_fires_for_a_moving_geography_across_years():
    warning = boundary_warning(geography="ward", start=date(2015, 1, 1), end=date(2026, 6, 30))
    assert warning is not None
    assert warning.code == "boundary_instability"
    assert "community_area" in warning.message


def test_boundary_warning_is_silent_for_the_one_stable_geography():
    assert boundary_warning(
        geography="community_area", start=date(2015, 1, 1), end=date(2026, 6, 30)
    ) is None


def test_boundary_warning_is_silent_within_a_single_year():
    warning = boundary_warning(geography="ward", start=date(2025, 1, 1), end=date(2025, 12, 31))
    assert warning is None


def test_boundary_warning_is_silent_citywide():
    warning = boundary_warning(geography="citywide", start=date(2015, 1, 1), end=date(2026, 1, 1))
    assert warning is None


def test_coverage_warning_fires_above_the_threshold():
    warning = coverage_warning(_coverage(affected=8475, total=117340), taxonomy="source")
    assert warning is not None
    assert warning.code == "code_coverage_drift"
    assert warning.detail["affected_share"] == 0.0722


def test_coverage_warning_stays_quiet_on_a_negligible_share():
    tiny = _coverage(affected=1, total=1_000_000)
    assert tiny.affected_share < COVERAGE_WARNING_THRESHOLD
    assert coverage_warning(tiny, taxonomy="source") is None


def test_coverage_warning_stays_quiet_when_no_code_drifted():
    clean = CoverageReport(codes=(), code_count=0, affected_incidents=0, total_incidents=50_000)
    assert coverage_warning(clean, taxonomy="source") is None


def test_coverage_warning_fires_in_comparable_mode_too():
    # `comparable` corrects only the drift someone curated, so it is never a
    # reason to suppress this.
    warning = coverage_warning(_coverage(affected=8475, total=117340), taxonomy="comparable")
    assert warning is not None


def test_only_source_mode_is_told_to_try_the_comparable_taxonomy():
    report = _coverage(affected=8475, total=117340)
    assert "taxonomy='comparable'" in coverage_warning(report, taxonomy="source").message
    assert "taxonomy='comparable'" not in coverage_warning(report, taxonomy="comparable").message


def test_the_coverage_share_is_labelled_citywide_because_the_source_table_is():
    warning = coverage_warning(_coverage(affected=8475, total=117340), taxonomy="source")
    assert "citywide" in warning.message
    assert warning.detail["scope"] == "citywide"


# --- warnings that only phrase ------------------------------------------------


def test_provisional_names_both_the_bucket_and_the_data_boundary():
    warning = provisional_warning(last_period=date(2026, 1, 1), coverage_end=date(2026, 7, 22))
    assert warning.code == "provisional"
    assert "2026-01-01" in warning.message and "2026-07-22" in warning.message


def test_partial_period_wording_matches_which_edge_is_actually_partial():
    assert "ends mid-year" in partial_period_warning(first=False, last=True, grain="year").message
    assert "starts mid-year" in partial_period_warning(first=True, last=False, grain="year").message
    both = partial_period_warning(first=True, last=True, grain="month").message
    assert "starts and ends mid-month" in both


def test_truncation_points_at_the_cursor_only_when_there_is_one():
    assert "cursor" in truncated_warning(returned=50, has_cursor=True).message
    assert "cursor" not in truncated_warning(returned=50, has_cursor=False).message


def test_ungeocoded_names_the_worst_type_because_the_bias_is_by_type_not_place():
    warning = ungeocoded_warning(
        rates_by_type={"THEFT": 0.981, "OFFENSE INVOLVING CHILDREN": 0.915}, overall_rate=0.9845
    )
    assert warning.code == "excludes_ungeocoded"
    assert "OFFENSE INVOLVING CHILDREN" in warning.message
    assert "by offense type, not by place" in warning.message


def test_ungeocoded_still_states_the_bare_fact_with_no_rates_available():
    warning = ungeocoded_warning()
    assert "excludes un-geocoded offenses" in warning.message
    assert warning.detail == {}


def test_multiple_matches_explains_why_a_case_number_is_not_a_key():
    warning = multiple_matches_warning(case_number="JJ195924", count=3)
    assert warning.detail["count"] == 3
    assert "Only id is unique" in warning.message


def test_empty_result_distinguishes_an_answer_from_a_failure():
    warning = empty_result_warning(filters=("types", "geography_values"))
    assert warning.code == "empty_result"
    assert "answer rather than a" in warning.message
    assert warning.detail["filters_applied"] == ["types", "geography_values"]


# --- the envelope itself ------------------------------------------------------


class _Row(BaseModel):
    n: int


def test_the_envelope_is_generic_over_both_payload_and_filters():
    result = ToolResult[list[_Row], _Filters](
        data=[_Row(n=1)],
        filters_applied=_Filters(kind="search"),
        row_count=1,
        route=RouteInfo(store="postgres", reason="x", elapsed_ms=1.0),
    )
    assert result.data[0].n == 1
    assert result.filters_applied.kind == "search"


def test_the_derived_schema_names_the_concrete_payload_and_filter_types():
    # This is what FastMCP publishes: an untyped bag here would tell the model
    # nothing about what comes back.
    schema = ToolResult[list[_Row], _Filters].model_json_schema()
    assert set(schema["$defs"]) >= {"_Row", "_Filters", "RouteInfo", "ResultWarning", "Provenance"}
    assert schema["properties"]["data"]["items"]["$ref"].endswith("_Row")


def test_an_envelope_needs_no_warnings_and_defaults_to_none():
    result = ToolResult[list[_Row], _Filters](
        data=[], filters_applied=_Filters(kind="search"), row_count=0,
        route=RouteInfo(store="duckdb", reason="x", elapsed_ms=1.0),
    )
    assert result.warnings == []
    assert result.truncated is False and result.cursor is None


def test_provenance_defaults_carry_the_source_caveats_without_a_dataset_read():
    provenance = Provenance()
    assert provenance.dataset_id == "ijzp-q8t2"
    assert any("block-level" in c for c in provenance.caveats)
    assert provenance.coverage_end is None


def test_provenance_from_dataset_meta_fills_the_coverage_window():
    meta = DatasetMeta(
        built_at=datetime(2026, 8, 1, 2, 6, 56),
        source_rows=2_884_106,
        partitions=12,
        min_date=datetime(2015, 1, 1),
        max_date=datetime(2026, 7, 22),
    )
    provenance = Provenance.from_dataset_meta(meta)
    assert provenance.coverage_start == date(2015, 1, 1)
    assert provenance.coverage_end == date(2026, 7, 22)
    assert provenance.rows == 2_884_106


def test_warnings_serialize_with_their_code_so_a_client_can_branch_on_it():
    warning = ResultWarning(code="provisional", message="m", detail={"k": 1})
    assert warning.model_dump(mode="json") == {
        "code": "provisional",
        "message": "m",
        "detail": {"k": 1},
    }
