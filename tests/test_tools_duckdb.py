"""Tests for the two DuckDB-backed tools: describe_schema and aggregate_incidents.

DuckDB is embedded, so these are unit tests over a temp-directory fixture.

The load-bearing assertions are the ones about the contract with the model
rather than about SQL, which ``test_duckdb_queries`` already covers: that an
invalid value produces an error naming the field and offering a correction, that
a valid-but-odd value is normalized instead of silently matching nothing, and
that the three taxonomy obligations hold.

Filter tests here follow the rule in ``tests.helpers.row``: the fixture carries
**negative rows** for every dimension a test filters on, and assertions are
relational (``filtered < unfiltered``) so a test cannot pass with the predicate
removed.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from chicago_crime_mcp.server.context import ServerContext, use_context
from chicago_crime_mcp.server.errors import InvalidArgumentError, UnknownValueError
from chicago_crime_mcp.server.tools import aggregate_incidents, describe_schema
from chicago_crime_mcp.store.config import StoreConfig
from chicago_crime_mcp.store.duckdb import rollups
from tests.helpers import row as _row
from tests.helpers import write_partition as _write_partition


@pytest.fixture
def context(tmp_path):
    """A DuckDB-only context over a fixture spanning three years.

    Every dimension a test filters on takes at least two values: two districts
    (010/011), two community areas (1/2), two source categories (BATTERY/THEFT)
    and -- via a curated remap on id=9 -- a comparable category that differs
    from its source one. `0810` appears only in 2026, so the coverage detector
    has an onset to find.
    """
    _write_partition(
        tmp_path / "parquet",
        2024,
        [
            _row(id=1, date=datetime(2024, 1, 5), community_area=1, arrest=True),
            _row(id=2, date=datetime(2024, 1, 20), community_area=1),
            _row(id=3, date=datetime(2024, 2, 10), community_area=2, district="011",
                 beat="1111", domestic=False),
            _row(id=4, date=datetime(2024, 3, 15), community_area=2, district="011",
                 beat="1111", primary_type_canonical="THEFT", stable_category="THEFT",
                 iucr="0820", domestic=False),
        ],
    )
    _write_partition(
        tmp_path / "parquet",
        2025,
        [_row(id=5, date=datetime(2025, 6, 1), community_area=1, arrest=True, domestic=False)],
    )
    _write_partition(
        tmp_path / "parquet",
        2026,
        [
            # `0810` exists only here: an onset the coverage report should find.
            _row(id=9, date=datetime(2026, 2, 4), iucr="0810", community_area=1,
                 primary_type_canonical="BURGLARY", stable_category="THEFT", domestic=False),
        ],
    )
    path = tmp_path / "db" / "crime.duckdb"
    conn = rollups.connect(duckdb_path=path, parquet_root=tmp_path / "parquet")
    rollups.build(conn)
    conn.close()

    ctx = ServerContext(StoreConfig(duckdb_path=path, parquet_root=tmp_path / "parquet"))
    ctx._open_duckdb()
    with use_context(ctx):
        yield ctx
    ctx.close()


# --- describe_schema --------------------------------------------------------


def test_describe_schema_publishes_the_valid_values(context):
    """Affordance #1: the closed sets come from the data, not from a constant."""
    schema = describe_schema()
    source = next(t for t in schema.taxonomies if t.taxonomy == "source")
    comparable = next(t for t in schema.taxonomies if t.taxonomy == "comparable")
    assert source.categories == ["BATTERY", "BURGLARY", "THEFT"]
    # BURGLARY's only row is remapped to THEFT, so it is not a comparable category.
    assert comparable.categories == ["BATTERY", "THEFT"]
    assert source.column == "primary_type_canonical"
    assert comparable.column == "stable_category"


def test_describe_schema_publishes_geography_values_and_stability(context):
    """The values a filter will accept, and which geography is safe over time."""
    schema = describe_schema()
    by_name = {g.name: g for g in schema.geographies}
    assert by_name["district"].values == ["010", "011"]
    assert by_name["community_area"].values == [1, 2]
    assert by_name["district"].value_count == 2
    # The same fact the boundary warning fires on, published before it is needed.
    assert by_name["community_area"].stable_over_time is True
    assert by_name["ward"].stable_over_time is False


def test_describe_schema_reports_coverage_and_limits(context):
    """The window and the caps, so a caller plans inside them."""
    schema = describe_schema()
    assert schema.provenance.coverage_start == date(2024, 1, 5)
    assert schema.provenance.rows == 6
    assert schema.limits.max_rows_per_call > 0
    assert schema.limits.max_radius_m > 0
    assert schema.grains == ["month", "quarter", "year"]


def test_describe_schema_reports_geocoding_by_type(context):
    """Published because it decides whether a radius count and an aggregate compare."""
    schema = describe_schema()
    assert 0.0 <= schema.geocoding.overall_rate <= 1.0
    assert {t.category for t in schema.geocoding.by_type} == {"BATTERY", "BURGLARY", "THEFT"}
    assert all(0.0 <= t.rate <= 1.0 for t in schema.geocoding.by_type)


def test_describe_schema_follows_a_rebuild(context, tmp_path):
    """It must not serve a vocabulary from a build that has been replaced."""
    import os

    _write_partition(tmp_path / "next" / "parquet", 2024,
                     [_row(id=1, date=datetime(2024, 1, 5), primary_type_canonical="ARSON",
                           stable_category="ARSON")])
    new_path = tmp_path / "next" / "db" / "crime.duckdb"
    conn = rollups.connect(duckdb_path=new_path, parquet_root=tmp_path / "next" / "parquet")
    rollups.build(conn)
    conn.close()
    os.replace(new_path, context.config.duckdb_path)

    source = next(t for t in describe_schema().taxonomies if t.taxonomy == "source")
    assert source.categories == ["ARSON"]


# --- teaching errors --------------------------------------------------------


def test_unknown_category_names_the_field_and_suggests(context):
    """Affordance #2: the error is a step in the conversation, not the end of it."""
    with pytest.raises(UnknownValueError) as exc:
        aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31), types=["BATERY"])
    rendered = str(exc.value)
    assert exc.value.field == "types"
    assert exc.value.nearest_match == "BATTERY"
    assert "Did you mean 'BATTERY'?" in rendered
    # The suggestion comes before the inventory, so a model reading one line
    # further than it needs to already has the answer.
    assert rendered.index("Did you mean") < rendered.index("Valid values")


def test_one_bad_value_among_good_ones_is_still_rejected(context):
    """The silent-partial-filter failure: an OR list hides a typo behind a hit.

    Without this check, ``["BATTERY", "BATERY"]`` returns a confident non-empty
    answer with half the filter quietly dropped, which the caller cannot detect.
    """
    with pytest.raises(UnknownValueError) as exc:
        aggregate_incidents(
            start=date(2024, 1, 1), end=date(2024, 12, 31), types=["BATTERY", "BATERY"]
        )
    assert exc.value.received == "BATERY"


def test_category_validity_depends_on_the_taxonomy(context):
    """BURGLARY exists under 'source' and not under 'comparable'."""
    aggregate_incidents(start=date(2026, 1, 1), end=date(2026, 12, 31), types=["BURGLARY"])
    with pytest.raises(UnknownValueError) as exc:
        aggregate_incidents(
            start=date(2026, 1, 1), end=date(2026, 12, 31),
            types=["BURGLARY"], taxonomy="comparable",
        )
    assert "comparable" in str(exc.value)


def test_unknown_geography_value_is_rejected(context):
    """A district that does not exist is an error, not an empty answer."""
    with pytest.raises(UnknownValueError) as exc:
        aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31),
                            geography="district", geography_values=["099"])
    assert exc.value.field == "geography_values"


def test_non_numeric_ward_names_the_field(context):
    """A coercion failure becomes a teaching error rather than a stack trace."""
    with pytest.raises(InvalidArgumentError) as exc:
        aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31),
                            geography="ward", geography_values=["downtown"])
    assert exc.value.field == "geography_values"


def test_inverted_range_names_the_end_argument(context):
    with pytest.raises(InvalidArgumentError) as exc:
        aggregate_incidents(start=date(2024, 12, 31), end=date(2024, 1, 1))
    assert exc.value.field == "end"


def test_limit_out_of_range_is_rejected(context):
    with pytest.raises(InvalidArgumentError) as exc:
        aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31), limit=99_999)
    assert exc.value.field == "limit"


# --- normalization ----------------------------------------------------------


def test_unpadded_district_is_normalized_and_actually_filters(context):
    """A caller passing 11 means '011', and the filter must genuinely exclude.

    Asserted relationally against the unfiltered total: an absolute count cannot
    distinguish "the filter matched everything" from "the filter was dropped".
    """
    unfiltered = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31),
                                     geography="district", breakdown_by_type=False)
    filtered = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31),
                                   geography="district", geography_values=[11],
                                   breakdown_by_type=False)
    total = sum(b.incidents for b in unfiltered.data.buckets)
    kept = sum(b.incidents for b in filtered.data.buckets)
    assert 0 < kept < total
    # And the echo shows the form the data was really filtered on.
    assert filtered.filters_applied.geography_values == ["011"]


def test_lowercase_category_is_upper_cased_and_filters(context):
    unfiltered = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31))
    filtered = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31),
                                   types=["theft"])
    total = sum(b.incidents for b in unfiltered.data.buckets)
    kept = sum(b.incidents for b in filtered.data.buckets)
    assert 0 < kept < total
    assert filtered.filters_applied.types == ["THEFT"]


# --- the taxonomy obligations ----------------------------------------------


@pytest.mark.parametrize("taxonomy", ["source", "comparable"])
def test_envelope_always_names_the_taxonomy_mode(context, taxonomy):
    """Obligation 2: in both modes, including the default.

    Silently returning re-grouped counts is the same class of failure as
    silently truncating: the caller cannot reason about a transformation it was
    not told about.
    """
    result = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31),
                                 taxonomy=taxonomy)
    assert result.taxonomy_mode == taxonomy


def test_taxonomy_defaults_to_source(context):
    """Obligation 1: never inferred from the question, never auto-switched."""
    result = aggregate_incidents(start=date(2024, 1, 1), end=date(2026, 12, 31), grain="year")
    assert result.taxonomy_mode == "source"


def test_taxonomy_changes_which_column_is_grouped(context):
    """The remapped row lands under BURGLARY in one mode and THEFT in the other."""
    source = aggregate_incidents(start=date(2026, 1, 1), end=date(2026, 12, 31))
    comparable = aggregate_incidents(start=date(2026, 1, 1), end=date(2026, 12, 31),
                                     taxonomy="comparable")
    assert [b.category for b in source.data.buckets] == ["BURGLARY"]
    assert [b.category for b in comparable.data.buckets] == ["THEFT"]


@pytest.mark.parametrize("taxonomy", ["source", "comparable"])
def test_coverage_warning_fires_in_both_modes(context, taxonomy):
    """Obligation 3: 'comparable' corrects only curated drift, so it never suppresses.

    The span covers `0810`'s onset, so a share of the rows comes from a code
    that does not cover the period in either mode.
    """
    result = aggregate_incidents(start=date(2024, 1, 1), end=date(2026, 12, 31),
                                 grain="year", taxonomy=taxonomy)
    codes = [w for w in result.warnings if w.code == "code_coverage_drift"]
    assert codes, f"no coverage warning under taxonomy={taxonomy}"
    assert codes[0].detail["affected_share"] > 0


def test_coverage_warning_remedy_depends_on_the_mode(context):
    """It suggests switching only to a caller who has not already switched."""
    source = aggregate_incidents(start=date(2024, 1, 1), end=date(2026, 12, 31), grain="year")
    comparable = aggregate_incidents(start=date(2024, 1, 1), end=date(2026, 12, 31),
                                     grain="year", taxonomy="comparable")
    source_msg = next(w.message for w in source.warnings if w.code == "code_coverage_drift")
    comparable_msg = next(
        w.message for w in comparable.warnings if w.code == "code_coverage_drift"
    )
    assert "taxonomy='comparable'" in source_msg
    assert "already the comparable taxonomy" in comparable_msg


def test_taxonomy_mode_is_not_buried_in_the_filters(context):
    """It is not a predicate -- it changes what the category column means."""
    result = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31),
                                 taxonomy="comparable")
    assert "taxonomy" not in result.filters_applied.model_dump()


# --- the other warnings -----------------------------------------------------


def test_provisional_fires_on_a_still_filling_period(context):
    """The last bucket runs past the newest incident, so it will grow."""
    result = aggregate_incidents(start=date(2026, 1, 1), end=date(2026, 12, 31), grain="year")
    assert any(w.code == "provisional" for w in result.warnings)


def test_partial_period_fires_on_a_mid_month_span(context):
    result = aggregate_incidents(start=date(2024, 1, 10), end=date(2024, 3, 20))
    warning = next(w for w in result.warnings if w.code == "partial_period")
    assert warning.detail == {"partial_first": True, "partial_last": True, "grain": "month"}


def test_boundary_warning_fires_for_ward_across_years_only(context):
    """Unstable geography plus a multi-year span; neither alone."""
    multi = aggregate_incidents(start=date(2024, 1, 1), end=date(2026, 12, 31),
                                grain="year", geography="ward")
    single = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 12, 31),
                                 geography="ward")
    stable = aggregate_incidents(start=date(2024, 1, 1), end=date(2026, 12, 31),
                                 grain="year", geography="community_area")
    assert any(w.code == "boundary_instability" for w in multi.warnings)
    assert not any(w.code == "boundary_instability" for w in single.warnings)
    assert not any(w.code == "boundary_instability" for w in stable.warnings)


def test_empty_result_is_answered_not_failed(context):
    """A valid filter matching nothing is an answer, and the warning says so."""
    result = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 1, 2),
                                 types=["THEFT"])
    assert result.row_count == 0
    warning = next(w for w in result.warnings if w.code == "empty_result")
    assert warning.detail["filters_applied"] == ["types"]


def test_truncation_is_flagged_not_hidden(context):
    result = aggregate_incidents(start=date(2024, 1, 1), end=date(2026, 12, 31), limit=1)
    assert result.truncated is True
    assert result.row_count == 1
    assert any(w.code == "truncated" for w in result.warnings)


# --- routing ----------------------------------------------------------------


def test_route_reports_the_tier_and_why(context):
    """A month-aligned span sums pre-summed months; an unaligned one scans."""
    aligned = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 3, 31))
    unaligned = aggregate_incidents(start=date(2024, 1, 10), end=date(2024, 3, 20))
    assert aligned.route.store == "duckdb"
    assert aligned.route.tier == "rollup"
    assert unaligned.route.tier == "scan"
    assert aligned.route.reason and unaligned.route.reason


def test_both_tiers_agree(context):
    """The routing story rests on this: the fallback is not an approximation."""
    aligned = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 3, 31),
                                  breakdown_by_type=False)
    # Same span, expressed so the router cannot use the rollup.
    scanned = aggregate_incidents(start=date(2024, 1, 1), end=date(2024, 3, 30),
                                  breakdown_by_type=False)
    assert aligned.route.tier == "rollup"
    assert scanned.route.tier == "scan"
    assert sum(b.incidents for b in aligned.data.buckets) == 4
    assert sum(b.incidents for b in scanned.data.buckets) == 4
