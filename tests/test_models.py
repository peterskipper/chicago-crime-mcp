"""Tests for the Pydantic views of the store dataclasses.

Two things are worth guarding here. The first is drift: rows are mapped by
field *name*, so a column added to a store dataclass and forgotten here
disappears silently rather than raising. The second is compaction: the summary
shape is smaller than the full one on purpose, and a well-meaning addition would
quietly undo it.
"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

from chicago_crime_mcp.server.models import (
    AggregateFilters,
    AggregatePayload,
    AggregateResponse,
    IncidentModel,
    IncidentSummaryModel,
    LookupFilters,
    LookupPayload,
    LookupResponse,
    NearbyFilters,
    NearbyPayload,
    NearbyResponse,
    SearchFilters,
    SearchPayload,
    SearchResponse,
)
from chicago_crime_mcp.store.duckdb.queries import AggregateQuery, AggregateRow
from chicago_crime_mcp.store.postgres.queries import (
    INCIDENT_COLUMNS,
    SUMMARY_COLUMNS,
    DistanceRing,
    IncidentSummary,
    LookupQuery,
    NearbyIncident,
    NearbyQuery,
    SearchQuery,
    TypeCount,
)


def _summary(**overrides) -> IncidentSummary:
    fields = {
        "id": 13787325,
        "case_number": "JJ195924",
        "date": datetime(2025, 3, 25, 0, 0),
        "primary_type_canonical": "BURGLARY",
        "stable_category": "BURGLARY",
        "description": "FORCIBLE ENTRY",
        "block": "013XX S WASHTENAW AVE",
        "location_description": "APARTMENT",
        "beat": "1023",
        "district": "010",
        "ward": 28,
        "community_area": 29,
        "arrest": False,
        "domestic": False,
        "latitude": 41.864259582,
        "longitude": -87.693347204,
    }
    return IncidentSummary(**{**fields, **overrides})


def test_the_full_model_mirrors_the_store_row_column_for_column():
    assert tuple(IncidentModel.model_fields) == INCIDENT_COLUMNS


def test_the_summary_model_mirrors_the_compact_store_row():
    assert tuple(IncidentSummaryModel.model_fields) == SUMMARY_COLUMNS


def test_the_summary_really_is_smaller_and_drops_the_expected_columns():
    dropped = set(IncidentModel.model_fields) - set(IncidentSummaryModel.model_fields)
    assert dropped == {
        "x_coordinate",
        "y_coordinate",
        "updated_on",
        "iucr",
        "primary_type",
        "fbi_code",
    }


def test_both_taxonomies_survive_compaction():
    # Dropping either would make the `taxonomy` parameter unverifiable from a
    # search result.
    assert {"primary_type_canonical", "stable_category"} <= set(IncidentSummaryModel.model_fields)


def test_summary_rows_map_off_the_store_dataclass_by_attribute():
    # The result container is a stand-in; the *rows* are the real dataclass,
    # because rows are what a schema change drifts.
    payload = SearchPayload.from_store(
        SimpleNamespace(rows=[_summary(), _summary(id=1, district="011")])
    )
    assert [row.id for row in payload.incidents] == [13787325, 1]
    assert payload.incidents[1].district == "011"
    assert payload.incidents[0].latitude == 41.864259582


def test_aggregate_buckets_map_off_the_store_dataclass():
    rows = (
        AggregateRow(
            period=date(2025, 1, 1),
            category="BURGLARY",
            geography_value=10,
            incidents=224,
            arrests=18,
            domestic=7,
            geocoded=221,
        ),
    )
    payload = AggregatePayload.from_store(SimpleNamespace(rows=rows))
    assert payload.buckets[0].incidents == 224 and payload.buckets[0].geocoded == 221
    assert payload.buckets[0].geography_value == 10


def test_a_citywide_bucket_keeps_its_nulls_rather_than_inventing_a_geography():
    rows = (
        AggregateRow(
            period=date(2025, 1, 1),
            category=None,
            geography_value=None,
            incidents=5,
            arrests=1,
            domestic=0,
            geocoded=5,
        ),
    )
    payload = AggregatePayload.from_store(SimpleNamespace(rows=rows))
    assert payload.buckets[0].category is None and payload.buckets[0].geography_value is None


def test_search_filters_echo_the_normalized_values_not_the_typed_ones():
    # The store normalizes on the way in; the echo is how a model confirms its
    # input was interpreted rather than ignored.
    query = SearchQuery(
        start=date(2025, 1, 1),
        end=date(2025, 3, 31),
        types=("BURGLARY",),
        geography="district",
        geography_values=("010",),
        limit=3,
        cursor="abc",
    )
    echoed = SearchFilters.from_query(query)
    assert echoed.geography_values == ["010"]
    assert echoed.types == ["BURGLARY"]


def test_search_filters_omit_the_paging_mechanism():
    # limit and cursor shape the page, not the matching set.
    assert not {"limit", "cursor"} & set(SearchFilters.model_fields)


def test_no_filter_model_echoes_the_taxonomy_because_the_envelope_names_it():
    # It is not a predicate -- it narrows nothing, it changes what the category
    # column means -- so it lives at the top level where it cannot be missed.
    for model in (SearchFilters, AggregateFilters, NearbyFilters, LookupFilters):
        assert "taxonomy" not in model.model_fields


def test_aggregate_filters_echo_the_dimensions_as_well_as_the_filters():
    query = AggregateQuery(
        start=date(2015, 1, 1),
        end=date(2026, 6, 30),
        grain="year",
        geography="ward",
        geography_values=(10,),
        types=("BURGLARY",),
        breakdown_by_type=False,
        limit=20,
    )
    echoed = AggregateFilters.from_query(query)
    assert echoed.grain == "year"
    assert echoed.breakdown_by_type is False
    assert echoed.geography_values == [10]
    assert "limit" not in AggregateFilters.model_fields


def test_nearby_filters_carry_no_geography_because_the_radius_is_the_geography():
    assert not {"geography", "geography_values"} & set(NearbyFilters.model_fields)


def test_nearby_filters_echo_the_point_and_radius():
    query = NearbyQuery(
        latitude=41.8827,
        longitude=-87.6233,
        radius_m=300,
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
    )
    echoed = NearbyFilters.from_query(query)
    assert (echoed.latitude, echoed.longitude, echoed.radius_m) == (41.8827, -87.6233, 300)


def test_lookup_filters_echo_whichever_identifier_was_used():
    assert LookupFilters.from_query(LookupQuery(incident_id=42)).incident_id == 42
    echoed = LookupFilters.from_query(LookupQuery(case_number="JJ195924"))
    assert echoed.case_number == "JJ195924" and echoed.incident_id is None


def test_the_nearby_payload_returns_no_rows_unless_they_were_asked_for():
    payload = NearbyPayload(total=930, by_type=[], rings=[], nearest=[])
    assert payload.incidents == []


def test_the_nearby_payload_maps_every_part_of_the_store_result():
    result = SimpleNamespace(
        total=930,
        by_type=[TypeCount(category="THEFT", incidents=426)],
        rings=[DistanceRing(lower_m=0.0, upper_m=75.0, incidents=12)],
        nearest=[NearbyIncident(incident=_summary(), distance_m=90.6)],
        rows=[_summary()],
    )
    payload = NearbyPayload.from_store(result)
    assert payload.total == 930
    assert payload.by_type[0].category == "THEFT"
    assert payload.rings[0].upper_m == 75.0
    assert payload.nearest[0].distance_m == 90.6
    assert payload.nearest[0].incident.id == 13787325
    assert payload.incidents[0].id == 13787325


def test_a_lookup_payload_is_a_list_because_a_case_number_is_not_a_key():
    payload = LookupPayload.from_store(SimpleNamespace(incidents=()))
    assert payload.incidents == []


def test_every_response_wraps_its_payload_in_a_named_object_not_a_bare_array():
    # Uniformity is the point: `data` is always an object with named fields, so
    # a caller never has to branch on which tool it called to read the result.
    for response in (LookupResponse, SearchResponse, AggregateResponse, NearbyResponse):
        data = response.model_json_schema()["properties"]["data"]
        assert "$ref" in data, response
