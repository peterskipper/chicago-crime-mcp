"""Unit tests for crime schema handling: IUCR normalization, canonicalization,
and on-disk type coercion.

The canonicalization tests use a small in-memory reference dict for isolation;
one test also loads the committed snapshot to confirm it is wired up correctly.
"""

from __future__ import annotations

import httpx
import pandas as pd
import pytest

from chicago_crime_mcp.ingest import schema
from chicago_crime_mcp.ingest.socrata import SodaClient
from tests.helpers import StubLocator

# A tiny stand-in for the real IUCR reference.
REF = {
    "0281": "CRIMINAL SEXUAL ASSAULT",
    "0265": "CRIMINAL SEXUAL ASSAULT",
    "0810": "THEFT",
}


# -- normalize_iucr --------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("281", "0281"),
        ("0281", "0281"),
        ("1305", "1305"),
        (281, "0281"),
        (" 810 ", "0810"),
        ("", None),
        (None, None),
        (float("nan"), None),
        (pd.NA, None),
    ],
)
def test_normalize_iucr(raw, expected):
    assert schema.normalize_iucr(raw) == expected


# -- add_canonical_primary_type -------------------------------------------


def test_canonical_collapses_relabeled_synonyms():
    df = pd.DataFrame(
        {
            "iucr": ["0281", "0281", "0810"],
            "primary_type": ["CRIM SEXUAL ASSAULT", "CRIMINAL SEXUAL ASSAULT", "THEFT"],
        }
    )
    out = schema.add_canonical_primary_type(df, REF)
    # Both drifted labels for code 0281 collapse to the reference label.
    assert list(out["primary_type_canonical"]) == [
        "CRIMINAL SEXUAL ASSAULT",
        "CRIMINAL SEXUAL ASSAULT",
        "THEFT",
    ]
    # Raw provenance column is untouched.
    assert list(out["primary_type"]) == [
        "CRIM SEXUAL ASSAULT",
        "CRIMINAL SEXUAL ASSAULT",
        "THEFT",
    ]


def test_canonical_falls_back_when_iucr_missing_or_unknown():
    df = pd.DataFrame(
        {
            "iucr": [None, "9999", "281"],  # null, not-in-ref, needs zero-pad
            "primary_type": ["ARSON", "GAMBLING", "CRIMINAL SEXUAL ASSAULT"],
        }
    )
    out = schema.add_canonical_primary_type(df, REF)
    assert out.loc[0, "primary_type_canonical"] == "ARSON"  # null iucr -> raw
    assert out.loc[1, "primary_type_canonical"] == "GAMBLING"  # unknown -> raw
    assert out.loc[2, "primary_type_canonical"] == "CRIMINAL SEXUAL ASSAULT"  # "281"->0281


def test_committed_snapshot_loads_and_maps():
    ref = schema.load_iucr_reference()
    assert len(ref) > 400
    assert ref["0281"] == "CRIMINAL SEXUAL ASSAULT"
    assert ref["0110"] == "HOMICIDE"


# -- add_stable_category ---------------------------------------------------

# The curated overrides: only codes that cross a primary-type boundary.
CURATED = {"0760": "THEFT"}


def test_stable_category_applies_the_curated_override():
    df = pd.DataFrame(
        {
            "iucr": ["0760", "760"],  # already padded, and needing the zero-pad
            "primary_type_canonical": ["BURGLARY", "BURGLARY"],
        }
    )
    out = schema.add_stable_category(df, CURATED)
    assert list(out["stable_category"]) == ["THEFT", "THEFT"]
    # The source taxonomy is untouched: both are still BURGLARY to the city.
    assert list(out["primary_type_canonical"]) == ["BURGLARY", "BURGLARY"]


def test_stable_category_falls_back_to_the_canonical_type():
    df = pd.DataFrame(
        {
            "iucr": ["0610", "9999", None],  # uncurated, unknown, null
            "primary_type_canonical": ["BURGLARY", "OTHER OFFENSE", "ARSON"],
        }
    )
    out = schema.add_stable_category(df, CURATED)
    assert list(out["stable_category"]) == ["BURGLARY", "OTHER OFFENSE", "ARSON"]


def test_stable_category_is_a_no_op_without_curation():
    """An empty override map must leave the two taxonomies identical."""
    df = pd.DataFrame(
        {"iucr": ["0760", "0610"], "primary_type_canonical": ["BURGLARY", "BURGLARY"]}
    )
    out = schema.add_stable_category(df, {})
    assert list(out["stable_category"]) == list(out["primary_type_canonical"])


def test_committed_curation_is_loaded_with_padded_codes():
    """IUCR codes are identifiers: `0760` must not arrive as the integer 760."""
    curated = schema.load_stable_category_map()
    assert curated["0760"] == "THEFT"
    # Only curated codes are returned -- an uncurated one is absent, not blank.
    assert "0110" not in curated
    assert all(len(code) == 4 for code in curated)


# -- coerce_types ----------------------------------------------------------


def test_coerce_types_casts_each_group():
    df = pd.DataFrame(
        {
            "id": ["12345"],
            "case_number": ["JK1"],
            "date": ["2024-01-01T00:00:00.000"],
            "updated_on": ["2024-02-01T12:00:00.000"],
            "beat": ["0111"],
            "district": ["017"],
            "ward": ["33"],
            "community_area": ["14"],
            "arrest": ["false"],
            "domestic": [True],
            "latitude": ["41.9"],
            "longitude": ["-87.6"],
            "primary_type": ["THEFT"],
        }
    )
    out = schema.coerce_types(df)

    assert out["id"].dtype == "Int64" and out.loc[0, "id"] == 12345
    assert out["latitude"].dtype == "float64"
    assert str(out["date"].dtype).startswith("datetime")
    # zero-padded codes survive as strings (would be lost as ints).
    assert out["beat"].dtype == "string" and out.loc[0, "beat"] == "0111"
    assert out.loc[0, "district"] == "017"


def test_coerce_bool_handles_string_false_and_native_bool():
    df = pd.DataFrame({"arrest": ["false", "true"], "domestic": [False, True]})
    out = schema.coerce_types(df)
    assert out["arrest"].dtype == "boolean"
    # "false" must map to False, not be truthy. tolist() yields plain Python
    # bools, so we compare lists directly rather than `== False` (E712).
    assert out["arrest"].tolist() == [False, True]
    assert out["domestic"].tolist() == [False, True]


def test_coerce_types_ignores_absent_columns():
    out = schema.coerce_types(pd.DataFrame({"id": ["1"], "primary_type": ["THEFT"]}))
    assert out["id"].dtype == "Int64"
    assert out["primary_type"].dtype == "string"


# -- refresh_iucr_snapshot diff -------------------------------------------


def _iucr_client(rows):
    """A SodaClient whose backend returns a fixed IUCR reference payload."""

    def handler(request):
        return httpx.Response(200, json=rows)

    return SodaClient(transport=httpx.MockTransport(handler), app_token="T")


def test_refresh_snapshot_reports_diff(tmp_path):
    path = tmp_path / "iucr_codes.csv"

    v1 = [
        {"iucr": "0110", "primary_description": "HOMICIDE"},
        {"iucr": "0281", "primary_description": "CRIM SEXUAL ASSAULT"},
    ]
    diff = schema.refresh_iucr_snapshot(_iucr_client(v1), path=path)
    assert diff == {"added": [], "removed": [], "relabeled": []}  # first write, no prior
    assert path.exists()

    v2 = [
        {"iucr": "0110", "primary_description": "HOMICIDE"},
        {"iucr": "0281", "primary_description": "CRIMINAL SEXUAL ASSAULT"},  # relabeled
        {"iucr": "0130", "primary_description": "HOMICIDE"},  # added
    ]
    diff = schema.refresh_iucr_snapshot(_iucr_client(v2), path=path)
    assert diff == {"added": ["0130"], "removed": [], "relabeled": ["0281"]}


def test_refresh_preserves_curated_columns(tmp_path):
    """A refresh mirrors upstream but must not wipe our own analytic columns.

    `stable_category` is a hand-made judgment that lives in the same file as the
    upstream snapshot; overwriting the file wholesale would silently discard it.
    """
    path = tmp_path / "iucr_codes.csv"
    upstream = [
        {"iucr": "0610", "primary_description": "BURGLARY"},
        {"iucr": "0760", "primary_description": "BURGLARY"},
    ]
    schema.refresh_iucr_snapshot(_iucr_client(upstream), path=path)

    curated = pd.read_csv(path, dtype=str)
    assert "stable_category" in curated.columns  # created even on a first write
    curated.loc[curated["iucr"] == "0760", "stable_category"] = "THEFT"
    curated.to_csv(path, index=False)

    schema.refresh_iucr_snapshot(_iucr_client(upstream), path=path)

    after = pd.read_csv(path, dtype=str).set_index("iucr")["stable_category"]
    assert after["0760"] == "THEFT"
    assert pd.isna(after["0610"])


def test_committed_snapshot_carries_the_curated_column():
    """The shipped snapshot has the column, curated sparsely and on purpose."""
    df = pd.read_csv(schema.IUCR_REFERENCE_PATH, dtype=str)
    curated = dict(
        zip(
            df.loc[df["stable_category"].notna(), "iucr"],
            df.loc[df["stable_category"].notna(), "stable_category"],
            strict=True,
        )
    )
    # `0760` BURGLARY FROM MOTOR VEHICLE: minted 2021, ramped 2024, and moved car
    # break-ins from THEFT into BURGLARY. Mapping it back makes both series
    # comparable across its introduction.
    assert curated["0760"] == "THEFT"
    # Curation is deliberately tiny -- every row needs evidence of measured drift,
    # and a full IUCR taxonomy is a research project, not this feature.
    assert len(curated) < 10, f"unexpectedly broad curation: {curated}"


# -- neighborhood ----------------------------------------------------------

# Two points that locate and one that does not, so no assertion below can pass
# by tagging everything alike.
WICKER_PARK = (41.9088, -87.6796)
LOOP = (41.8781, -87.6298)
LAKE = (41.8800, -87.5500)
LOCATOR = StubLocator({WICKER_PARK: "Wicker Park", LOOP: "Loop"})


def _points(*coords) -> pd.DataFrame:
    return pd.DataFrame(
        {"latitude": [c[0] for c in coords], "longitude": [c[1] for c in coords]}
    )


def test_add_neighborhood_tags_each_point_independently():
    tagged = schema.add_neighborhood(_points(WICKER_PARK, LOOP, WICKER_PARK), LOCATOR)
    assert tagged["neighborhood"].tolist() == ["Wicker Park", "Loop", "Wicker Park"]


def test_add_neighborhood_is_null_where_no_polygon_contains_the_point():
    """~1.8% of real rows. Nullable is the design, not an oversight."""
    tagged = schema.add_neighborhood(_points(LOOP, LAKE), LOCATOR)
    assert tagged["neighborhood"].isna().tolist() == [False, True]


def test_add_neighborhood_accepts_the_strings_soda_returns():
    """This runs before coerce_types, where latitude is still a string."""
    raw = pd.DataFrame({"latitude": ["41.8781"], "longitude": ["-87.6298"]})
    assert schema.add_neighborhood(raw, LOCATOR)["neighborhood"].tolist() == ["Loop"]


def test_add_neighborhood_leaves_the_input_frame_alone():
    frame = _points(LOOP)
    schema.add_neighborhood(frame, LOCATOR)
    assert "neighborhood" not in frame.columns


def test_coerce_types_pins_the_neighborhood_column_to_string():
    """`coerce_types` is the single authority on the on-disk schema.

    ``add_neighborhood`` already emits a string Series, so this looks redundant
    -- but Parquet partitions written across years have to share a schema, and
    that guarantee belongs to one function rather than to every producer of a
    column. Given an object-dtype column, it still lands as ``string``.
    """
    raw = pd.DataFrame({"neighborhood": ["Loop", None]}, dtype=object)
    assert schema.coerce_types(raw)["neighborhood"].dtype == "string"


def test_add_neighborhood_needs_coordinates():
    """A pull that dropped latitude must fail loudly, not write a null column."""
    with pytest.raises(KeyError):
        schema.add_neighborhood(pd.DataFrame({"id": [1]}), LOCATOR)


# -- the shared pipeline ---------------------------------------------------


def test_prepare_derives_every_column_and_coerces():
    raw = pd.DataFrame(
        {
            "id": ["1", "2"],
            "iucr": ["0760", "0810"],
            "primary_type": ["BURGLARY", "THEFT"],
            "arrest": ["true", "false"],
            "latitude": ["41.9088", "41.8800"],
            "longitude": ["-87.6796", "-87.5500"],
        }
    )
    out = schema.prepare(raw, {"0760": "BURGLARY", "0810": "THEFT"}, CURATED, LOCATOR)

    assert out["primary_type_canonical"].tolist() == ["BURGLARY", "THEFT"]
    # 0760 is the one curated code: the city moved it out of THEFT.
    assert out["stable_category"].tolist() == ["THEFT", "THEFT"]
    assert out["neighborhood"].tolist()[0] == "Wicker Park"
    assert out["neighborhood"].isna().tolist() == [False, True]
    assert out["id"].dtype == "Int64"
    assert out["arrest"].dtype == "boolean"
    assert out["neighborhood"].dtype == "string"


def test_prepare_is_the_only_derivation_both_ingest_paths_run():
    """Guard the consolidation: a fourth column added here must reach both paths.

    The backfill used to hold a private ``_prepare`` and the incremental sync
    repeated its body inline, so each new derived column had to be remembered
    twice. If either path stops calling this, that drift is back.
    """
    import inspect

    from chicago_crime_mcp.ingest import backfill, incremental

    for module in (backfill, incremental):
        assert "schema.prepare(" in inspect.getsource(module), module.__name__
        assert not hasattr(module, "_prepare"), module.__name__
