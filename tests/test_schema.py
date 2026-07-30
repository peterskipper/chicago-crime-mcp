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
