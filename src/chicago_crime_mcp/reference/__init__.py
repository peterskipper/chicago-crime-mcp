"""Pinned reference data shared by the ingest and store layers.

``iucr_codes.csv`` is a snapshot of Chicago's IUCR reference table (Socrata
dataset ``c7ck-438e``), committed so canonicalization is reproducible and works
offline. It lives here rather than under ``ingest/`` because both layers read it:
ingest uses it to canonicalize ``primary_type``, and the DuckDB rollup build
joins it to tag every incident with a ``stable_category``.

**The file has two kinds of columns, with two different owners:**

- ``iucr``, ``primary_description``, ``secondary_description``, ``index_code``,
  ``active`` are mirrored from upstream. Refresh them with
  ``ingest.schema.refresh_iucr_snapshot``; never hand-edit them.
- ``stable_category`` is **ours** - an analytic judgment, not a fact from the
  city. It is blank for all but a handful of codes; blank means "no drift found,
  use ``primary_description``". ``refresh_iucr_snapshot`` carries the column
  across a refresh rather than overwriting it.

Keeping both in one file is deliberate: one row per IUCR code, one place to look.
The cost is that the file is no longer a pure upstream mirror, which is why the
ownership split is spelled out here and enforced by the merge-on-refresh test.

See the "On comparing crime over time" section of the README for the evidence
behind each curated row.

**The three neighborhood files** follow the same pattern for the same reason:
the city does not tag incidents with a neighborhood, so we derive that column
ourselves at ingest and the polygons have to be pinned for the derivation to be
reproducible and offline. Their ownership splits three ways:

- ``neighborhoods.geojson`` is a **mirror** of Socrata ``y6yq-dbs2``
  (Neighborhoods_2012b): 98 named polygons at full coordinate precision.
  Refresh it with ``scripts/build_neighborhood_reference.py``; never hand-edit
  it, and never simplify it - see that script for the 2.4% misassignment that
  buys.
- ``neighborhood_areas.csv`` is **ours, but derived**: which community area
  contains each neighborhood, and how much of each contains the other. Produced
  by the same script from the two boundary sets. Also never hand-edited - a
  wrong number here is fixed by fixing the derivation.
- ``neighborhood_aliases.csv`` is **ours, and hand-curated**: what people
  actually call these places. This is the one file a human writes, and it is the
  safety mechanism rather than a convenience. Fuzzy matching is unsafe on
  neighborhood names because the set is inherently incomplete - ``Bronzeville``
  is a real place with no polygon, and difflib confidently matches it to
  ``Andersonville``, 19.4 km away. An alias row is what keeps that from
  happening. It is deliberately small; ``resolve_neighborhood`` misses are the
  telemetry signal for what to add next.

The two derived neighborhood files are validated against each other and against
the curated one by ``tests/test_neighborhood_reference.py``.
"""

from __future__ import annotations

from pathlib import Path

IUCR_REFERENCE_PATH = Path(__file__).parent / "iucr_codes.csv"
NEIGHBORHOOD_BOUNDARIES_PATH = Path(__file__).parent / "neighborhoods.geojson"
NEIGHBORHOOD_AREAS_PATH = Path(__file__).parent / "neighborhood_areas.csv"
NEIGHBORHOOD_ALIASES_PATH = Path(__file__).parent / "neighborhood_aliases.csv"

__all__ = [
    "IUCR_REFERENCE_PATH",
    "NEIGHBORHOOD_ALIASES_PATH",
    "NEIGHBORHOOD_AREAS_PATH",
    "NEIGHBORHOOD_BOUNDARIES_PATH",
]
