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
"""

from __future__ import annotations

from pathlib import Path

IUCR_REFERENCE_PATH = Path(__file__).parent / "iucr_codes.csv"

__all__ = ["IUCR_REFERENCE_PATH"]
