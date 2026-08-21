"""Refresh the pinned neighborhood boundary snapshot and its containment table.

Chicago publishes 98 named neighborhood polygons. They are the sharper
geography than the 77 community areas the incident feed carries -- "robberies in
Wicker Park" is a question about 21% of West Town -- but the city does not tag
incidents with them, so we derive the tag ourselves at ingest. That needs the
polygons on disk, pinned, so ingest is reproducible and works offline.

This script writes two of the three neighborhood reference files. It follows the
``iucr_codes.csv`` precedent: a git-tracked snapshot plus a refresh script, with
the ownership split spelled out in ``reference/__init__.py``.

- ``neighborhoods.geojson`` is a **mirror** of Socrata ``y6yq-dbs2``
  (Neighborhoods_2012b), at full coordinate precision. Never hand-edit it.
- ``neighborhood_areas.csv`` is **ours**: the containment table, derived here by
  intersecting those polygons with the community areas (``igwz-8jzy``). Never
  hand-edit it either -- rerun this script.
- ``neighborhood_aliases.csv`` is **ours**, hand-curated, and this script does
  not touch it. It is validated against the other two by the tests.

Community-area boundaries are fetched on demand rather than vendored: they are
needed to compute the shares and for nothing else, and the incident feed already
carries the community-area number on every row.

**Do not simplify the polygons.** Measured on the 2.88M-row dataset:
``ST_Simplify`` at 1e-5 (~1 m) shrinks the file from 2.19 MB to 0.34 MB but
reassigns 2.4% of rows -- 4,682 move to a different neighborhood and 827 fall
into gaps -- because simplifying each polygon independently breaks the edges it
shares with its neighbors. 1.85 MB is not worth a 2.4% error.

Run from the repo root::

    python scripts/build_neighborhood_reference.py
    python scripts/build_neighborhood_reference.py --dry-run   # report only

Needs network, both for Socrata and (on first use) for DuckDB's ``spatial``
extension, which caches to ``~/.duckdb/extensions``. That is why this is a
script and not part of the test suite.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from chicago_crime_mcp.ingest.socrata import SodaClient
from chicago_crime_mcp.reference import NEIGHBORHOOD_AREAS_PATH, NEIGHBORHOOD_BOUNDARIES_PATH

log = logging.getLogger(__name__)

#: Socrata dataset holding the 98 named neighborhood polygons (Neighborhoods_2012b).
#: NOT ``bbvz-uum9``, the current "Boundaries - Neighborhoods": it reports 98 rows
#: but exposes zero columns through the API and its GeoJSON export returns an
#: empty FeatureCollection. Broken as of 2026-08-17.
NEIGHBORHOOD_DATASET_ID = "y6yq-dbs2"

#: Socrata dataset holding the 77 community-area polygons. ``cauq-8yn6`` is the
#: map view of the same data and returns empty objects through the API.
COMMUNITY_AREA_DATASET_ID = "igwz-8jzy"

#: Metric CRS for the area maths (UTM zone 16N, metres). Shares are ratios, so
#: the projection cancels to first order, but computing them in degrees would
#: still bake in the latitude distortion. Cheap to do properly.
METRIC_CRS = "EPSG:26916"

#: Minimum share of a neighborhood that must fall inside a community area for
#: the pair to be recorded. Adjacent polygons drawn from different sources never
#: agree exactly along a shared edge, so every neighborhood technically clips a
#: sliver of its neighbors' community areas: without a floor the join returns 410
#: pairs for 98 neighborhoods.
#:
#: The measured distribution leaves no judgment call. The smallest genuine
#: straddle is Streeterville at 5.2% in the Loop; the largest sliver is Galewood
#: at 0.15% in Montclare. Nothing falls in between, so this sits in the middle of
#: a 34x gap and the exact value does not matter. 104 pairs survive.
MIN_SHARE = 0.01


def fetch_boundaries(client: SodaClient, dataset_id: str, name_fields: list[str]) -> dict:
    """Fetch one boundary dataset and shape it into a GeoJSON FeatureCollection.

    Socrata returns ``the_geom`` already decoded as a GeoJSON geometry object,
    so this reassembles rather than parses.

    Args:
        client: An open :class:`~chicago_crime_mcp.ingest.socrata.SodaClient`.
        dataset_id: Socrata dataset identifier.
        name_fields: Non-geometry columns to carry through as feature properties.

    Returns:
        A GeoJSON FeatureCollection dict, one feature per row, with coordinates
        exactly as served.

    Raises:
        SodaError: Propagated from the underlying request.
        KeyError: If a row is missing ``the_geom``, which would mean the dataset
            changed shape and the derived table cannot be trusted.
    """
    rows = client.get(dataset_id, {"$limit": 1000})
    features = [
        {
            "type": "Feature",
            "geometry": row["the_geom"],
            "properties": {field: row.get(field) for field in name_fields},
        }
        for row in rows
    ]
    log.info("fetched %d features from %s", len(features), dataset_id)
    return {"type": "FeatureCollection", "features": features}


def compute_containment(
    neighborhoods: dict, community_areas: dict, min_share: float = MIN_SHARE
) -> pd.DataFrame:
    """Intersect the two boundary sets and report both overlap directions.

    Two fractions, because they answer different questions:

    - ``share_of_neighborhood_in_ca`` -- how completely the community area
      contains the neighborhood. At or above 98% for 92 of the 98; below that
      only for the three genuine straddlers.
    - ``share_of_ca_covered_by_neighborhood`` -- the dilution. How much broader a
      community-area answer is than the neighborhood that was actually asked
      about. Median 20%: Greektown is 1% of the Near West Side, Little Village is
      all of South Lawndale.

    Args:
        neighborhoods: FeatureCollection with a ``pri_neigh`` property.
        community_areas: FeatureCollection with ``community`` and ``area_numbe``.
        min_share: Drop pairs below this share of the neighborhood, to discard
            edge slivers. See :data:`MIN_SHARE`.

    Returns:
        One row per surviving (neighborhood, community area) pair, sorted by
        neighborhood then descending share. Straddlers appear more than once.
    """
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")

    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for key, collection in (("hood", neighborhoods), ("ca", community_areas)):
            paths[key] = Path(tmp) / f"{key}.geojson"
            paths[key].write_text(json.dumps(collection))

        # Project once into metres, then every area below is a plain ST_Area.
        for key, columns in (
            ("hood", "pri_neigh AS neighborhood"),
            ("ca", "community AS community_area, CAST(area_numbe AS INTEGER) AS ca_number"),
        ):
            conn.execute(
                f"""
                CREATE TABLE {key} AS
                SELECT {columns},
                       ST_Transform(geom, 'EPSG:4326', '{METRIC_CRS}', always_xy := true) AS g
                FROM ST_Read('{paths[key]}')
                """
            )

        df = conn.execute(
            """
            WITH overlap AS (
                SELECT hood.neighborhood,
                       ca.ca_number AS community_area_number,
                       ca.community_area,
                       ST_Area(ST_Intersection(hood.g, ca.g)) AS shared,
                       ST_Area(hood.g) AS hood_area,
                       ST_Area(ca.g) AS ca_area
                FROM hood JOIN ca ON ST_Intersects(hood.g, ca.g)
            )
            SELECT neighborhood,
                   community_area_number,
                   community_area,
                   shared / hood_area AS share_of_neighborhood_in_ca,
                   shared / ca_area AS share_of_ca_covered_by_neighborhood
            FROM overlap
            WHERE shared / hood_area >= ?
            ORDER BY neighborhood, share_of_neighborhood_in_ca DESC
            """,
            [min_share],
        ).fetchdf()

    conn.close()
    return df.round({"share_of_neighborhood_in_ca": 4, "share_of_ca_covered_by_neighborhood": 4})


def main(argv: list[str] | None = None) -> None:
    """Refresh the boundary snapshot and rebuild the containment table.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change without writing."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with SodaClient() as client:
        neighborhoods = fetch_boundaries(
            client, NEIGHBORHOOD_DATASET_ID, ["pri_neigh", "sec_neigh"]
        )
        community_areas = fetch_boundaries(
            client, COMMUNITY_AREA_DATASET_ID, ["community", "area_numbe"]
        )

    names = {f["properties"]["pri_neigh"] for f in neighborhoods["features"]}
    log.info(
        "neighborhoods: %d features, %d distinct names",
        len(neighborhoods["features"]), len(names),
    )

    areas = compute_containment(neighborhoods, community_areas)
    straddlers = sorted(areas["neighborhood"].value_counts().loc[lambda s: s > 1].index)
    unmatched = sorted(names - set(areas["neighborhood"]))
    log.info(
        "containment: %d pairs, %d straddle >1 community area (%s), %d unmatched (%s)",
        len(areas), len(straddlers), ", ".join(straddlers) or "none",
        len(unmatched), ", ".join(unmatched) or "none",
    )

    if args.dry_run:
        log.info("dry run: nothing written")
        return

    _write(NEIGHBORHOOD_BOUNDARIES_PATH, json.dumps(neighborhoods))
    _write(NEIGHBORHOOD_AREAS_PATH, areas.to_csv(index=False))


def _write(path: Path, text: str) -> None:
    """Write text to ``path`` via a sibling temp file and an atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    log.info("wrote %s (%.2f MB)", path, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
