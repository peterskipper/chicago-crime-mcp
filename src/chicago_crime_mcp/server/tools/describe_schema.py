"""The ``describe_schema`` tool: what is in the data and what counts as valid.

A tool rather than an MCP resource, on purpose. One code path means every call
lands in the same telemetry as every other, and resource support is
client-dependent -- the Anthropic MCP connector this is demonstrated through may
never fetch a resource, which would leave the single most useful affordance
reachable only by clients that happen to implement an optional part of the
protocol.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import time

from chicago_crime_mcp.server.context import get_context
from chicago_crime_mcp.server.envelope import Provenance, RouteInfo
from chicago_crime_mcp.server.schema import (
    GEOGRAPHY_NOTES,
    TAXONOMY_NOTES,
    CategoryVocabulary,
    GeographyDimension,
    Limits,
    SchemaDescription,
    geocode_model,
)
from chicago_crime_mcp.store.duckdb.queries import MAX_LIMIT as MAX_BUCKETS
from chicago_crime_mcp.store.normalize import TYPE_COLUMN
from chicago_crime_mcp.store.postgres.queries import (
    DEFAULT_LIMIT,
    DISTANCE_RINGS,
    MAX_LIMIT,
    MAX_RADIUS_M,
)

#: Time bucket sizes ``aggregate_incidents`` accepts, published so the value does
#: not have to be guessed from the tool's schema alone.
GRAINS: tuple[str, ...] = ("month", "quarter", "year")


def describe_schema() -> SchemaDescription:
    """Describe the dataset: its coverage, its valid values, and its limits.

    Call this before the first query of a conversation. It returns every closed
    set the other tools validate against -- the offense categories that exist
    under each taxonomy, and every beat, district, ward and community area that
    occurs in the data -- so a filter value never has to be guessed. It also
    returns the window the data covers, how much of it carries coordinates, and
    the caps a request will be held to.

    Points worth reading before building a query:

    * Offense categories come in two taxonomies. ``source`` is faithful to CPD's
      own labels and is the default; ``comparable`` re-groups codes that moved
      between categories, and is the right choice for a multi-year trend.
    * The data stops about a week before today. The feed withholds the most
      recent seven days, so "this week" is never answerable.
    * ``community_area`` is the only geography whose boundaries have held still
      across the whole window. Wards and police districts have been redrawn.
    * A row is one **offense recorded in CLEAR**, not one crime and not one
      report. Unreported crime is not here at all.

    Returns:
        The dataset's coverage, vocabularies, geocoding completeness and limits.
    """
    context = get_context()
    started = time.perf_counter()
    vocabulary = context.vocabulary()
    elapsed_ms = (time.perf_counter() - started) * 1000

    return SchemaDescription(
        provenance=Provenance.from_dataset_meta(vocabulary.dataset),
        taxonomies=[
            CategoryVocabulary(
                taxonomy=taxonomy,
                column=column,
                description=TAXONOMY_NOTES[taxonomy],
                categories=list(vocabulary.categories_for(taxonomy)),
            )
            for taxonomy, column in TYPE_COLUMN.items()
        ],
        geographies=[
            GeographyDimension(
                name=geography,
                description=description,
                storage=storage,
                stable_over_time=stable,
                value_count=len(vocabulary.values_for(geography)),
                values=list(vocabulary.values_for(geography)),
            )
            for geography, (description, storage, stable) in GEOGRAPHY_NOTES.items()
        ],
        grains=list(GRAINS),
        geocoding=geocode_model(vocabulary.geocoding),
        limits=Limits(
            max_rows_per_call=MAX_LIMIT,
            default_rows_per_call=DEFAULT_LIMIT,
            max_aggregate_buckets=MAX_BUCKETS,
            max_radius_m=MAX_RADIUS_M,
            distance_rings=DISTANCE_RINGS,
        ),
        route=RouteInfo(
            store="duckdb",
            table="rollup_meta",
            reason=(
                "dataset description, read from the rollup database's own metadata and "
                "reference tables; cached until the nightly rebuild replaces them"
            ),
            elapsed_ms=round(elapsed_ms, 3),
        ),
    )


__all__ = ["GRAINS", "describe_schema"]
