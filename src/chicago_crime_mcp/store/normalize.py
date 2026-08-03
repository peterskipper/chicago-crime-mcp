"""Filter vocabulary shared by both stores, and the coercion that makes it match.

Postgres and DuckDB answer different question shapes -- rows and radii on one
side, aggregates on the other -- but they filter on the *same* columns, from the
same caller-supplied values, and they must agree about what those values mean.
This module is the single place that decides.

**Why it is shared rather than duplicated.** The failure this prevents is
specific and silent. The feed stores district ``010`` as zero-padded text, so a
model that passes the integer ``10`` matches no rows. That is not an error --
it is a successful query returning zero, which reads as "there was no crime in
district 10" rather than "you spelled the district wrong". Every filter here
has that property. If ``search_incidents`` and ``aggregate_incidents`` were to
normalize differently, the same filter would silently mean two things and the
two tools would disagree about the same slice of the same dataset, which is
worse than either being wrong alone: the envelope cannot warn about an
inconsistency it has no way to see.

**Values, not query objects.** The coercers take bare values so both stores can
call them from dataclasses that share no shape. Each store keeps its own query
type and its own normalizing wrapper; they share the rules, not the containers.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

#: Geography dimension. ``citywide`` means no geography column at all -- for the
#: rollups it selects the dimensionless table, and for a row query it means no
#: geography predicate. It is never a filter value.
Geography = Literal["citywide", "beat", "district", "community_area", "ward"]

#: Which offense taxonomy to filter and group by. ``source`` reports what the
#: city called it; ``comparable`` reports the curated stable category, which is
#: what makes a cross-year comparison valid. See the README's "On comparing
#: crime over time". Defaults to ``source`` everywhere: normalizing counts is an
#: analytic choice the caller has to make explicitly, never one a store makes
#: for them.
Taxonomy = Literal["source", "comparable"]

#: The offense-category column each taxonomy selects. Both are materialized on
#: every relation in both stores -- ``stable_category`` is derived once at ingest
#: rather than joined at read time -- so choosing a taxonomy is choosing a column
#: name, with no join, no fallback and no Python-side code-set resolution.
TYPE_COLUMN: dict[Taxonomy, str] = {
    "source": "primary_type_canonical",
    "comparable": "stable_category",
}

#: The geography column each dimension reads. The names are identical in
#: Postgres, in the rollup tables and in the tagged Parquet view, which is why
#: one mapping serves all three. None for citywide: no column, no predicate, no
#: GROUP BY term.
GEO_COLUMN: dict[Geography, str | None] = {
    "citywide": None,
    "beat": "beat",
    "district": "district",
    "community_area": "community_area",
    "ward": "ward",
}

#: Geographies stored as zero-padded text, with their widths. The rest
#: (``ward``, ``community_area``) are integers and are coerced as such.
PADDED_GEOGRAPHIES: dict[str, int] = {"beat": 4, "district": 3}


def normalize_types(types: Iterable[str]) -> tuple[str, ...]:
    """Coerce offense categories to the case the columns actually hold.

    Both taxonomy columns store upper-case categories, so a caller asking for
    ``"burglary"`` would otherwise match nothing.

    Args:
        types: Offense categories as supplied, in any case, possibly padded.

    Returns:
        The categories upper-cased and stripped, in the order given.
    """
    return tuple(t.strip().upper() for t in types)


def normalize_geography_values(
    geography: Geography, values: Iterable[str | int]
) -> tuple[str | int, ...]:
    """Coerce geography values to the storage type of the chosen geography.

    Args:
        geography: Which geography the values name.
        values: The values as supplied, as strings or ints.

    Returns:
        The coerced values, in the order given. Empty for a citywide query,
        where a geography filter is meaningless because there is no column to
        filter on.

    Raises:
        ValueError: If a ward or community area is not an integer. The message
            names the geography so the server can turn it into a teaching error
            that tells the model which field it got wrong.
    """
    if geography == "citywide":
        return ()
    width = PADDED_GEOGRAPHIES.get(geography)
    if width is not None:
        return tuple(str(v).strip().zfill(width) for v in values)
    coerced: list[str | int] = []
    for value in values:
        try:
            coerced.append(int(str(value).strip()))
        except ValueError as exc:
            raise ValueError(f"{geography} must be an integer, got {value!r}") from exc
    return tuple(coerced)
