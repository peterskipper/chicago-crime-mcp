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
from functools import cache
from typing import Literal

#: Geography dimension. ``citywide`` means no geography column at all -- for the
#: rollups it selects the dimensionless table, and for a row query it means no
#: geography predicate. It is never a filter value.
Geography = Literal[
    "citywide", "beat", "district", "community_area", "neighborhood", "ward"
]

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
    "neighborhood": "neighborhood",
    "ward": "ward",
}

#: Geographies stored as zero-padded text, with their widths. ``ward`` and
#: ``community_area`` are integers and are coerced as such.
PADDED_GEOGRAPHIES: dict[str, int] = {"beat": 4, "district": 3}

#: Geographies stored as free text, where the stored spelling is the only form
#: the column will match. Every other geography is a code -- an integer or a
#: zero-padded string -- so coercing it is arithmetic. A neighborhood is a name
#: someone typed, and names carry case, spacing and punctuation: ``Little Italy,
#: UIC``, ``O'Hare``, ``Rush & Division``. Matching one has to go through a
#: normalized key and come back out as the exact stored string, or the predicate
#: carries the caller's spelling into SQL and returns a confident empty page.
TEXT_GEOGRAPHIES: frozenset[str] = frozenset({"neighborhood"})


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
    if geography in TEXT_GEOGRAPHIES:
        return tuple(_stored_spelling(str(v)) for v in values)
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


@cache
def _neighborhood_spellings() -> dict[str, str]:
    """Map a normalized lookup key to the stored spelling of each neighborhood.

    Cached for the life of the process, which is correct here and would not be
    for the offense categories or the geography value lists: those come from the
    data and change when the nightly rollup lands, whereas these 98 names come
    from a git-tracked boundary file and change only when someone edits the repo.

    Imported lazily because this module is on every store's import path and most
    callers never touch a neighborhood.

    Returns:
        A ``match_key -> stored spelling`` map for the 98 named neighborhoods.
    """
    from chicago_crime_mcp.geo.resolve import NeighborhoodIndex, match_key

    return {match_key(name): name for name in NeighborhoodIndex.load().names}


def _stored_spelling(value: str) -> str:
    """Return the stored spelling of a neighborhood, or the value untouched.

    Only case and spacing are reconciled -- ``wicker  PARK`` is the same filter
    as ``Wicker Park``, in the same way ``burglary`` is the same filter as
    ``BURGLARY``. It deliberately does **not** apply the curated aliases or the
    containing-area fallbacks: ``Pilsen`` has no polygon and resolves to a
    *community area*, which is a different geography and therefore a different
    argument, not something a value coercion can quietly substitute. Turning a
    colloquial name into a filter is
    :mod:`~chicago_crime_mcp.geo.resolve`'s job, reached through the
    ``resolve_neighborhood`` tool.

    An unrecognized name is returned unchanged rather than rejected here, so the
    caller's own value reaches the vocabulary check and comes back in a teaching
    error that lists what does exist -- a better answer than anything this
    function knows how to say.

    Args:
        value: A neighborhood name as supplied.

    Returns:
        The stored spelling if the name is one of the 98, else ``value``
        unchanged.
    """
    from chicago_crime_mcp.geo.resolve import match_key

    return _neighborhood_spellings().get(match_key(value), value)
