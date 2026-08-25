"""The ``resolve_neighborhood`` tool: a name someone said -> a filter that answers it.

The other five tools take arguments a caller could have read off
``describe_schema``. This one takes the thing a person actually says. "How bad is
Bronzeville?" is a perfectly ordinary question about a real Chicago
neighborhood, and there is no ``Bronzeville`` in any column of the dataset.

**Why this is a tool and not a fuzzy match inside the other tools.** The
alternative -- quietly accepting any close-enough name in ``geography_values`` --
was measured and is unsafe. The 98 named boundaries the city publishes are not
the names Chicagoans use, so a miss is usually a real place with no boundary
rather than a typo, and difflib answers those with total confidence:
``Bronzeville`` comes back as ``Andersonville``, 19.4 km away at the opposite end
of the city, with a plausible number attached. A model has no way to notice. So
the guess is made here, where it can be labelled a guess, carry a score, and be
handed back as a question instead of an answer.

**The two geographies it returns are not interchangeable.** A neighborhood with a
boundary is answerable exactly. One without resolves to the *community area*
containing it, which is a broader question than the one asked -- Wicker Park is
about a fifth of West Town -- so every candidate carries the overlap shares that
say how much broader.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import time
from functools import cache

from chicago_crime_mcp.geo.resolve import NeighborhoodIndex
from chicago_crime_mcp.server.context import get_context
from chicago_crime_mcp.server.envelope import Provenance, RouteInfo
from chicago_crime_mcp.server.errors import InvalidArgumentError, UnknownValueError
from chicago_crime_mcp.server.models import (
    NeighborhoodCandidateModel,
    NeighborhoodResolution,
)


@cache
def _index() -> NeighborhoodIndex:
    """Return the process-wide resolution index.

    Cached for the life of the process rather than per rollup build, unlike
    :meth:`ServerContext.vocabulary`. The distinction is where the data comes
    from: the vocabulary is read out of the rollups and changes when the nightly
    rebuild lands, while these tables are git-tracked reference files that change
    only when the repository does. Tying them to the inode swap would reload two
    CSVs nightly to get the same answer.

    The two must still agree on the 98 names, since a value this tool hands back
    is one the other tools have to accept. That is asserted in the tests rather
    than reconciled at runtime: a divergence means the reference file and the
    loaded data disagree, which is a deployment problem to fix, not a condition
    to paper over per call.

    Returns:
        The loaded index.
    """
    return NeighborhoodIndex.load()


def resolve_neighborhood(name: str) -> NeighborhoodResolution:
    """Turn a Chicago neighborhood name into the geography filter that answers it.

    Call this before filtering by neighborhood on any name you have not seen in
    ``describe_schema``'s list. It is cheap, it never runs a query over the
    incidents, and it is the difference between an answer about Bronzeville and
    an answer about somewhere 19 km away.

    It returns one of four kinds of match, and the kind matters:

    * ``exact`` / ``alias`` -- the place has a boundary of its own. Pass the
      candidate's ``value`` with ``geography="neighborhood"`` and the answer is
      exactly the place asked about.
    * ``containing`` -- the place is real but has no boundary in the city's data
      (Pilsen, Bronzeville, Back of the Yards). The candidate answers under
      ``geography="community_area"`` instead, which is **broader than the
      question**. Say so when reporting the result.
    * ``suggestion`` -- nothing matched and this is a guess. **Do not filter on
      it.** Ask the user whether they meant it, then call again with their
      answer.

    Every candidate also reports the community areas the place sits inside and
    how much of each it covers, so a wider answer can be qualified rather than
    passed off as a narrow one.

    Args:
        name: A neighborhood name as a person would say it. Case, extra spaces
            and a leading "the" do not matter.

    Returns:
        The ranked interpretations, with the geography and value to pass back.

    Raises:
        InvalidArgumentError: If the name is empty.
        UnknownValueError: If nothing matched and nothing is even close. The
            message lists all 98 resolvable names and points at the other ways
            to ask about a place.
    """
    context = get_context()
    started = time.perf_counter()

    if not name or not name.strip():
        raise InvalidArgumentError(
            "A neighborhood name is required.",
            field="name",
            received=name,
            hint="Pass a name such as 'Wicker Park' or 'Pilsen'.",
        )

    index = _index()
    resolution = index.resolve(name)
    vocabulary = context.vocabulary()
    elapsed_ms = (time.perf_counter() - started) * 1000

    if not resolution.candidates:
        raise UnknownValueError(
            f"{name!r} does not match any Chicago neighborhood this server knows.",
            field="name",
            received=name,
            valid_values=index.names,
            # Redundant today and kept anyway. Reaching here means tier 4 found
            # nothing above the cutoff, and the error would run the same matcher
            # at the same cutoff -- probed over 4,000 inputs without finding one
            # where the two disagree. But they are two matchers over two
            # spellings of the set (normalized keys there, display names here),
            # so a change to either could reintroduce the hazard silently. This
            # says the error is never the place a guess comes from.
            suggest_nearest=False,
            # All 98, not the usual 40: for this tool the inventory *is* the
            # answer, and it costs about 1.3 KB against a wasted round trip.
            max_listed=len(index.names),
            hint=(
                "These are the neighborhoods with boundaries of their own. Many well-known "
                "places are not among them and resolve to the community area containing them "
                "instead -- try the name as locals write it. A place can also be asked about "
                "by community_area, ward, district or beat, or by coordinates with "
                "nearby_incidents."
            ),
        )

    return NeighborhoodResolution(
        query=resolution.query,
        resolved=resolution.resolved,
        candidates=[
            NeighborhoodCandidateModel.model_validate(candidate)
            for candidate in resolution.candidates
        ],
        provenance=Provenance.from_dataset_meta(vocabulary.dataset),
        route=RouteInfo(
            store="reference",
            table="neighborhood_areas",
            reason=(
                "name resolved against the pinned neighborhood boundaries and the curated "
                "alias table; no incident data was read"
            ),
            elapsed_ms=round(elapsed_ms, 3),
        ),
    )


__all__ = ["resolve_neighborhood"]
