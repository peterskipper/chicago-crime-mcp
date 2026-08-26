"""The neighborhood resolution ladder: a name a person typed -> a value to filter on.

This is the serving half of ``geo``, and it holds **no geometry**. Every answer
comes from the two reference tables, so the server never links DuckDB's spatial
extension and never reaches the network that ``INSTALL spatial`` needs on first
use. Geometry lives in :mod:`geo.boundaries`, which only ingest imports.

**Why fuzzy matching may suggest but never resolve.** ``server.errors.suggest``
is safe on offense categories because that set is *complete*: a near-miss really
is a typo. The neighborhood set is *inherently incomplete* -- the colloquial tail
has no end -- so a near-miss is usually a real place that is simply absent, and
difflib answers it with breezy confidence. Measured against the 98 names:

    the loop       -> West Loop       2.4 km off
    Roscoe Village -> East Village    5.5 km off
    Bronzeville    -> Andersonville   19.4 km off, and 417 vs 3,306 incidents

Same matcher, opposite safety properties. So the curated alias table is the
safety mechanism rather than a convenience, and tier 4 exists to *ask*, never to
answer. A caller must not act on a ``suggestion`` candidate without confirming
it, even though the candidate carries a usable value -- it carries one so that
confirming is a single cheap round trip.

**Two geographies come back, and they are not interchangeable.** A neighborhood
that has a polygon is answerable exactly. One that does not resolves to the
*community area* containing it, which is a broader question than the one asked:
Wicker Park is 21% of West Town, so a community-area answer to "robberies in
Wicker Park" is roughly five times too big. Callers are expected to say so, which
is why every candidate carries its containing areas and their overlap shares.

**Facts, not prose.** These dataclasses report what matched and how much broader
it is; turning that into a sentence a model reads is the tool layer's job.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import pandas as pd

from chicago_crime_mcp.reference import NEIGHBORHOOD_ALIASES_PATH, NEIGHBORHOOD_AREAS_PATH

#: Which rung of the ladder produced a candidate. Closed, and carried on every
#: result: "how did you get this answer" is not something a caller should have to
#: infer, and telemetry buckets on it to find which tier is doing the work.
MatchKind = Literal["exact", "alias", "containing", "suggestion"]

#: Similarity floor for offering a suggestion. Deliberately the same value as
#: ``server.errors.NEAREST_CUTOFF`` and deliberately not imported from it: that
#: module imports fastmcp, which is a ``[server]`` extra, and ``geo`` is on the
#: ingest path. Below this difflib starts matching on a shared prefix and
#: nothing else.
NEAREST_CUTOFF = 0.6

#: How many suggestions to offer on a miss. One is often the wrong one; a short
#: ranked list lets the caller recognize the right answer instead of guessing
#: again.
MAX_SUGGESTIONS = 3

#: Leading article stripped during normalization. "The Loop" is the Loop, and
#: this one rule is what makes tier 1 handle it without an alias row.
LEADING_ARTICLE = "the "


def match_key(name: str) -> str:
    """Reduce a name to the key the lookup tables are indexed by.

    Casefolds, collapses runs of whitespace, and drops a leading article. It
    deliberately leaves punctuation alone: several stored names carry it
    (``Little Italy, UIC``, ``O'Hare``, ``Rush & Division``), and folding it away
    would merge keys that name different places. Bridging punctuation is the
    alias table's job, where each row is a deliberate decision.

    Args:
        name: A neighborhood name as supplied, in any case or spacing.

    Returns:
        The normalized lookup key.
    """
    key = " ".join(name.casefold().split())
    return key[len(LEADING_ARTICLE) :] if key.startswith(LEADING_ARTICLE) else key


@dataclass(frozen=True)
class ContainingArea:
    """A community area containing some or all of a neighborhood.

    Attributes:
        number: The community area number, which is what the incident rows
            actually carry and therefore what a filter takes.
        name: The community area's name, in the city's spelling.
        share_of_neighborhood: How much of the neighborhood falls inside this
            community area, 0-1. ``None`` when the neighborhood has no polygon,
            so the overlap cannot be measured at all.
        share_of_area: How much of the community area the neighborhood covers,
            0-1 -- the dilution, and the number that says how much broader a
            community-area answer would be. ``None`` for the same reason.
    """

    number: int
    name: str
    share_of_neighborhood: float | None
    share_of_area: float | None


@dataclass(frozen=True)
class Candidate:
    """One way of interpreting the name that was asked about.

    Attributes:
        match_kind: Which rung produced this. See :data:`MatchKind`.
        label: The place itself, in its canonical spelling -- what the caller
            meant, not what they typed.
        geography: The filter field that answers for this place. ``neighborhood``
            when a polygon exists, ``community_area`` when the answer has to
            widen to the containing area.
        value: The value to pass to that field. A name for ``neighborhood``, a
            number for ``community_area``.
        containing_areas: The community areas this place sits in, widest share
            first. Usually one; six neighborhoods straddle two.
        score: difflib similarity, 0-1. Set only on a ``suggestion``, where the
            caller needs to judge whether to accept it; ``None`` otherwise,
            because an exact match has nothing to weigh.
    """

    match_kind: MatchKind
    label: str
    geography: Literal["neighborhood", "community_area"]
    value: str | int
    containing_areas: tuple[ContainingArea, ...]
    score: float | None = None


@dataclass(frozen=True)
class Resolution:
    """What the ladder made of one query.

    Attributes:
        query: The name as supplied, echoed back so an error can quote it.
        candidates: Ranked interpretations. Exactly one when the name resolved,
            up to :data:`MAX_SUGGESTIONS` when it only nearly did, and empty on a
            miss.
    """

    query: str
    candidates: tuple[Candidate, ...]

    @property
    def resolved(self) -> bool:
        """Whether the name was resolved outright, rather than merely guessed at."""
        return bool(self.candidates) and self.candidates[0].match_kind != "suggestion"


class NeighborhoodIndex:
    """The lookup tables behind the ladder, loaded once and reused.

    Attributes:
        names: The 98 neighborhood names that have polygons, sorted, in their
            stored spelling. This is the list a teaching error spells out.
    """

    def __init__(self, areas: pd.DataFrame, aliases: pd.DataFrame) -> None:
        """Build the indexes from the two reference tables.

        Args:
            areas: The containment table -- one row per (neighborhood,
                community area) pair, with both overlap shares.
            aliases: The curated alias table.
        """
        self.names: tuple[str, ...] = tuple(sorted(areas["neighborhood"].unique()))
        self._by_key = {match_key(name): name for name in self.names}

        # Rows are read as plain dicts rather than through `itertuples()`: both
        # reference tables are small (hundreds of rows, loaded once), and a dict
        # of cell values is a real boundary out of pandas -- `itertuples()`
        # hands back attributes typed as every scalar pandas can hold, so the
        # coercions below could not be checked against it.
        ordered = areas.sort_values("share_of_neighborhood_in_ca", ascending=False)
        self._areas: dict[str, tuple[ContainingArea, ...]] = {
            str(name): tuple(
                ContainingArea(
                    number=int(row["community_area_number"]),
                    name=row["community_area"],
                    share_of_neighborhood=float(row["share_of_neighborhood_in_ca"]),
                    share_of_area=float(row["share_of_ca_covered_by_neighborhood"]),
                )
                for row in group.to_dict("records")
            )
            for name, group in ordered.groupby("neighborhood", sort=False)
        }

        self._aliases: dict[str, Candidate] = {}
        for row in aliases.to_dict("records"):
            key = match_key(row["alias"])
            if row["match_kind"] == "alias":
                self._aliases[key] = self._exact(row["target_value"], "alias")
            else:
                self._aliases[key] = Candidate(
                    match_kind="containing",
                    label=row["alias"],
                    geography="community_area",
                    value=int(row["target_value"]),
                    # No polygon exists for this place, so there is nothing to
                    # intersect the community area with: the shares are not
                    # merely unknown, they are unmeasurable.
                    containing_areas=(
                        ContainingArea(
                            number=int(row["target_value"]),
                            name=row["target_label"],
                            share_of_neighborhood=None,
                            share_of_area=None,
                        ),
                    ),
                )

    @classmethod
    def load(
        cls,
        areas_path: Path = NEIGHBORHOOD_AREAS_PATH,
        aliases_path: Path = NEIGHBORHOOD_ALIASES_PATH,
    ) -> Self:
        """Load the index from the vendored reference files.

        Args:
            areas_path: Path to the containment table.
            aliases_path: Path to the curated alias table.

        Returns:
            A ready-to-use index.

        Raises:
            FileNotFoundError: If either reference file is missing.
        """
        return cls(
            pd.read_csv(areas_path),
            pd.read_csv(aliases_path, dtype={"target_value": str}),
        )

    def _exact(self, name: str, match_kind: MatchKind) -> Candidate:
        """Build the candidate for one of the 98 polygon-backed neighborhoods."""
        return Candidate(
            match_kind=match_kind,
            label=name,
            geography="neighborhood",
            value=name,
            containing_areas=self._areas[name],
        )

    def resolve(self, query: str) -> Resolution:
        """Walk the ladder, first hit wins.

        Tiers 1-3 resolve: normalized exact match, curated alias to one of the
        98, curated pointer to a containing community area. Tier 4 only
        suggests, over both the stored names and the alias keys, so a typo on
        ``Bronzevile`` can still reach the row that keeps it away from
        Andersonville. An empty result is tier 5, the caller's cue to raise a
        teaching error listing :attr:`names`.

        Args:
            query: A neighborhood name as a person would type it.

        Returns:
            The resolution. Check :attr:`Resolution.resolved` before using a
            candidate's ``value``: a ``suggestion`` needs confirming first.
        """
        key = match_key(query)
        if key in self._by_key:
            return Resolution(query, (self._exact(self._by_key[key], "exact"),))
        if key in self._aliases:
            return Resolution(query, (self._aliases[key],))
        return Resolution(query, self._suggest(key))

    def _suggest(self, key: str) -> tuple[Candidate, ...]:
        """Rank near misses over the stored names and the alias keys alike."""
        # One dict of both shapes: `_by_key` holds names, `_aliases` holds
        # already-built candidates, and the loop below branches on which.
        known: dict[str, str | Candidate] = {**self._by_key, **self._aliases}
        near = difflib.get_close_matches(
            key, known, n=MAX_SUGGESTIONS, cutoff=NEAREST_CUTOFF
        )
        candidates = []
        for candidate_key in near:
            hit = known[candidate_key]
            base = self._exact(hit, "suggestion") if isinstance(hit, str) else hit
            score = difflib.SequenceMatcher(None, key, candidate_key).ratio()
            candidates.append(
                Candidate(
                    match_kind="suggestion",
                    label=base.label,
                    geography=base.geography,
                    value=base.value,
                    containing_areas=base.containing_areas,
                    score=round(score, 3),
                )
            )
        return tuple(candidates)
