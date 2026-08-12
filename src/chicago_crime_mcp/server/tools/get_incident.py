"""The ``get_incident`` tool: resolve one identifier to its full record(s).

Docstrings follow the Google Python style.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from chicago_crime_mcp.server.context import get_context
from chicago_crime_mcp.server.envelope import (
    Provenance,
    ResultWarning,
    RouteInfo,
    empty_result_warning,
    multiple_matches_warning,
)
from chicago_crime_mcp.server.errors import InvalidArgumentError
from chicago_crime_mcp.server.models import LookupFilters, LookupPayload, LookupResponse
from chicago_crime_mcp.store.postgres import queries


def get_incident(
    incident_id: Annotated[
        int | None,
        Field(description="The unique offense id, as returned by search_incidents."),
    ] = None,
    case_number: Annotated[
        str | None,
        Field(
            description="A CPD RD number. Case-insensitive. Identifies a report rather than an "
            "offense, so it can match several rows."
        ),
    ] = None,
) -> LookupResponse:
    """Fetch the full record for one offense id, or every offense sharing a case number.

    Returns every stored column, including the ones search results omit: the raw
    IUCR and FBI codes, the source's own spelling of the category, the State
    Plane coordinates, and when the record was last revised.

    Pass **exactly one** identifier.

    ``id`` is the only unique key. A ``case_number`` is CPD's RD number and
    identifies a *report*; a report that records several offenses has one row
    per offense, so a case-number lookup legitimately returns more than one and
    the response says so.

    Args:
        incident_id: The unique offense id.
        case_number: A CPD RD number.

    Returns:
        The matching offenses in full, wrapped in the standard result envelope.

    Raises:
        InvalidArgumentError: If neither or both identifiers were supplied.
    """
    if (incident_id is None) == (case_number is None):
        raise InvalidArgumentError(
            "Provide exactly one of incident_id or case_number.",
            field="incident_id" if incident_id is not None else "case_number",
            received={"incident_id": incident_id, "case_number": case_number},
            hint=(
                "They are not interchangeable: incident_id identifies one offense, case_number "
                "identifies a report and may cover several."
            ),
        )

    context = get_context()
    try:
        query = queries.LookupQuery(incident_id=incident_id, case_number=case_number)
    except ValueError as exc:  # pragma: no cover - the check above already covers it
        raise InvalidArgumentError.from_value_error(exc, field="incident_id") from exc

    with context.postgres() as conn:
        result = queries.lookup(conn, query)

    warnings: list[ResultWarning] = []
    if not result.incidents:
        warnings.append(
            empty_result_warning(
                filters=("incident_id",) if incident_id is not None else ("case_number",)
            )
        )
    elif result.query.case_number is not None and len(result.incidents) > 1:
        warnings.append(
            multiple_matches_warning(
                case_number=result.query.case_number, count=len(result.incidents)
            )
        )

    return LookupResponse(
        data=LookupPayload.from_store(result),
        filters_applied=LookupFilters.from_query(result.query),
        row_count=len(result.incidents),
        route=RouteInfo.from_store(result.timing, store="postgres"),
        # Deliberately unset: the full record carries *both* taxonomy columns,
        # so no single mode was applied and naming one would misdescribe it.
        taxonomy_mode=None,
        warnings=warnings,
        provenance=Provenance.from_dataset_meta(context.vocabulary().dataset),
    )


__all__ = ["get_incident"]
