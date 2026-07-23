"""Reusable Socrata (SODA) API client.

Dataset-agnostic: keyset pagination, retry/backoff, and count live here; the
crime-specific field list and canonicalization live in ``schema.py``. Every
request sends ``X-App-Token`` when ``SOCRATA_APP_TOKEN`` is set (per-token quota
instead of the shared anonymous throttle).

Keyset (seek) pagination is used instead of ``$offset`` so a large backfill is
O(1) per page and correct while the source dataset is being updated mid-pull.
It requires a unique, ordered key to anchor on - for the crime dataset that is
the numeric ``id`` column.

Docstrings follow the Google Python style (Args / Returns / Raises / Yields).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator

import httpx

log = logging.getLogger(__name__)

DOMAIN = "data.cityofchicago.org"
DEFAULT_PAGE_SIZE = 50_000  # Socrata's per-request maximum
RETRY_STATUS = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 6
MAX_BACKOFF_SECONDS = 30


class SodaError(RuntimeError):
    """A SODA request failed permanently (non-retryable status, or retries exhausted)."""


class SodaClient:
    """Thin wrapper over an ``httpx.Client`` for one Socrata domain.

    Attributes:
        domain: The Socrata host, e.g. ``"data.cityofchicago.org"``.
        has_token: Whether an app token was supplied (affects rate limits only).
    """

    def __init__(
        self,
        domain: str = DOMAIN,
        app_token: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialize the client and its underlying HTTP session.

        Args:
            domain: Socrata host to talk to.
            app_token: Socrata app token. Falls back to the ``SOCRATA_APP_TOKEN``
                environment variable. If neither is present, requests use the
                shared anonymous throttle and a warning is logged.
            timeout: Per-request timeout in seconds.
            transport: Optional ``httpx`` transport, used by tests to inject a
                mock backend. Production code leaves this ``None``.
        """
        self.domain = domain
        token = app_token or os.environ.get("SOCRATA_APP_TOKEN")
        headers = {"Accept": "application/json"}
        if token:
            headers["X-App-Token"] = token
        else:
            log.warning(
                "No SOCRATA_APP_TOKEN set - using the shared anonymous throttle "
                "(fine for small pulls, expect 429s on a full backfill)."
            )
        self._http = httpx.Client(headers=headers, timeout=timeout, transport=transport)
        self.has_token = bool(token)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._http.close()

    def __enter__(self) -> SodaClient:
        """Enter a context manager, returning ``self``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the context manager, closing the HTTP session."""
        self.close()

    # -- requests ----------------------------------------------------------

    def _url(self, dataset_id: str) -> str:
        """Build the JSON resource URL for a dataset id."""
        return f"https://{self.domain}/resource/{dataset_id}.json"

    def get(self, dataset_id: str, params: dict) -> list[dict]:
        """Perform one SODA request, retrying transient failures.

        Retries on HTTP 429 and 5xx with exponential backoff (doubling, capped at
        ``MAX_BACKOFF_SECONDS``), up to ``MAX_ATTEMPTS`` total attempts.

        Args:
            dataset_id: Socrata dataset identifier, e.g. ``"ijzp-q8t2"``.
            params: SoQL query parameters (``$select``, ``$where``, ``$limit`` ...).

        Returns:
            The decoded JSON body: a list of row dicts. Note SODA returns every
            value as a string (including numbers and booleans-as-``"true"``).

        Raises:
            SodaError: If the response is a non-retryable error status, or if all
                retry attempts are exhausted on retryable statuses.
        """
        url = self._url(dataset_id)
        for attempt in range(MAX_ATTEMPTS):
            r = self._http.get(url, params=params)
            if r.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS - 1:
                wait = min(2**attempt, MAX_BACKOFF_SECONDS)
                log.warning(
                    "SODA %s on %s - retry %d/%d in %ds",
                    r.status_code, dataset_id, attempt + 1, MAX_ATTEMPTS - 1, wait,
                )
                time.sleep(wait)
                continue
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise SodaError(
                    f"SODA request failed: {r.status_code} {r.text[:200]}"
                ) from e
            return r.json()
        raise SodaError(f"SODA request exhausted {MAX_ATTEMPTS} attempts for {dataset_id}")

    def count(self, dataset_id: str, where: str | None = None) -> int:
        """Count rows matching a predicate.

        Args:
            dataset_id: Socrata dataset identifier.
            where: Optional SoQL ``$where`` predicate. When ``None``, counts the
                whole dataset.

        Returns:
            The number of matching rows.

        Raises:
            SodaError: Propagated from :meth:`get` on request failure.
        """
        params: dict = {"$select": "count(1) as n"}
        if where:
            params["$where"] = where
        return int(self.get(dataset_id, params)[0]["n"])

    def paginate_keyset(
        self,
        dataset_id: str,
        select: str,
        where: str | None = None,
        order_key: str = "id",
        page_size: int = DEFAULT_PAGE_SIZE,
        cursor_numeric: bool = True,
    ) -> Iterator[list[dict]]:
        """Yield successive pages via keyset (seek) pagination on ``order_key``.

        Each request is ``... $order=<order_key> $where <order_key> > <cursor>``,
        seeking past the last row already seen rather than skipping N rows with
        ``$offset``. ``order_key`` must be unique for the paging to be correct
        (no skipped or duplicated rows across page boundaries).

        Args:
            dataset_id: Socrata dataset identifier.
            select: Comma-separated ``$select`` field list. If it omits
                ``order_key``, that column is appended so the next cursor can be
                read from each page.
            where: Optional base ``$where`` predicate, ANDed with the cursor
                clause on every page after the first.
            order_key: Unique, ordered column to page on. Defaults to ``"id"``.
            page_size: Rows per request (Socrata max is 50,000).
            cursor_numeric: If ``True`` the cursor is interpolated unquoted for a
                numeric comparison (correct for the numeric ``id`` column); if
                ``False`` it is single-quoted for a text comparison.

        Yields:
            One list of row dicts per page, in ascending ``order_key`` order.
            Iteration stops after a page smaller than ``page_size`` or an empty
            page, without issuing a further request.

        Raises:
            SodaError: Propagated from :meth:`get` on request failure.
        """
        # The order key must come back in each page so we can read the next cursor.
        selected = [s.strip() for s in select.split(",")]
        if order_key not in selected:
            select = f"{select},{order_key}"

        cursor: str | None = None
        while True:
            clause = where
            if cursor is not None:
                cmp = f"{order_key} > {cursor}" if cursor_numeric else f"{order_key} > '{cursor}'"
                clause = f"({where}) AND {cmp}" if where else cmp

            params: dict = {"$select": select, "$order": order_key, "$limit": page_size}
            if clause:
                params["$where"] = clause

            rows = self.get(dataset_id, params)
            if not rows:
                return
            yield rows

            cursor = rows[-1][order_key]
            if len(rows) < page_size:
                return
