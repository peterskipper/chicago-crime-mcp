"""Unit tests for the Socrata client.

No network: every test injects an ``httpx.MockTransport`` backend, so we assert
on the SoQL we *send* and how the client *behaves* (retries, cursor advance),
not on Chicago's servers.
"""

from __future__ import annotations

import re

import httpx
import pytest

from chicago_crime_mcp.ingest import socrata
from chicago_crime_mcp.ingest.socrata import MAX_ATTEMPTS, SodaClient, SodaError

DS = "ijzp-q8t2"


def client(handler, **kwargs) -> SodaClient:
    """Build a SodaClient wired to a mock transport handler."""
    kwargs.setdefault("app_token", "TESTTOKEN")
    return SodaClient(transport=httpx.MockTransport(handler), **kwargs)


# -- token header ----------------------------------------------------------


def test_app_token_header_sent_when_present():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=[])

    c = client(handler, app_token="SECRET123")
    c.get(DS, {"$limit": "1"})
    assert captured[0].headers.get("X-App-Token") == "SECRET123"
    assert c.has_token is True


def test_no_token_header_absent(monkeypatch):
    monkeypatch.delenv("SOCRATA_APP_TOKEN", raising=False)
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=[])

    c = SodaClient(transport=httpx.MockTransport(handler), app_token=None)
    c.get(DS, {"$limit": "1"})
    assert "X-App-Token" not in captured[0].headers
    assert c.has_token is False


# -- retry / error handling ------------------------------------------------


def test_get_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(socrata.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=[{"ok": "1"}])

    result = client(handler).get(DS, {"$limit": "1"})
    assert result == [{"ok": "1"}]
    assert calls["n"] == 3  # two 429s then success


def test_get_exhausts_retries_and_raises(monkeypatch):
    sleeps = []
    monkeypatch.setattr(socrata.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    with pytest.raises(SodaError):
        client(handler).get(DS, {"$limit": "1"})
    assert calls["n"] == MAX_ATTEMPTS
    assert len(sleeps) == MAX_ATTEMPTS - 1  # no sleep after the final attempt


def test_get_non_retryable_status_raises_immediately():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad SoQL")

    with pytest.raises(SodaError):
        client(handler).get(DS, {"$where": "bogus"})
    assert calls["n"] == 1  # 400 is not retried


# -- count -----------------------------------------------------------------


def test_count_parses_and_forwards_where():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json=[{"n": "1034"}])

    n = client(handler).count(DS, where="year=2024")
    assert n == 1034
    assert captured[0].url.params["$select"] == "count(1) as n"
    assert captured[0].url.params["$where"] == "year=2024"


# -- keyset pagination -----------------------------------------------------


def numeric_backend(all_ids, captured, order_key="id"):
    """A fake dataset that honors keyset paging on a numeric column."""

    def handler(request):
        captured.append(request)
        params = request.url.params
        limit = int(params["$limit"])
        where = params.get("$where", "")
        m = re.search(rf"{order_key} > (\d+)", where)
        lo = int(m.group(1)) if m else 0
        rows = [i for i in all_ids if i > lo][:limit]
        return httpx.Response(
            200, json=[{"id": str(i), "primary_type": "THEFT"} for i in rows]
        )

    return handler


def test_paginate_pages_and_advances_cursor():
    captured = []
    handler = numeric_backend(list(range(1, 251)), captured)  # ids 1..250
    pages = list(
        client(handler).paginate_keyset(DS, select="id,primary_type", page_size=100)
    )

    assert [len(p) for p in pages] == [100, 100, 50]
    # 3 requests only: the 50-row short page stops iteration, no 4th request.
    assert len(captured) == 3
    # first request has no cursor; subsequent ones seek past the last id seen.
    assert "$where" not in captured[0].url.params
    assert captured[1].url.params["$where"] == "id > 100"
    assert captured[2].url.params["$where"] == "id > 200"
    # global correctness: unique, ascending, complete.
    ids = [int(r["id"]) for p in pages for r in p]
    assert ids == sorted(ids) == list(range(1, 251))


def test_paginate_stops_on_empty_page():
    captured = []
    handler = numeric_backend([], captured)  # nothing matches
    pages = list(client(handler).paginate_keyset(DS, select="id", page_size=100))
    assert pages == []
    assert len(captured) == 1  # one request, empty result, stop


def test_paginate_injects_order_key_into_select():
    captured = []
    handler = numeric_backend([1, 2, 3], captured)
    list(client(handler).paginate_keyset(DS, select="primary_type", page_size=100))
    # 'id' was appended so the cursor is readable even though caller omitted it.
    assert "id" in captured[0].url.params["$select"].split(",")


def test_paginate_combines_existing_where_with_cursor():
    captured = []
    handler = numeric_backend(list(range(1, 151)), captured)  # ids 1..150
    list(
        client(handler).paginate_keyset(
            DS, select="id", where="year=2024", page_size=100
        )
    )
    assert captured[0].url.params["$where"] == "year=2024"
    assert captured[1].url.params["$where"] == "(year=2024) AND id > 100"


def test_paginate_text_cursor_is_quoted():
    captured = []

    def handler(request):
        captured.append(request)
        params = request.url.params
        where = params.get("$where", "")
        # page 1: no cursor -> return two rows; page 2: cursor present -> empty.
        if "case_number >" in where:
            return httpx.Response(200, json=[])
        return httpx.Response(
            200, json=[{"case_number": "JA100"}, {"case_number": "JA200"}]
        )

    list(
        client(handler).paginate_keyset(
            DS,
            select="case_number",
            order_key="case_number",
            page_size=2,
            cursor_numeric=False,
        )
    )
    # text cursor must be single-quoted for a string comparison.
    assert captured[1].url.params["$where"] == "case_number > 'JA200'"
