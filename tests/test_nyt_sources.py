"""
crawler.sources.nyt unit tests: parsing and API error branches without real HTTP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from crawler.sources.nyt import NYTAPIError, map_nyt_doc_to_raw_article, search_nyt_articles


def test_map_nyt_doc_to_raw_article_parses_fields() -> None:
    doc = {
        "_id": "nyt://article/123",
        "web_url": "https://www.nytimes.com/2026/04/15/technology/ai-policy.html",
        "headline": {"main": "AI policy debate intensifies"},
        "abstract": "A short abstract.",
        "lead_paragraph": "A longer lead paragraph.",
        "pub_date": "2026-04-15T12:00:00Z",
        "section_name": "Technology",
    }
    a = map_nyt_doc_to_raw_article(doc)
    assert a.title == "AI policy debate intensifies"
    assert a.web_url.startswith("https://www.nytimes.com/")
    assert a.trail_text == "A short abstract."
    assert a.body_text == "A longer lead paragraph."
    assert a.section_name == "Technology"
    assert a.guardian_id == "nyt://article/123"


def test_search_nyt_articles_raises_when_key_missing() -> None:
    with pytest.raises(NYTAPIError, match="NYT_API_KEY"):
        search_nyt_articles(api_key="")


def test_search_nyt_articles_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "status": "OK",
        "response": {
            "docs": [
                {
                    "_id": "nyt://article/abc",
                    "web_url": "https://www.nytimes.com/2026/04/15/technology/test.html",
                    "headline": {"main": "Hello NYT"},
                    "abstract": "Lead text",
                }
            ],
            "meta": {
                "hits": 1,
                "offset": 0,
            },
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/svc/search/v2/articlesearch.json"
        assert "api-key" in str(request.url)
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    class _ClientCtx:
        def __init__(self, timeout: float = 45.0, **_kwargs: object) -> None:
            self._client = real_client(transport=transport, timeout=timeout)

        def __enter__(self) -> httpx.Client:
            return self._client

        def __exit__(self, *args: object) -> None:
            self._client.close()

    monkeypatch.setattr("crawler.sources.nyt.httpx.Client", _ClientCtx)
    page = search_nyt_articles(api_key="test-key", base_url="https://api.nytimes.com")
    assert page.status == "OK"
    assert page.hits == 1
    assert len(page.articles) == 1
    assert page.articles[0].title == "Hello NYT"
