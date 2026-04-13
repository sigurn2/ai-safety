"""
crawler.sources.guardian 单元测试：解析与错误分支（不发起真实网络请求）。
"""

from __future__ import annotations

import httpx
import pytest

from crawler.sources.guardian import (
    GuardianAPIError,
    map_result_to_raw_article,
    search_articles,
)


def test_map_result_to_raw_article_parses_fields() -> None:
    item = {
        "id": "tech/2025/jan/01/foo",
        "webTitle": "Test AI regulation headline",
        "webUrl": "https://www.theguardian.com/technology/2025/jan/01/foo",
        "apiUrl": "https://content.guardianapis.com/technology/2025/jan/01/foo",
        "webPublicationDate": "2025-01-01T12:00:00Z",
        "sectionName": "Technology",
        "fields": {"trailText": "Lead paragraph.", "bodyText": "Full body here."},
    }
    a = map_result_to_raw_article(item)
    assert a.title == "Test AI regulation headline"
    assert a.web_url.startswith("https://www.theguardian.com/")
    assert a.trail_text == "Lead paragraph."
    assert a.body_text == "Full body here."
    assert a.section_name == "Technology"
    assert a.guardian_id == "tech/2025/jan/01/foo"


def test_map_result_falls_back_api_url() -> None:
    item = {
        "webTitle": "No webUrl",
        "apiUrl": "https://content.guardianapis.com/world/2025/jan/02/bar",
        "fields": {},
    }
    a = map_result_to_raw_article(item)
    assert a.web_url == "https://content.guardianapis.com/world/2025/jan/02/bar"


def test_search_articles_raises_when_key_missing() -> None:
    with pytest.raises(GuardianAPIError, match="GUARDIAN_API_KEY"):
        search_articles(api_key="")


def test_search_articles_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "response": {
            "status": "ok",
            "total": 1,
            "pageSize": 10,
            "currentPage": 1,
            "pages": 1,
            "results": [
                {
                    "id": "x",
                    "webTitle": "Hello",
                    "webUrl": "https://www.theguardian.com/a",
                    "fields": {"trailText": "T"},
                }
            ],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert "api-key" in str(request.url)
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    RealClient = httpx.Client

    class _ClientCtx:
        def __init__(self, timeout: float = 45.0, **_kwargs: object) -> None:
            self._client = RealClient(transport=transport, timeout=timeout)

        def __enter__(self) -> httpx.Client:
            return self._client

        def __exit__(self, *args: object) -> None:
            self._client.close()

    monkeypatch.setattr("crawler.sources.guardian.httpx.Client", _ClientCtx)
    page = search_articles(api_key="test-key", base_url="https://content.guardianapis.com")
    assert page.status == "ok"
    assert len(page.articles) == 1
    assert page.articles[0].title == "Hello"
