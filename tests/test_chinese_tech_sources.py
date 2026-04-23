"""
Offline tests for Sina Tech and Xinhua Tech source adapters.
"""

from __future__ import annotations

import httpx
import pytest

from crawler.sources.sina_tech import (
    SINA_TECH_URL,
    extract_sina_tech_links,
    parse_sina_tech_article,
    search_sina_tech_articles,
)
from crawler.sources.xinhua_net import (
    XINHUA_TECH_URL,
    extract_xinhua_tech_links,
    parse_xinhua_article,
    search_xinhua_tech_articles,
)


def test_extract_sina_tech_links_filters_and_dedupes() -> None:
    html = """
    <a href="https://tech.sina.com.cn/csj/2026-04-23/doc-test.shtml">Tech</a>
    <a href="https://tech.sina.com.cn/csj/2026-04-23/doc-test.shtml#comments">Duplicate canonical</a>
    <a href="https://finance.sina.com.cn/tech/2026-04-23/doc-finance.shtml">Finance tech</a>
    <a href="https://sports.sina.com.cn/foo.shtml">Sports</a>
    <a href="/roll/2026-04-23/doc-relative.shtml">Relative</a>
    """

    assert extract_sina_tech_links(html) == [
        "https://tech.sina.com.cn/csj/2026-04-23/doc-test.shtml",
        "https://finance.sina.com.cn/tech/2026-04-23/doc-finance.shtml",
        "https://tech.sina.com.cn/roll/2026-04-23/doc-relative.shtml",
    ]


def test_extract_xinhua_tech_links_filters_and_dedupes() -> None:
    html = """
    <div id="focus">
      <a href="/tech/20260423/focus/c.html">Focus item should be ignored</a>
    </div>
    <div id="class">
      <a href="/tech/20260423/abc/c.html">Tech</a>
      <a href="https://www.news.cn/tech/20260423/abc/c.html#share">Duplicate canonical</a>
      <a href="https://www.news.cn/world/20260423/abc/c.html">World</a>
      <a href="https://www.news.cn/tech/20260423/not-article.html">Not article</a>
    </div>
    """

    assert extract_xinhua_tech_links(html) == [
        "https://www.news.cn/tech/20260423/abc/c.html",
    ]


def test_parse_sina_tech_article_maps_raw_article_fields() -> None:
    html = """
    <html>
      <head>
        <title>Ignored fallback - Sina</title>
        <meta property="og:title" content="Meta fallback title">
        <meta name="description" content="Short Sina lead.">
      </head>
      <body>
        <h1>Sina Tech headline</h1>
        <div>2026-04-23 09:30 Source: Sina Tech</div>
        <p>This is the first useful Sina technology paragraph for extraction.</p>
        <p>This is the second useful Sina technology paragraph for extraction.</p>
      </body>
    </html>
    """

    article = parse_sina_tech_article(
        html,
        web_url="https://tech.sina.com.cn/csj/2026-04-23/doc-test.shtml",
    )

    assert article.title == "Sina Tech headline"
    assert article.trail_text == "Short Sina lead."
    assert article.web_publication_date == "2026-04-23 09:30"
    assert article.section_name == "Sina Tech / Sina"
    assert "first useful Sina technology paragraph" in (article.body_text or "")
    assert "second useful Sina technology paragraph" in (article.body_text or "")


def test_parse_xinhua_article_maps_raw_article_fields() -> None:
    html = """
    <html>
      <head>
        <title>Fallback title - Xinhua</title>
        <meta property="og:title" content="Meta fallback title">
        <meta name="description" content="Short Xinhua lead.">
      </head>
      <body>
        <h1>Xinhua Tech headline</h1>
        <div>2026-04-23 10:45 Source: Xinhua</div>
        <p>This is the first useful Xinhua technology paragraph for extraction.</p>
        <p>This is the second useful Xinhua technology paragraph for extraction.</p>
      </body>
    </html>
    """

    article = parse_xinhua_article(
        html,
        web_url="https://www.news.cn/tech/20260423/abc/c.html",
    )

    assert article.title == "Xinhua Tech headline"
    assert article.trail_text == "Short Xinhua lead."
    assert article.web_publication_date == "2026-04-23 10:45"
    assert article.section_name == "Xinhua Tech / Xinhua"
    assert "first useful Xinhua technology paragraph" in (article.body_text or "")
    assert "second useful Xinhua technology paragraph" in (article.body_text or "")


def test_search_sina_tech_articles_uses_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = """
    <a href="https://tech.sina.com.cn/csj/2026-04-23/doc-test.shtml">Tech</a>
    """
    article = """
    <html><body>
      <h1>Sina fetched headline</h1>
      <div>2026-04-23 11:00 Source: Sina Tech</div>
      <p>This fetched Sina paragraph is long enough to keep.</p>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == SINA_TECH_URL:
            return httpx.Response(200, text=listing)
        return httpx.Response(200, text=article)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    class _ClientCtx:
        def __init__(self, timeout: float = 45.0, **kwargs: object) -> None:
            self._client = real_client(transport=transport, timeout=timeout, **kwargs)

        def __enter__(self) -> httpx.Client:
            return self._client

        def __exit__(self, *args: object) -> None:
            self._client.close()

    monkeypatch.setattr("crawler.sources.sina_tech.httpx.Client", _ClientCtx)
    page = search_sina_tech_articles(max_articles=1, article_delay_sec=0)

    assert page.status_code == 200
    assert page.article_urls == ["https://tech.sina.com.cn/csj/2026-04-23/doc-test.shtml"]
    assert len(page.articles) == 1
    assert page.articles[0].title == "Sina fetched headline"


def test_search_xinhua_tech_articles_uses_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = """
    <div id="focus">
      <a href="https://www.news.cn/tech/20260423/focus/c.html">Focus</a>
    </div>
    <div id="class">
      <a href="https://www.news.cn/tech/20260423/abc/c.html">Tech</a>
    </div>
    """
    article = """
    <html><body>
      <h1>Xinhua fetched headline</h1>
      <div>2026-04-23 11:30 Source: Xinhua</div>
      <p>This fetched Xinhua paragraph is long enough to keep.</p>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == XINHUA_TECH_URL:
            return httpx.Response(200, text=listing)
        return httpx.Response(200, text=article)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    class _ClientCtx:
        def __init__(self, timeout: float = 45.0, **kwargs: object) -> None:
            self._client = real_client(transport=transport, timeout=timeout, **kwargs)

        def __enter__(self) -> httpx.Client:
            return self._client

        def __exit__(self, *args: object) -> None:
            self._client.close()

    monkeypatch.setattr("crawler.sources.xinhua_net.httpx.Client", _ClientCtx)
    page = search_xinhua_tech_articles(max_articles=1, article_delay_sec=0)

    assert page.status_code == 200
    assert page.article_urls == ["https://www.news.cn/tech/20260423/abc/c.html"]
    assert len(page.articles) == 1
    assert page.articles[0].title == "Xinhua fetched headline"
