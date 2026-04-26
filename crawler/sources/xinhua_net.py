"""
Xinhua News tech channel crawler.

This adapter scrapes the public Xinhua technology channel at
`https://www.news.cn/tech/` and maps article pages into the shared
`crawler.sources.guardian.RawArticle` shape used by downstream extraction.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from crawler.sources.guardian import RawArticle

XINHUA_TECH_URL = "https://www.news.cn/tech/"
_XINHUA_TECH_LIST_CONTAINER_ID = "list"
_DEFAULT_PAGE_DELAY_SEC = 1.0
_DEFAULT_ARTICLE_DELAY_SEC = 0.25
_DEFAULT_TIMEOUT_SEC = 45.0
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class XinhuaNetError(Exception):
    """Raised when a Xinhua page request or parse step fails."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class XinhuaTechPage:
    """Normalized listing result from one Xinhua technology channel page."""

    articles: List[RawArticle]
    article_urls: List[str]
    page_url: str
    status_code: int


class _LinkExtractor(HTMLParser):
    def __init__(self, *, target_id: Optional[str] = None) -> None:
        super().__init__(convert_charrefs=True)
        self._target_id = target_id
        self._target_depth = 0
        self.links: List[Tuple[str, str]] = []
        self._href_stack: List[Optional[str]] = []
        self._text_parts: List[str] = []

    def _in_target_scope(self) -> bool:
        return self._target_id is None or self._target_depth > 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_d = {k.lower(): v for k, v in attrs if k}
        tag_l = tag.lower()
        if self._target_id is not None:
            if self._target_depth > 0:
                self._target_depth += 1
            elif tag_l == "div" and attrs_d.get("id") == self._target_id:
                self._target_depth = 1
            else:
                return

        if tag_l != "a":
            return
        href = attrs_d.get("href")
        self._href_stack.append(href)
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_target_scope() and self._href_stack:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if self._in_target_scope() and tag_l == "a" and self._href_stack:
            href = self._href_stack.pop()
            text = _clean_text("".join(self._text_parts))
            self._text_parts = []
            if href:
                self.links.append((href, text))

        if self._target_id is not None and self._target_depth > 0:
            self._target_depth -= 1
        elif tag_l == "a" and self._href_stack:
            self._href_stack.pop()
            self._text_parts = []
            return


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: Dict[str, str] = {}
        self.h1_parts: List[str] = []
        self.title_parts: List[str] = []
        self.paragraphs: List[str] = []
        self.all_text_parts: List[str] = []
        self._capture_h1 = False
        self._capture_title = False
        self._p_depth = 0
        self._p_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_l = tag.lower()
        attrs_d = {k.lower(): v for k, v in attrs if k and v}
        if tag_l == "meta":
            key = attrs_d.get("property") or attrs_d.get("name")
            content = attrs_d.get("content")
            if key and content:
                self.meta[key.lower()] = _clean_text(content)
        elif tag_l == "h1":
            self._capture_h1 = True
        elif tag_l == "title":
            self._capture_title = True
        elif tag_l == "p":
            self._p_depth += 1
            self._p_parts = []

    def handle_data(self, data: str) -> None:
        self.all_text_parts.append(data)
        if self._capture_h1:
            self.h1_parts.append(data)
        if self._capture_title:
            self.title_parts.append(data)
        if self._p_depth:
            self._p_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l == "h1":
            self._capture_h1 = False
        elif tag_l == "title":
            self._capture_title = False
        elif tag_l == "p" and self._p_depth:
            self._p_depth -= 1
            text = _clean_text("".join(self._p_parts))
            if _looks_like_body_paragraph(text):
                self.paragraphs.append(text)
            self._p_parts = []


def _clean_text(value: str) -> str:
    value = unescape(value or "")
    return re.sub(r"\s+", " ", value).strip()


def _looks_like_body_paragraph(text: str) -> bool:
    if len(text) < 10:
        return False
    lowered = text.lower()
    blocked = (
        "copyright",
        "all rights reserved",
        "责任编辑",
        "分享到",
        "新华网",
        "客户端",
    )
    return not any(token in lowered for token in blocked)


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            "",
            parsed.query,
            "",
        )
    )


def _is_xinhua_tech_article(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return (
        host in {"www.news.cn", "news.cn", "xinhuanet.com", "www.xinhuanet.com"}
        and "/tech/" in path
        and path.endswith("/c.html")
    )


def extract_xinhua_tech_links(html: str, *, base_url: str = XINHUA_TECH_URL) -> List[str]:
    """Extract de-duplicated article URLs from Xinhua's `div#class` list."""
    parser = _LinkExtractor(target_id=_XINHUA_TECH_LIST_CONTAINER_ID)
    parser.feed(html)

    seen: set[str] = set()
    urls: List[str] = []
    for href, _text in parser.links:
        absolute = _canonical_url(urljoin(base_url, href.strip()))
        if not _is_xinhua_tech_article(absolute) or absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


def _extract_date(text: str) -> Optional[str]:
    match = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?", text)
    if match:
        return match.group(0)
    match = re.search(r"\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}", text)
    if match:
        return match.group(0)
    return None


def _extract_source(text: str) -> Optional[str]:
    match = re.search(r"(?:来源|Source)[:：]\s*([^ \n\r\t|｜]{2,40})", text)
    if match:
        return _clean_text(match.group(1))
    return None


def _best_title(parser: _ArticleParser) -> str:
    for value in (
        "".join(parser.h1_parts),
        parser.meta.get("og:title", ""),
        parser.meta.get("twitter:title", ""),
        "".join(parser.title_parts),
    ):
        title = _clean_text(value)
        if title:
            title = re.sub(r"[_-]新华网.*$", "", title).strip()
            return title or "(no title)"
    return "(no title)"


def parse_xinhua_article(html: str, *, web_url: str) -> RawArticle:
    """Parse one Xinhua article HTML document into `RawArticle`."""
    parser = _ArticleParser()
    parser.feed(html)

    all_text = _clean_text(" ".join(parser.all_text_parts))
    body_text = "\n".join(parser.paragraphs) or None
    description = (
        parser.meta.get("og:description")
        or parser.meta.get("description")
        or parser.meta.get("twitter:description")
    )
    trail_text = _clean_text(description or "") or (parser.paragraphs[0] if parser.paragraphs else None)
    source = _extract_source(all_text)

    return RawArticle(
        web_url=web_url,
        title=_best_title(parser),
        trail_text=trail_text,
        body_text=body_text,
        web_publication_date=(
            parser.meta.get("article:published_time")
            or parser.meta.get("pubdate")
            or _extract_date(all_text)
        ),
        section_name=f"Xinhua Tech{f' / {source}' if source else ''}",
        api_url=None,
        guardian_id=None,
    )


def _request_text(client: httpx.Client, url: str) -> Tuple[str, int]:
    resp = client.get(url)
    if resp.status_code == 429:
        raise XinhuaNetError("Xinhua returned rate limit status (429)", status_code=429)
    if not resp.is_success:
        text = resp.text[:300] if resp.text else ""
        raise XinhuaNetError(f"Xinhua HTTP {resp.status_code}: {text}", status_code=resp.status_code)
    return resp.text, resp.status_code


def fetch_xinhua_tech_article(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
) -> RawArticle:
    """Fetch and parse one Xinhua technology article URL."""
    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        html, _status = _request_text(client, url)
    return parse_xinhua_article(html, web_url=_canonical_url(url))


def search_xinhua_tech_articles(
    *,
    page_url: str = XINHUA_TECH_URL,
    max_articles: int = 10,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
    article_delay_sec: float = _DEFAULT_ARTICLE_DELAY_SEC,
) -> XinhuaTechPage:
    """
    Fetch one Xinhua tech channel page and parse up to `max_articles` articles.

    Xinhua does not require an API key for these public pages, so this function
    scrapes the channel links and then requests article pages directly.
    """
    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        html, status = _request_text(client, page_url)
        article_urls = extract_xinhua_tech_links(html, base_url=page_url)
        if max_articles > 0:
            article_urls = article_urls[: max(1, int(max_articles))]

        articles: List[RawArticle] = []
        for index, url in enumerate(article_urls):
            if index > 0:
                time.sleep(max(0.0, float(article_delay_sec)))
            article_html, _article_status = _request_text(client, url)
            articles.append(parse_xinhua_article(article_html, web_url=url))

    return XinhuaTechPage(
        articles=articles,
        article_urls=article_urls,
        page_url=page_url,
        status_code=status,
    )


def search_xinhua_tech_articles_multipage(
    *,
    page_urls: Optional[Iterable[str]] = None,
    max_articles_per_page: int = 10,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
    page_delay_sec: float = _DEFAULT_PAGE_DELAY_SEC,
    article_delay_sec: float = _DEFAULT_ARTICLE_DELAY_SEC,
) -> List[RawArticle]:
    """Fetch one or more Xinhua tech listing pages and concatenate articles."""
    urls = list(page_urls or [XINHUA_TECH_URL])
    all_rows: List[RawArticle] = []
    seen: set[str] = set()

    for index, url in enumerate(urls):
        if index > 0:
            time.sleep(max(0.0, float(page_delay_sec)))
        page = search_xinhua_tech_articles(
            page_url=url,
            max_articles=max_articles_per_page,
            timeout=timeout,
            article_delay_sec=article_delay_sec,
        )
        for article in page.articles:
            if article.web_url in seen:
                continue
            seen.add(article.web_url)
            all_rows.append(article)

    return all_rows


def main() -> None:
    """Debug entrypoint for manually checking the Xinhua Tech crawler."""
    parser = argparse.ArgumentParser(description="Debug crawl Xinhua Tech articles.")
    parser.add_argument(
        "--page-url",
        default=XINHUA_TECH_URL,
        help="Xinhua Tech listing URL to crawl.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="Maximum number of article pages to fetch.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_SEC,
        help="HTTP timeout in seconds.",
    )
    args = parser.parse_args()

    page = search_xinhua_tech_articles(
        page_url=args.page_url,
        max_articles=args.max_articles,
        timeout=args.timeout,
    )
    print(f"Fetched listing: {page.page_url} ({page.status_code})")
    print(f"Discovered article URLs: {len(page.article_urls)}")
    print(f"Parsed articles: {len(page.articles)}")

    for index, article in enumerate(page.articles, start=1):
        print()
        print(f"[{index}] {article.title}")
        print(f"URL: {article.web_url}")
        print(f"Published: {article.web_publication_date or '-'}")
        print(f"Section: {article.section_name or '-'}")
        if article.trail_text:
            print(f"Lead: {article.trail_text[:180]}")
        if article.body_text:
            print(f"Body: {article.body_text[:300]}")


if __name__ == "__main__":
    main()
