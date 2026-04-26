"""
Sina Tech crawler.

Fetches article links from https://tech.sina.com.cn/ and maps article pages
into the shared RawArticle shape used by the crawler pipeline.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from crawler.sources.guardian import RawArticle

SINA_TECH_URL = "https://tech.sina.com.cn/"
_DEFAULT_TIMEOUT_SEC = 45.0
_DEFAULT_ARTICLE_DELAY_SEC = 0.25
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class SinaTechError(Exception):
    """Raised when a Sina Tech page request or parse step fails."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SinaTechPage:
    articles: List[RawArticle]
    article_urls: List[str]
    page_url: str
    status_code: int


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[Tuple[str, str]] = []
        self._href_stack: List[Optional[str]] = []
        self._text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        attrs_d = {k.lower(): v for k, v in attrs if k}
        self._href_stack.append(attrs_d.get("href"))
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href_stack:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._href_stack:
            return
        href = self._href_stack.pop()
        text = _clean_text("".join(self._text_parts))
        self._text_parts = []
        if href:
            self.links.append((href, text))


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
    blocked = (
        "copyright",
        "all rights reserved",
        "责任编辑",
        "新浪简介",
        "广告服务",
        "联系我们",
    )
    return not any(token in text.lower() for token in blocked)


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, ""))


def _is_sina_tech_article(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    allowed_hosts = {
        "tech.sina.com.cn",
        "finance.sina.com.cn",
        "mobile.sina.com.cn",
        "digi.sina.com.cn",
        "zhongce.sina.com.cn",
    }
    return host in allowed_hosts and path.endswith((".shtml", ".html"))


def extract_sina_tech_links(html: str, *, base_url: str = SINA_TECH_URL) -> List[str]:
    """Extract de-duplicated Sina tech article URLs from a listing page."""
    parser = _LinkExtractor()
    parser.feed(html)

    seen: set[str] = set()
    urls: List[str] = []
    for href, _text in parser.links:
        absolute = _canonical_url(urljoin(base_url, href.strip()))
        if not _is_sina_tech_article(absolute) or absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


def _best_title(parser: _ArticleParser) -> str:
    for value in (
        "".join(parser.h1_parts),
        parser.meta.get("og:title", ""),
        parser.meta.get("twitter:title", ""),
        "".join(parser.title_parts),
    ):
        title = _clean_text(value)
        if title:
            title = re.sub(r"[_-].*新浪.*$", "", title).strip()
            return title or "(no title)"
    return "(no title)"


def _extract_date(text: str) -> Optional[str]:
    match = re.search(r"\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}", text)
    if match:
        return match.group(0)
    match = re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?", text)
    if match:
        return match.group(0)
    return None


def _extract_source(text: str) -> Optional[str]:
    match = re.search(r"(?:来源|Source)[:：]\s*([^ \n\r\t|｜]{2,40})", text)
    if match:
        return _clean_text(match.group(1))
    return None


def parse_sina_tech_article(html: str, *, web_url: str) -> RawArticle:
    """Parse one Sina article HTML document into RawArticle."""
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
            or parser.meta.get("weibo:article:create_at")
            or _extract_date(all_text)
        ),
        section_name=f"Sina Tech{f' / {source}' if source else ''}",
        api_url=None,
        guardian_id=None,
    )


def _request_text(client: httpx.Client, url: str) -> Tuple[str, int]:
    resp = client.get(url)
    if resp.status_code == 429:
        raise SinaTechError("Sina Tech returned rate limit status (429)", status_code=429)
    if not resp.is_success:
        text = resp.text[:300] if resp.text else ""
        raise SinaTechError(f"Sina Tech HTTP {resp.status_code}: {text}", status_code=resp.status_code)
    return resp.text, resp.status_code


def fetch_sina_tech_article(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
) -> RawArticle:
    """Fetch and parse one Sina Tech article URL."""
    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        html, _status = _request_text(client, url)
    return parse_sina_tech_article(html, web_url=_canonical_url(url))


def search_sina_tech_articles(
    *,
    page_url: str = SINA_TECH_URL,
    max_articles: int = 10,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
    article_delay_sec: float = _DEFAULT_ARTICLE_DELAY_SEC,
) -> SinaTechPage:
    """Fetch the Sina Tech listing page and parse up to max_articles articles."""
    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        html, status = _request_text(client, page_url)
        article_urls = extract_sina_tech_links(html, base_url=page_url)
        if max_articles > 0:
            article_urls = article_urls[: max(1, int(max_articles))]

        articles: List[RawArticle] = []
        for index, url in enumerate(article_urls):
            if index > 0:
                time.sleep(max(0.0, float(article_delay_sec)))
            article_html, _article_status = _request_text(client, url)
            articles.append(parse_sina_tech_article(article_html, web_url=url))

    return SinaTechPage(
        articles=articles,
        article_urls=article_urls,
        page_url=page_url,
        status_code=status,
    )
