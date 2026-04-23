"""
New York Times Article Search API adapter.

This module mirrors the Guardian source adapter style so downstream crawler code
can consume NYT content using the shared `RawArticle` shape.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from core.config import NYT_API_BASE, NYT_API_KEY
from crawler.sources.guardian import RawArticle

DEFAULT_NYT_AI_GOVERNANCE_QUERY = (
    '"artificial intelligence" OR "AI safety" OR "AI governance" '
    'OR "AI regulation" OR "AI policy" OR "large language model"'
)

_DEFAULT_PAGE_DELAY_SEC = 1.0
_ARTICLE_SEARCH_PATH = "/svc/search/v2/articlesearch.json"


class NYTAPIError(Exception):
    """Raised when the NYT API request fails or returns an invalid payload."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


@dataclass(frozen=True)
class NYTSearchPage:
    """Normalized page result from the NYT Article Search API."""

    articles: List[RawArticle]
    hits: int
    offset: int
    current_page: int
    status: str


def _headline_main(doc: Dict[str, Any]) -> str:
    headline = doc.get("headline")
    if isinstance(headline, dict):
        main = headline.get("main")
        if isinstance(main, str) and main.strip():
            return main.strip()
    title = doc.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return "(no title)"


def _best_trail_text(doc: Dict[str, Any]) -> Optional[str]:
    for key in ("abstract", "snippet", "lead_paragraph"):
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def map_nyt_doc_to_raw_article(doc: Dict[str, Any]) -> RawArticle:
    """Map one NYT `docs[]` item into the shared `RawArticle` dataclass."""
    web_url = str(doc.get("web_url") or "").strip()
    lead = doc.get("lead_paragraph")
    lead_s = lead.strip() if isinstance(lead, str) and lead.strip() else None
    section_name = doc.get("section_name") or doc.get("news_desk") or None
    return RawArticle(
        web_url=web_url,
        title=_headline_main(doc),
        trail_text=_best_trail_text(doc),
        body_text=lead_s,
        web_publication_date=(doc.get("pub_date") or None),
        section_name=str(section_name).strip() if section_name else None,
        api_url=None,
        guardian_id=(doc.get("_id") or None),
    )


def _build_section_filter(section: Optional[str]) -> Optional[str]:
    if not section:
        return None
    sec = section.strip()
    if not sec:
        return None
    escaped = sec.replace('"', '\\"')
    return f'section_name:("{escaped}")'


def search_nyt_articles(
    *,
    query: Optional[str] = None,
    page: int = 0,
    begin_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort: str = "newest",
    section: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 45.0,
) -> NYTSearchPage:
    """
    Fetch one page from the NYT Article Search API.

    The NYT docs expose this endpoint at
    `https://api.nytimes.com/svc/search/v2/articlesearch.json`.
    """
    key = (api_key if api_key is not None else NYT_API_KEY).strip()
    if not key:
        raise NYTAPIError("NYT_API_KEY is not configured")

    root = (base_url if base_url is not None else NYT_API_BASE).rstrip("/")
    url = f"{root}{_ARTICLE_SEARCH_PATH}"
    q = (query if query is not None else DEFAULT_NYT_AI_GOVERNANCE_QUERY).strip()
    if not q:
        q = DEFAULT_NYT_AI_GOVERNANCE_QUERY

    params: Dict[str, Any] = {
        "api-key": key,
        "q": q,
        "page": max(0, min(99, int(page))),
        "sort": sort if sort in {"newest", "oldest"} else "newest",
    }
    if begin_date:
        params["begin_date"] = begin_date.strip()
    if end_date:
        params["end_date"] = end_date.strip()
    fq = _build_section_filter(section)
    if fq:
        params["fq"] = fq

    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, params=params)

    if resp.status_code == 429:
        raise NYTAPIError("NYT API rate limit exceeded (429)", status_code=429)
    if not resp.is_success:
        text = resp.text[:500] if resp.text else ""
        raise NYTAPIError(
            f"NYT API HTTP {resp.status_code}: {text}",
            status_code=resp.status_code,
        )

    try:
        data = resp.json()
    except Exception as e:
        raise NYTAPIError(f"NYT API returned non-JSON: {e}") from e

    response = data.get("response")
    if not isinstance(response, dict):
        raise NYTAPIError("NYT API payload missing response object", payload=data)
    docs = response.get("docs")
    if not isinstance(docs, list):
        docs = []
    meta = response.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    articles = [
        map_nyt_doc_to_raw_article(doc)
        for doc in docs
        if isinstance(doc, dict) and str(doc.get("web_url") or "").strip()
    ]
    return NYTSearchPage(
        articles=articles,
        hits=int(meta.get("hits") or 0),
        offset=int(meta.get("offset") or 0),
        current_page=max(0, min(99, int(page))),
        status=str(data.get("status") or ""),
    )


def search_nyt_articles_multipage(
    *,
    query: Optional[str] = None,
    max_pages: int = 3,
    begin_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort: str = "newest",
    section: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    page_delay_sec: float = _DEFAULT_PAGE_DELAY_SEC,
) -> List[RawArticle]:
    """Fetch multiple NYT Article Search pages and concatenate the results."""
    pages = max(1, min(100, int(max_pages)))
    all_rows: List[RawArticle] = []
    for page in range(pages):
        if page > 0:
            time.sleep(max(0.0, float(page_delay_sec)))
        result = search_nyt_articles(
            query=query,
            page=page,
            begin_date=begin_date,
            end_date=end_date,
            sort=sort,
            section=section,
            api_key=api_key,
            base_url=base_url,
        )
        all_rows.extend(result.articles)
        if not result.articles:
            break
    return all_rows
