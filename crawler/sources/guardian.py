"""
The Guardian Content API 客户端（search）：拉取与 AI 安全、治理相关的新闻条目。

功能：用 httpx 调用 /search，解析 results 为 RawArticle；支持 show-fields 获取导语与正文。
输入：可选覆盖 api_key/base_url（默认 core.config）；query 默认指向 AI 安全与治理相关英文检索词。
输出：RawArticle 列表与分页元数据；失败抛出 GuardianAPIError。
上下游：供 orchestrator、scripts/smoke_guardian 调用；下游可将 RawArticle 拼文本后走 LLM 抽取（models.ExtractionResult）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from core.config import GUARDIAN_API_BASE, GUARDIAN_API_KEY

# ---------------------------------------------------------------------------
# 默认检索：卫报为英文稿，用英文关键词覆盖 AI 安全、治理、监管、伦理与对齐等语义。
# 说明：search 为相关性排序，不保证条条命中；后续可由 LLM 按 ExtractionResult 再过滤。
# ---------------------------------------------------------------------------
# 不使用 OR/括号布尔语法（各环境索引行为不一），用高密度英文词偏向 AI 安全、治理、监管与伦理报道。
#人工智能、AI 安全、治理、监管、政策、伦理、对齐、机器学习、大语言模型
DEFAULT_AI_GOVERNANCE_QUERY = (
    "artificial intelligence AI safety governance regulation policy ethics alignment "
    "machine learning large language model"
)

# Developer 档建议约 1 请求/秒；翻页时使用。
_DEFAULT_PAGE_DELAY_SEC = 1.0


class GuardianAPIError(Exception):
    """
    卫报 API 调用失败（HTTP 非 2xx、429、或 response.status != ok）。

    功能：携带状态码与响应片段便于日志与 UI 展示。
    输入：构造时传入 message，可选 status_code、payload。
    输出：异常；无 IO。
    """

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
class RawArticle:
    """
    单条卫报 search 结果的标准化视图（供下游抽取与去重）。

    功能：从 API 的 result 项映射为稳定字段；正文/导语可能为空（视 show-fields 与条目类型）。
    输入：由 map_result_to_raw_article 从 JSON 构造。
    输出：只读数据对象。
    """

    web_url: str
    title: str
    trail_text: Optional[str]
    body_text: Optional[str]
    web_publication_date: Optional[str]
    section_name: Optional[str]
    api_url: Optional[str]
    guardian_id: Optional[str]


@dataclass(frozen=True)
class GuardianSearchPage:
    """
    单次 search 响应的分页与结果封装。

    功能：便于编排器多页拉取与日志。
    输入：由 search_articles 返回。
    输出：只读数据对象。
    """

    articles: List[RawArticle]
    total: int
    page_size: int
    current_page: int
    pages: int
    status: str


def _get_fields_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    """从 result 项解析 fields（API 可能给 dict 或极少情况需兼容）。"""
    raw = item.get("fields")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    return {}


def map_result_to_raw_article(item: Dict[str, Any]) -> RawArticle:
    """
    功能：将单条 Guardian search `results[]` 元素转为 RawArticle。
    输入：JSON dict。
    输出：RawArticle；web_url 缺失时退化为 apiUrl 或空串（调用方应过滤）。
    """
    fields = _get_fields_dict(item)
    web_url = (item.get("webUrl") or "").strip()
    if not web_url:
        web_url = (item.get("apiUrl") or "").strip()
    title = (item.get("webTitle") or item.get("title") or "").strip() or "(no title)"
    trail = fields.get("trailText")
    body = fields.get("bodyText")
    trail_s = trail.strip() if isinstance(trail, str) else None
    body_s = body.strip() if isinstance(body, str) else None
    if trail_s == "":
        trail_s = None
    if body_s == "":
        body_s = None
    return RawArticle(
        web_url=web_url,
        title=title,
        trail_text=trail_s,
        body_text=body_s,
        web_publication_date=(item.get("webPublicationDate") or None),
        section_name=(item.get("sectionName") or None),
        api_url=(item.get("apiUrl") or None),
        guardian_id=(item.get("id") or None),
    )


def search_articles(
    *,
    query: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    section: Optional[str] = None,
    show_fields: str = "trailText,bodyText",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 45.0,
) -> GuardianSearchPage:
    """
    功能：调用 GET /search 拉取一页结果；默认 query 面向 AI 安全与治理相关报道。
    输入：query 默认 DEFAULT_AI_GOVERNANCE_QUERY；page_size 建议不超过 50；section 可选如 technology。
    输出：GuardianSearchPage；429/4xx/5xx 或 response.status!=ok 时抛 GuardianAPIError。
    副作用：一次 HTTP GET。
    """
    key = (api_key if api_key is not None else GUARDIAN_API_KEY).strip()
    if not key:
        raise GuardianAPIError("GUARDIAN_API_KEY 未配置，请在 .env 中设置")

    root = (base_url if base_url is not None else GUARDIAN_API_BASE).rstrip("/")
    url = f"{root}/search"
    q = (query if query is not None else DEFAULT_AI_GOVERNANCE_QUERY).strip()
    if not q:
        q = DEFAULT_AI_GOVERNANCE_QUERY

    params: Dict[str, Any] = {
        "api-key": key,
        "q": q,
        "page": max(1, int(page)),
        "page-size": max(1, min(50, int(page_size))),
        "show-fields": show_fields,
    }
    if section:
        params["section"] = section.strip()

    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, params=params)

    if resp.status_code == 429:
        raise GuardianAPIError(
            "卫报 API 频率或日配额限制（429），请降低请求频率或次日再试",
            status_code=429,
        )

    if not resp.is_success:
        text = resp.text[:500] if resp.text else ""
        raise GuardianAPIError(
            f"卫报 API HTTP {resp.status_code}: {text}",
            status_code=resp.status_code,
        )

    try:
        data = resp.json()
    except Exception as e:
        raise GuardianAPIError(f"卫报 API 返回非 JSON: {e}") from e

    response = data.get("response") if isinstance(data, dict) else None
    if not isinstance(response, dict):
        raise GuardianAPIError("卫报 API 响应缺少 response 对象", payload=data if isinstance(data, dict) else {})

    status = str(response.get("status", ""))
    if status != "ok":
        msg = str(response.get("message") or response.get("status") or "unknown error")
        raise GuardianAPIError(f"卫报 API status 非 ok: {msg}", payload=response)

    results = response.get("results")
    if not isinstance(results, list):
        results = []

    articles = [map_result_to_raw_article(item) for item in results if isinstance(item, dict)]
    articles = [a for a in articles if a.web_url]

    total = int(response.get("total") or 0)
    page_size_out = int(response.get("pageSize") or page_size)
    current_page = int(response.get("currentPage") or page)
    pages = int(response.get("pages") or 1)

    return GuardianSearchPage(
        articles=articles,
        total=total,
        page_size=page_size_out,
        current_page=current_page,
        pages=max(1, pages),
        status=status,
    )


def search_articles_multipage(
    *,
    query: Optional[str] = None,
    max_pages: int = 3,
    page_size: int = 10,
    section: Optional[str] = None,
    show_fields: str = "trailText,bodyText",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    page_delay_sec: float = _DEFAULT_PAGE_DELAY_SEC,
) -> List[RawArticle]:
    """
    功能：按页拉取多页 search 结果并合并为列表（页间 sleep 以降低 429 风险）。
    输入：max_pages 上限；其余同 search_articles。
    输出：RawArticle 列表（多页拼接，可能含重复 URL，调用方可按 web_url 去重）。
    副作用：多次 HTTP GET 与 sleep。
    """
    max_pages = max(1, int(max_pages))
    all_rows: List[RawArticle] = []
    first = search_articles(
        query=query,
        page=1,
        page_size=page_size,
        section=section,
        show_fields=show_fields,
        api_key=api_key,
        base_url=base_url,
    )
    all_rows.extend(first.articles)
    pages_to_fetch = min(max_pages, first.pages)

    for p in range(2, pages_to_fetch + 1):
        time.sleep(max(0.0, float(page_delay_sec)))
        page_result = search_articles(
            query=query,
            page=p,
            page_size=page_size,
            section=section,
            show_fields=show_fields,
            api_key=api_key,
            base_url=base_url,
        )
        all_rows.extend(page_result.articles)

    return all_rows


def raw_article_to_llm_context(article: RawArticle) -> str:
    """
    功能：将 RawArticle 拼成一段供 LLM 抽取的上下文（标题 + 导语 + 正文优先）。
    输入：RawArticle。
    输出：单字符串；无 IO。
    """
    parts: List[str] = [f"Title: {article.title}"]
    if article.trail_text:
        parts.append(f"Lead: {article.trail_text}")
    if article.body_text:
        parts.append(f"Body:\n{article.body_text}")
    elif not article.trail_text:
        parts.append("(No body or trail text in API response.)")
    return "\n\n".join(parts)
