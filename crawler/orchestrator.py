"""
信源编排器：从卫报 Content API 拉取文章 → 并发 LLM 抽取 → 可选 RAG 精炼 → 入库。

功能：async_sync_guardian 并发执行「拉取 → URL 去重 → 并发抽取 → 关键词池 → 入库」，
     同步包装 sync_guardian 供脚本与 Streamlit 直接调用。
     并发度由 EXTRACT_CONCURRENCY（默认 5）控制，防止向 LLM 发起过多并发被限流。
输入：query/max_pages/section 等拉取参数；api_key/base_url 可覆盖 .env；rag_enabled 控制 RAG。
输出：SyncResult（入库数、跳过数、debug 日志）；副作用：Guardian HTTP + 并发 LLM + SQLite 写入。
上下游：上游为 app.py 或 scripts/sync_sources；下游 core.db.save_incident + 关键词池。
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.config import API_KEY, BASE_URL, DB_PATH
from core.db import (
    incident_from_extraction,
    save_incident,
    update_watched_keywords,
)
from core.llm_client import OpenAICompatibleBackend
from crawler.extraction import async_extract_incidents_from_text
from crawler.sources.guardian import (
    DEFAULT_AI_GOVERNANCE_QUERY,
    GuardianAPIError,
    RawArticle,
    raw_article_to_llm_context,
    search_articles_multipage,
)
# engine.rag_ingestion 依赖 chromadb（可选安装），延迟到运行时导入，
# 避免仅 import orchestrator 就要求 chromadb 存在。

# 并发 LLM 请求上限：防止向厂商发起过多并发被 429 限流。
EXTRACT_CONCURRENCY = 5


@dataclass
class SyncResult:
    """
    sync_guardian 的运行结果摘要。

    功能：便于 Streamlit UI 和脚本统一展示；包含入库数、跳过数与完整 debug 日志。
    """

    saved: int = 0
    skipped_url_dup: int = 0
    skipped_no_incident: int = 0
    failed: int = 0
    new_keywords: List[str] = field(default_factory=list)
    new_subdomains: List[str] = field(default_factory=list)
    debug_log: List[str] = field(default_factory=list)


def _url_already_in_db(web_url: str) -> bool:
    """
    功能：按 url 查 incidents 表是否已存在，用于去重（主键为 id，不保证 url 唯一，仅判断 >=1 条）。
    输入：文章网页 URL。
    输出：布尔；只读数据库。
    """
    if not web_url:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("SELECT 1 FROM incidents WHERE url = ? LIMIT 1", (web_url,))
        exists = cur.fetchone() is not None
        conn.close()
        return exists
    except Exception:
        return False


def _build_llm_backend(api_key: Optional[str], base_url: Optional[str]) -> OpenAICompatibleBackend:
    k = (api_key or "").strip() or API_KEY
    b = (base_url or "").strip() or BASE_URL
    return OpenAICompatibleBackend(api_key=k, base_url=b)


def _save_incidents_batch(
    incidents_rag: List[Dict[str, Any]],
    source_url: str,
    result: SyncResult,
    all_new_tags: List[str],
) -> None:
    """
    功能：将一篇文章的 incidents 写入 SQLite，收集标签与子域。
    输入：incidents_rag（已经过 RAG 精炼）、来源 URL、result（可变）、all_new_tags（可变）。
    输出：无；副作用：修改 result 与 all_new_tags、写 SQLite。
    """
    for inc_dict in incidents_rag:
        all_new_tags.extend(inc_dict.get("tags") or [])
        try:
            inc = incident_from_extraction(inc_dict)
            ok, tax_new = save_incident(inc, source_url=source_url)
            if ok:
                result.saved += 1
                if tax_new:
                    result.new_subdomains.append(inc.risk_subdomain)
                    result.debug_log.append(f"🆕 新子域: {inc.risk_subdomain}")
            else:
                result.debug_log.append(f"⚠️ 入库冲突（可能重复 id）: {inc.title[:40]}")
        except Exception as e:
            result.failed += 1
            result.debug_log.append(f"❌ incident 解析/入库失败: {type(e).__name__}: {e}")


async def async_sync_guardian(
    *,
    query: Optional[str] = None,
    max_pages: int = 2,
    page_size: int = 10,
    section: Optional[str] = None,
    show_fields: str = "trailText,bodyText",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    guardian_api_key: Optional[str] = None,
    rag_enabled: Optional[bool] = None,
    concurrency: int = EXTRACT_CONCURRENCY,
) -> SyncResult:
    """
    功能：同 sync_guardian，但 LLM 抽取步骤并发执行（asyncio.gather + Semaphore）。
    输入：concurrency 控制最大并发 LLM 请求数（默认 5）；其余参数同 sync_guardian。
    输出：SyncResult；副作用：Guardian HTTP（同步，在线程池）+ 并发 LLM 异步请求 + SQLite 写入。
    上下游：sync_guardian 通过 asyncio.run 调用；也可在已有事件循环中 await。
    """
    result = SyncResult()
    log = result.debug_log
    llm_backend = _build_llm_backend(api_key, base_url)

    # 1. 从卫报 API 拉文章列表（同步 httpx，放到线程池执行，不阻塞事件循环）
    try:
        log.append(f"📡 拉取 Guardian（query={query or DEFAULT_AI_GOVERNANCE_QUERY[:40]}... pages≤{max_pages}）")
        articles: List[RawArticle] = await asyncio.to_thread(
            search_articles_multipage,
            query=query,
            max_pages=max_pages,
            page_size=page_size,
            section=section,
            show_fields=show_fields,
            api_key=guardian_api_key,
        )
        log.append(f"✓ 拉取到 {len(articles)} 条（含可能重复）")
    except GuardianAPIError as e:
        log.append(f"❌ Guardian API 失败: {e}")
        result.failed += 1
        return result
    except Exception as e:
        log.append(f"❌ 拉取异常: {type(e).__name__}: {e}")
        result.failed += 1
        return result

    # 2. URL 去重（本次批次内也去重）
    seen_urls: set[str] = set()
    deduped: List[RawArticle] = []
    for art in articles:
        if not art.web_url:
            continue
        if art.web_url in seen_urls:
            continue
        seen_urls.add(art.web_url)
        if _url_already_in_db(art.web_url):
            result.skipped_url_dup += 1
            log.append(f"⏭ 已存在，跳过: {art.title[:60]}")
            continue
        deduped.append(art)
    log.append(f"✓ 去重后待处理 {len(deduped)} 篇")

    if not deduped:
        log.append("💡 本次无新文章需要处理")
        return result

    # 3. 并发 LLM 抽取（Semaphore 限制最大并发数，防止 429）
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _extract_one(art: RawArticle) -> Tuple[RawArticle, List[Dict[str, Any]], List[str]]:
        async with sem:
            context = raw_article_to_llm_context(art)
            incidents_raw, ext_log = await async_extract_incidents_from_text(
                context,
                source_url=art.web_url,
                backend=llm_backend,
            )
        return art, incidents_raw, ext_log

    log.append(f"🚀 并发抽取 {len(deduped)} 篇（并发上限 {concurrency}）...")
    extract_results = await asyncio.gather(*[_extract_one(a) for a in deduped], return_exceptions=False)

    # 4. 顺序处理抽取结果（RAG + 入库必须串行，SQLite 不支持并发写）
    all_new_tags: List[str] = []
    for art, incidents_raw, ext_log in extract_results:
        log.extend(ext_log)

        if not incidents_raw:
            result.skipped_no_incident += 1
            continue

        # RAG 精炼（可选）；延迟导入以支持未安装 chromadb 的轻量环境。
        try:
            from engine.rag_ingestion import apply_rag_to_incidents as _rag
            incidents_rag, rag_log = _rag(
                incidents_raw,
                llm_backend=llm_backend,
                enabled=rag_enabled,
            )
        except ImportError:
            incidents_rag = incidents_raw
            rag_log = ["⚠️ chromadb 未安装，RAG 步骤跳过"]
        log.extend(rag_log)

        _save_incidents_batch(incidents_rag, art.web_url, result, all_new_tags)

    # 5. 更新关键词池
    if all_new_tags:
        newly_added = update_watched_keywords(all_new_tags)
        result.new_keywords = newly_added
        log.append(f"📊 关键词池新增 {len(newly_added)} 条")

    log.append(
        f"✅ 完成 | 入库 {result.saved} 条，跳过已有 {result.skipped_url_dup}，"
        f"无关 {result.skipped_no_incident}，失败 {result.failed}"
    )
    return result


def sync_guardian(
    *,
    query: Optional[str] = None,
    max_pages: int = 2,
    page_size: int = 10,
    section: Optional[str] = None,
    show_fields: str = "trailText,bodyText",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    guardian_api_key: Optional[str] = None,
    rag_enabled: Optional[bool] = None,
    concurrency: int = EXTRACT_CONCURRENCY,
) -> SyncResult:
    """
    功能：sync_guardian 是 async_sync_guardian 的同步包装，供 CLI 脚本与 Streamlit 直接调用。
    输入：同 async_sync_guardian。
    输出：SyncResult；副作用同 async_sync_guardian。
    上下游：scripts/sync_sources.py、app.py；不应在已有事件循环中调用（用 await async_sync_guardian 代替）。
    """
    return asyncio.run(
        async_sync_guardian(
            query=query,
            max_pages=max_pages,
            page_size=page_size,
            section=section,
            show_fields=show_fields,
            api_key=api_key,
            base_url=base_url,
            guardian_api_key=guardian_api_key,
            rag_enabled=rag_enabled,
            concurrency=concurrency,
        )
    )
