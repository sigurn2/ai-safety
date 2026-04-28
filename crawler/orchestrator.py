"""
信源编排器：从卫报 Content API 拉取文章 → 并发 LLM 抽取 → 可选 RAG 精炼 → 入库。

功能：async_sync_guardian 并发执行「拉取 → URL 去重（MySQL）→ 并发抽取 → 写 MySQL + Chroma」，
     同步包装 sync_guardian 供脚本与 Streamlit 直接调用。
     并发度由 EXTRACT_CONCURRENCY（默认 5）控制，防止向 LLM 发起过多并发被限流。
输入：query/max_pages/section 等拉取参数；api_key/base_url 可覆盖 .env；rag_enabled 控制 RAG。
输出：SyncResult（MySQL 入库篇数、跳过数、debug 日志）；副作用：Guardian HTTP + 并发 LLM + MySQL/Chroma。
上下游：app.py、scripts/sync_sources；下游 core.mysql_db。
说明：RAG 子域路由若开启，仍读取 core.db（SQLite）中 risk_taxonomy；不需要时可设 RAG_ENABLED=false。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.config import API_KEY, BASE_URL, LLM_MODEL
from core.llm_client import OpenAICompatibleBackend
from crawler.extraction import (
    async_extract_article_from_text,
    article_dict_to_incident_like,
    merge_article_with_rag,
)
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


def _persist_mysql_phase1(
    art: RawArticle,
    merged_extraction: Dict[str, Any],
    result: SyncResult,
    llm_backend: Optional[OpenAICompatibleBackend] = None,
    *,
    force_reindex: bool = False,
) -> None:
    """
    Persist article + one article-level extraction row to MySQL（MVP：不写事件/专题表）。
    This path is best-effort；成功写入 extraction 时递增 result.saved。
    """
    if not merged_extraction:
        return
    try:
        from core.mysql_db import save_article, save_extraction
    except Exception as e:
        result.debug_log.append(f"⚠️ MySQL Phase1 未启用: {type(e).__name__}: {e}")
        return

    summary = art.trail_text or (art.body_text or "")[:512]
    content = art.body_text or art.trail_text or ""

    # 解析原文发布时间（Guardian API 返回 ISO 8601 格式，如 "2025-03-15T12:00:00Z"）
    published_at: Optional[datetime] = None
    if art.web_publication_date:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(art.web_publication_date[:19], fmt[:19])
                published_at = dt
                break
            except ValueError:
                continue

    try:
        article_id, is_new = save_article(
            url=art.web_url,
            title=art.title,
            summary=summary,
            content=content,
            published_at=published_at,
            source="guardian",
        )
    except Exception as e:
        result.debug_log.append(f"⚠️ MySQL article 写入失败: {type(e).__name__}: {e}")
        return

    # 写入全文向量索引（best-effort，不中断主流程）
    if is_new or force_reindex:
        try:
            from engine.article_index.indexer import index_article
            # 取首条 incident 的 risk_domain 作为 chunk metadata（多条时取第一条）
            first_domain = str(merged_extraction.get("risk_domain", "")).strip()
            pub_at_str = published_at.strftime("%Y-%m-%d") if published_at else ""
            n_chunks = index_article(
                article_id=article_id,
                title=art.title,
                summary=summary,
                content=content,
                source="guardian",
                risk_domain=first_domain,
                published_at=pub_at_str,
                url=art.web_url,
                backend=llm_backend,
                extraction_ctx=merged_extraction,
            )
            result.debug_log.append(f"🔢 向量索引写入 {n_chunks} chunks (article_id={article_id})")
        except Exception as e:
            result.debug_log.append(f"⚠️ 向量索引写入失败: {type(e).__name__}: {e}")

    try:
        extraction_id = save_extraction(
            article_id=article_id,
            extraction_dict=merged_extraction,
            model_name=(LLM_MODEL or "").strip(),
        )
        result.saved += 1
        if is_new:
            result.debug_log.append(f"💾 MySQL 写入 article_id={article_id}, extraction_id={extraction_id}")
        elif force_reindex:
            result.debug_log.append(
                f"💾 MySQL 重索引后更新 extraction article_id={article_id}, extraction_id={extraction_id}"
            )
    except Exception as e:
        result.debug_log.append(f"⚠️ MySQL extraction 写入失败: {type(e).__name__}: {e}")


@dataclass
class SyncResult:
    """
    sync_guardian 的运行结果摘要。

    功能：便于 Streamlit UI 和脚本统一展示；saved = 成功 upsert MySQL article_extractions 的篇数。
    """

    saved: int = 0
    skipped_url_dup: int = 0
    skipped_no_incident: int = 0
    failed: int = 0
    new_keywords: List[str] = field(default_factory=list)
    new_subdomains: List[str] = field(default_factory=list)
    debug_log: List[str] = field(default_factory=list)


def _url_already_in_mysql(web_url: str) -> bool:
    """
    按规范化 URL 查 MySQL articles 是否已存在（与 save_article 去重一致）。
    MySQL 不可用时返回 False（避免整批被跳过），错误仅能被后续写入阶段发现。
    """
    if not web_url:
        return False
    try:
        from core.mysql_db import get_article_by_url, normalize_url

        nu = normalize_url(web_url)
        if not nu:
            return False
        return get_article_by_url(nu) is not None
    except Exception:
        return False


def _build_llm_backend(api_key: Optional[str], base_url: Optional[str]) -> OpenAICompatibleBackend:
    k = (api_key or "").strip() or API_KEY
    b = (base_url or "").strip() or BASE_URL
    return OpenAICompatibleBackend(api_key=k, base_url=b)


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
    force_reindex: bool = False,
) -> SyncResult:
    """
    功能：同 sync_guardian，但 LLM 抽取步骤并发执行（asyncio.gather + Semaphore）。
    输入：concurrency 控制最大并发 LLM 请求数（默认 5）；其余参数同 sync_guardian。
    输出：SyncResult；副作用：Guardian HTTP（同步，在线程池）+ 并发 LLM + MySQL 顺序写入。
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
        if _url_already_in_mysql(art.web_url):
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

    async def _extract_one(art: RawArticle) -> Tuple[RawArticle, Optional[Dict[str, Any]], List[str]]:
        async with sem:
            context = raw_article_to_llm_context(art)
            article_dict, ext_log = await async_extract_article_from_text(
                context,
                source_url=art.web_url,
                backend=llm_backend,
            )
        return art, article_dict, ext_log

    log.append(f"🚀 并发抽取 {len(deduped)} 篇（并发上限 {concurrency}）...")
    extract_results = await asyncio.gather(*[_extract_one(a) for a in deduped], return_exceptions=False)

    # 4. 顺序处理抽取结果（RAG + MySQL/Chroma 顺序写入）
    for art, article_dict, ext_log in extract_results:
        log.extend(ext_log)

        if not article_dict or not article_dict.get("is_relevant"):
            result.skipped_no_incident += 1
            continue

        incident_like = article_dict_to_incident_like(article_dict)
        # RAG 精炼（可选）；延迟导入以支持未安装 chromadb 的轻量环境。
        try:
            from engine.rag_ingestion import apply_rag_to_incidents as _rag
            incidents_rag, rag_log = _rag(
                [incident_like],
                llm_backend=llm_backend,
                enabled=rag_enabled,
            )
        except ImportError:
            incidents_rag = [incident_like]
            rag_log = ["⚠️ chromadb 未安装，RAG 步骤跳过"]
        log.extend(rag_log)

        inc_rag = incidents_rag[0] if incidents_rag else incident_like
        merged = merge_article_with_rag(article_dict, inc_rag)
        _persist_mysql_phase1(
            art, merged, result, llm_backend=llm_backend, force_reindex=force_reindex
        )

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
    force_reindex: bool = False,
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
            force_reindex=force_reindex,
        )
    )
