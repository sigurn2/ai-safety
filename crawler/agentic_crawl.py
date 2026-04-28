"""
基于 Crawl4AI 的异步抓取与 LLM **文章级** 结构化抽取，并在开启 RAG 时对子域做检索增强路由。

功能：访问目标 URL，输出每篇文章一个 JSON 对象（无关则仅 is_relevant + reject_reason）；合并关键词池。
输出：(incident_like 列表，长度 0 或 1，供 SQLite 展示行入库)、新关键词、debug。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, List, Optional, Tuple

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

try:
    from crawl4ai.async_configs import LLMConfig  # type: ignore
except Exception:  # pragma: no cover
    from crawl4ai.config import LLMConfig  # type: ignore

from core.config import API_KEY, BASE_URL, CRAWL_PAGE_TIMEOUT_MS, CRAWL_WAIT_UNTIL, LLM_MODEL
from core.db import update_watched_keywords
from core.llm_client import OpenAICompatibleBackend
from crawler.extraction import RISK_DOMAIN_LLM_GUIDANCE, _parse_article_obj, article_dict_to_incident_like
from engine.rag_ingestion import apply_rag_to_incidents
from models.schema import ArticleExtractionPayload


def _normalize_optional(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = s.strip()
    return t or None


def _rag_backend(api_key: Optional[str], base_url: Optional[str]) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        api_key=_normalize_optional(api_key),
        base_url=_normalize_optional(base_url),
    )


async def run_agentic_crawl(
    url: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    debug: bool = False,
) -> Tuple[List[Any], List[str], List[str]]:
    _ = debug
    debug_log: List[str] = []

    run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    _api_key = api_key or API_KEY
    _base_url = base_url or BASE_URL
    _model = f"openai/{LLM_MODEL}"
    crawl_page_timeout_ms = CRAWL_PAGE_TIMEOUT_MS
    crawl_wait_until = CRAWL_WAIT_UNTIL
    crawl_session_id = f"ai_monitor_{run_id}"

    if not _api_key:
        debug_log.append("❌ API Key 未配置，请在 .env 文件中设置 DASHSCOPE_API_KEY")
        return [], [], debug_log

    try:
        debug_log.append(f"✓ API 配置已验证（模型: {_model}）")

        llm_config = LLMConfig(
            provider=_model,
            api_token=_api_key,
            base_url=_base_url,
        )
        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            schema=ArticleExtractionPayload.model_json_schema(),
            instruction=(
                "你是 AI 治理与安全分析师。对页面输出**一个** JSON 对象，字段须符合 schema，"
                "且与 MySQL article_extractions 一致：相关时 is_relevant=true，并填 "
                "content_type(news/meeting/report/policy/opinion/other)、main_topic(含法案/会议等线索)、"
                "risk_subdomains、entities、summary_structured、tags；\n"
                + RISK_DOMAIN_LLM_GUIDANCE
                + "可选 relevance_reason（仅调试用）；"
                "会议稿用会议级信息，勿拆成多条发言人；"
                "不要输出 event_hint；"
                "不相关时仅 {\"is_relevant\":false,\"reject_reason\":\"no_ai_governance_content\"}。"
            ),
        )

        config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            cache_mode=CacheMode.BYPASS,
            wait_until=crawl_wait_until,
            page_timeout=crawl_page_timeout_ms,
            session_id=crawl_session_id,
            js_code="window.scrollTo(0, document.body.scrollHeight);",
        )

        async with AsyncWebCrawler() as crawler:
            debug_log.append("📡 正在爬取目标 URL...")
            result: Any = await crawler.arun(url=url, config=config)

            debug_log.append(f"✓ 爬虫返回: success={result.success}")

            if not result.success:
                debug_log.append(
                    f"❌ 爬虫失败: {result.error_message if hasattr(result, 'error_message') else '未知错误'}"
                )
                return [], [], debug_log

            if not result.extracted_content:
                debug_log.append("❌ 爬虫获取内容为空，页面可能被屏蔽或不存在")
                return [], [], debug_log

            debug_log.append(f"✓ 获取到内容（长度: {len(str(result.extracted_content))} 字符）")

            try:
                raw = result.extracted_content
                data = json.loads(raw) if isinstance(raw, str) else raw
                debug_log.append(f"✓ LLM 提取完成，返回类型: {type(data)}")
            except Exception as e:
                debug_log.append(f"❌ JSON 解析失败: {str(e)}")
                debug_log.append(f"   原始内容: {str(raw)[:200]}")
                return [], [], debug_log

            art = _parse_article_obj(data)
            if art is None:
                debug_log.append("❌ 无法解析为文章级抽取")
                return [], [], debug_log

            debug_log.append(f"✓ is_relevant={art.get('is_relevant')}")
            if not art.get("is_relevant"):
                reason = art.get("reject_reason", "")
                debug_log.append(f"💡 未入库: {reason or 'not relevant'}")
                return [], [], debug_log

            incident_like = article_dict_to_incident_like(art)
            rb = _rag_backend(api_key, base_url)
            incidents_data, rag_lines = apply_rag_to_incidents([incident_like], llm_backend=rb)
            for line in rag_lines:
                debug_log.append(line)

            all_tags: List[str] = []
            for inc in incidents_data:
                all_tags.extend(inc.get("tags", []))
            newly_added = update_watched_keywords(all_tags)
            debug_log.append(f"📊 新增关键词: {len(newly_added)}")

            return incidents_data, newly_added, debug_log

    except Exception as e:
        debug_log.append(f"❌ 执行异常: {type(e).__name__}: {str(e)}")
        import traceback

        debug_log.append(f"   堆栈: {traceback.format_exc()[:300]}")
        return [], [], debug_log
