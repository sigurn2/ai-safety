"""
基于 Crawl4AI 的异步抓取与 LLM 结构化抽取，并在开启 RAG 时对子域做检索增强路由。

功能：访问目标 URL，用语义策略抽取 AI 治理相关 incidents；合并关键词池；可选在入库前用向量检索历史 taxonomy 精炼子域。
输入：url、可选 api_key/base_url（与 Streamlit 侧一致时覆盖 .env）、debug 预留。
输出：(incidents 字典列表, 本次新入库关键词, debug 字符串列表)。
上下游：下游为 Streamlit 或脚本（core.db 入库）；依赖 crawl4ai、core、models、engine.rag_ingestion。
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
from engine.rag_ingestion import apply_rag_to_incidents
from models.schema import ExtractionResult


def _normalize_optional(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = s.strip()
    return t or None


def _rag_backend(api_key: Optional[str], base_url: Optional[str]) -> OpenAICompatibleBackend:
    """
    功能：为单次爬取构造与 crawl4ai LLM 配置一致的 OpenAI 兼容后端，供 RAG 嵌入/路由复用同一密钥与 base_url。
    输入：可选密钥与 base；空串视为未传。
    输出：OpenAICompatibleBackend 实例；未传字段回退 core.config 环境解析值。
    """
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
    """
    功能：执行一次「抓取 → 抽取 → RAG 精炼（可关）→ 关键词池更新」。
    输入：目标 URL；api_key/base_url 覆盖 .env；debug 当前未改行为，预留扩展。
    输出：(incidents 原始/精炼 dict 列表, 新关键词列表, 人类可读 debug 行)。
    副作用：网络请求、可选 embedding/chat、写 watched_keywords（不写 incidents，由 UI 入库）。
    """
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
            schema=ExtractionResult.model_json_schema(),
            instruction=(
                "你是一个 AI 治理与安全领域的专家分析师。仔细分析网页内容，识别所有与 AI 治理、AI 安全、AI 政策、AI 监管相关的内容。\n"
                "对每条相关内容，精确提取以下字段（JSON 格式）：\n"
                "1. title（标题）：事件、报告、新闻或会议的标题\n"
                "2. entity（涉及主体）：提及的机构、公司、政府或人物名称\n"
                "3. risk_level（风险等级）：根据内容判断，填写「高」「中」「低」之一\n"
                "4. risk_domain（风险主域 — 意图与来源三元模型）：必须从以下三项中**原样**选一条字符串（含英文与中文括号）：\n"
                "   - Malicious Use (恶意滥用)：人类恶意利用 AI，或对 AI 系统发起主动攻击（越狱、投毒、深度伪造诈骗、自动化网络攻击等）。\n"
                "   - Accidental Failure (意外失效)：无恶意攻击者，因系统缺陷、幻觉或复杂环境下的失效（严重幻觉、自动驾驶误判、域外泛化失败等）。\n"
                "   - Systemic & Ethical Risk (系统性与伦理风险)：系统按预期运行，但对社会或个人权益产生负面影响（算法偏见、隐私、版权、信息茧房、就业冲击等）。\n"
                "5. risk_subdomain（风险子域）：在该主域下的简短专业子类型（如「越狱攻击」「数据投毒」「算法偏见」）。同类现象用语尽量一致，新现象可创造清晰新词。\n"
                "6. summary（摘要）：一句话不超过 60 字\n"
                "7. tags（标签）：3-8 个关键词，中英文均可\n\n"
                "重要提示：\n"
                "- 只提取与 AI 治理/安全相关的内容；risk_domain 三条字符串必须与上文完全一致\n"
                "- 不要捏造事实\n"
                "- 若无相关内容，incidents 为空数组\n"
                "- 返回 JSON：{\"incidents\": [...]} "
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

            incidents_data: List[Any] = []
            if isinstance(data, list):
                incidents_data = data
                debug_log.append("💡 LLM 直接返回了 List 格式")
            elif isinstance(data, dict):
                incidents_data = data.get("incidents", [])
                if not incidents_data and data:
                    incidents_data = [data]
                debug_log.append("💡 LLM 返回了 Dict 格式")
            else:
                debug_log.append(f"❌ LLM 返回格式错误：期望 dict 或 list，实际 {type(data)}")
                return [], [], debug_log

            debug_log.append(f"✓ 最终提取到 {len(incidents_data)} 条情报")

            if not incidents_data:
                debug_log.append("💡 可能原因：1) 页面内容无相关信息 2) Schema 验证失败 3) LLM 输出格式异常")
                return [], [], debug_log

            rb = _rag_backend(api_key, base_url)
            incidents_data, rag_lines = apply_rag_to_incidents(incidents_data, llm_backend=rb)
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
