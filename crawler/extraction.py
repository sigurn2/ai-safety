"""
统一 LLM 抽取模块：将任意「正文 + 来源 URL」抽取为 AIIncident dict 列表。

功能：复用 agentic_crawl 里的提示词逻辑，但输入是已拿到的文本字符串，不依赖 Crawl4AI 或 LLMExtractionStrategy；
     提供同步版（extract_incidents_from_text）与异步版（async_extract_incidents_from_text）两个入口；
     异步版供 orchestrator 并发调用（asyncio.gather + Semaphore），显著减少批量同步等待时间。
输入：body_text（文章正文/导语拼接文本）+ source_url；可选 llm_backend/覆盖 api_key/base_url。
输出：(incidents dict 列表, debug 字符串列表)；LLM 出错时返回 ([], [错误行])。
上下游：上游为 crawler.sources 或 Crawl4AI Markdown；下游为 engine.rag_ingestion + core.db.save_incident。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.config import API_KEY, BASE_URL
from core.llm_client import OpenAICompatibleBackend

# _parse_raw_obj 提取为独立函数，同步/异步版共用。

# ---------------------------------------------------------------------------
# 提示词：与 agentic_crawl 对齐，统一更新入口，避免分叉。
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "你是一个 AI 治理与安全领域的专家分析师。"
    "仔细分析用户提供的文章内容，识别所有与 AI 治理、AI 安全、AI 政策、AI 监管相关的内容。"
)

_USER_INSTRUCTION = (
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
    "- 返回 JSON：{\"incidents\": [...]}"
)

# 正文超长时截断（字符数），防止超出模型上下文窗口。
_BODY_TRUNCATE_CHARS = 8000


def _parse_raw_obj(raw_obj: Any) -> List[Dict[str, Any]]:
    """
    功能：将 LLM 返回的 Python 对象（dict / list / 其他）统一解析为 incident dict 列表。
    输入：json.loads 之后的对象。
    输出：List[Dict]，无效项过滤掉；无 IO。
    """
    incidents: List[Dict[str, Any]] = []
    if isinstance(raw_obj, dict):
        maybe = raw_obj.get("incidents")
        if isinstance(maybe, list):
            incidents = maybe
        elif raw_obj:
            incidents = [raw_obj]
    elif isinstance(raw_obj, list):
        incidents = raw_obj
    return [i for i in incidents if isinstance(i, dict)]


def _build_backend(api_key: Optional[str], base_url: Optional[str]) -> OpenAICompatibleBackend:
    """
    功能：构造 LLM 后端；优先使用调用方传入的 key/url，回退 core.config 配置。
    输入：可选覆盖参数（空串视为未传）。
    输出：OpenAICompatibleBackend；无 IO。
    """
    k = (api_key or "").strip() or API_KEY
    b = (base_url or "").strip() or BASE_URL
    return OpenAICompatibleBackend(api_key=k, base_url=b)


def extract_incidents_from_text(
    body_text: str,
    source_url: str = "",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    backend: Optional[OpenAICompatibleBackend] = None,
    temperature: float = 0.1,
    timeout: float = 120.0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    功能：对单篇文章文本调用 LLM，返回结构化 AIIncident dict 列表。
    输入：body_text 为导语/正文拼接字符串；source_url 仅记日志（入库由调用方完成）；
         backend 优先级高于 api_key/base_url。
    输出：(incidents_list, debug_lines)；incidents_list 可能为空（无关或模型返回异常）。
    副作用：单次 HTTP Chat Completions 请求。
    上下游：上游 orchestrator / agentic_crawl；下游 apply_rag_to_incidents + save_incident。
    """
    debug: List[str] = []
    text = body_text.strip()

    if not text:
        debug.append(f"❌ 文本为空，跳过 [{source_url[:80]}]")
        return [], debug

    # 防止正文过长超出上下文
    if len(text) > _BODY_TRUNCATE_CHARS:
        text = text[:_BODY_TRUNCATE_CHARS]
        debug.append(f"⚠️ 正文截断为 {_BODY_TRUNCATE_CHARS} 字符")

    llm = backend or _build_backend(api_key, base_url)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{_USER_INSTRUCTION}\n\n---\n{text}"},
    ]

    try:
        raw_obj = llm.chat_completion_json(messages, temperature=temperature, timeout=timeout)
    except Exception as e:
        debug.append(f"❌ LLM 调用失败 [{source_url[:60]}]: {type(e).__name__}: {e}")
        return [], debug

    incidents = _parse_raw_obj(raw_obj)
    debug.append(f"✓ 抽取完成，{len(incidents)} 条 [{source_url[:60]}]")
    return incidents, debug


async def async_extract_incidents_from_text(
    body_text: str,
    source_url: str = "",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    backend: Optional[OpenAICompatibleBackend] = None,
    temperature: float = 0.1,
    timeout: float = 120.0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    功能：异步版单篇抽取；与 extract_incidents_from_text 逻辑相同，使用 httpx.AsyncClient。
    输入：同同步版；backend 共享实例安全（AsyncClient 在函数内独立创建）。
    输出：同同步版；副作用：一次异步 HTTP Chat Completions 请求。
    上下游：crawler.orchestrator.async_sync_guardian 通过 asyncio.gather 并发调用。
    """
    debug: List[str] = []
    text = body_text.strip()

    if not text:
        debug.append(f"❌ 文本为空，跳过 [{source_url[:80]}]")
        return [], debug

    if len(text) > _BODY_TRUNCATE_CHARS:
        text = text[:_BODY_TRUNCATE_CHARS]
        debug.append(f"⚠️ 正文截断为 {_BODY_TRUNCATE_CHARS} 字符")

    llm = backend or _build_backend(api_key, base_url)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{_USER_INSTRUCTION}\n\n---\n{text}"},
    ]

    try:
        raw_obj = await llm.async_chat_completion_json(
            messages, temperature=temperature, timeout=timeout
        )
    except Exception as e:
        debug.append(f"❌ LLM 异步调用失败 [{source_url[:60]}]: {type(e).__name__}: {e}")
        return [], debug

    incidents = _parse_raw_obj(raw_obj)
    debug.append(f"✓ 异步抽取完成，{len(incidents)} 条 [{source_url[:60]}]")
    return incidents, debug
