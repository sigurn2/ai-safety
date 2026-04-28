"""
统一 LLM 抽取模块：将「正文 + 来源 URL」抽成单篇文章一条结构化 JSON（ArticleExtractionPayload）。

功能：与 agentic_crawl 提示词语义对齐；提供同步/异步入口；异步版供 orchestrator 并发调用。
输入：body_text、source_url；可选 llm_backend。
输出：(article dict | None, debug)；无关或失败时 article 为 None。
下游：article_dict_to_incident_like → apply_rag_to_incidents（单元素列表）→ incident_from_extraction / save_extraction。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.config import API_KEY, BASE_URL
from core.llm_client import OpenAICompatibleBackend

_ALLOWED_CONTENT_TYPES = frozenset({"news", "meeting", "report", "policy", "opinion", "other"})
_CONTENT_TYPE_ALIASES = {
    "policy_paper": "policy",
    "op_ed": "opinion",
    "research": "report",
}

# ---------------------------------------------------------------------------
# 提示词：与 agentic_crawl 对齐，统一更新入口。
# ---------------------------------------------------------------------------
# 与 models.schema.RISK_DOMAIN_CHOICES 三条字符串完全一致；供 extraction 与 agentic_crawl 共用。
RISK_DOMAIN_LLM_GUIDANCE = (
    "risk_domain（意图与来源三元模型）：须从下列三项中**原样**选一整行字符串（含英文与中文括号）：\n"
    "  - Malicious Use (恶意滥用)：人类恶意利用 AI，或对 AI 系统发起主动攻击（越狱、投毒、深度伪造诈骗、自动化网络攻击等）。\n"
    "  - Accidental Failure (意外失效)：无恶意攻击者，因系统缺陷、幻觉或复杂环境下的失效（严重幻觉、自动驾驶误判、域外泛化失败等）。\n"
    "  - Systemic & Ethical Risk (系统性与伦理风险)：系统按预期运行，但对社会或个人权益产生负面影响（算法偏见、隐私、版权、信息茧房、就业冲击等）。\n"
)

_SYSTEM_PROMPT = (
    "你是一个 AI 治理与安全领域的专家分析师。"
    "对**整篇文章**只输出**一个** JSON 对象，描述「这篇材料在 AI 治理视角下是什么」，"
    "不要拆成多条 incident，不要输出 incidents 数组。"
)

_USER_INSTRUCTION = (
    "【相关时】输出一个 JSON 对象，字段如下（与库表 article_extractions 一一对应，外加流程字段）：\n"
    "- is_relevant: true（流程用，不入 extraction 行）\n"
    "- content_type: 必选其一 news | meeting | report | policy | opinion | other\n"
    "- main_topic: 一句话概括，≤512 字；新闻写报道核心，会议写主题/讨论焦点；"
    "法案/标准/常设会议/政策进程等线索也写进 main_topic（不要单独键）\n"
    "- " + RISK_DOMAIN_LLM_GUIDANCE
    + "- risk_subdomains: 字符串数组，治理或风险议题短标签（落库 JSON 数组）\n"
    "- entities: 字符串数组，主要机构/公司/政府/人物（落库 JSON 数组）\n"
    "- summary_structured: 一句话摘要，≤512 字\n"
    "- tags: 字符串数组，3–8 个检索关键词（落库 JSON 数组）\n"
    "- relevance_reason: 可选，简短说明为何相关（仅调试，不入 extraction 表）\n\n"
    "【新闻】侧重：讲什么、主体、治理/风险议题、tags。\n"
    "【会议】会议级一条：会议名/主办方/时间地点（若有）/讨论主题/主体/与 AI 治理或安全的关系；勿拆每位发言人。\n\n"
    "【不相关时】**仅**输出两键，不要填其它字段：\n"
    '{ "is_relevant": false, "reject_reason": "no_ai_governance_content" }\n'
    "（reject_reason 可用简短英文代码或短语。）\n\n"
    "不要捏造事实；不要 {\"incidents\":[...]}；不要输出 event_hint 等已废弃字段。"
)

_BODY_TRUNCATE_CHARS = 8000


def _normalize_content_type(raw: Any) -> str:
    t = str(raw or "other").strip().lower()
    t = _CONTENT_TYPE_ALIASES.get(t, t)
    if t in _ALLOWED_CONTENT_TYPES:
        return t
    return "other"


def _as_str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        t = val.strip()
        return [t] if t else []
    return []


def _parse_article_obj(raw_obj: Any) -> Optional[Dict[str, Any]]:
    """
    将 LLM 返回对象规范为 article 级 dict；无法识别时返回 None。
    兼容旧版 {\"incidents\":[...]}：取首条若含 title/summary 则尽量回填为单篇结构（不推荐）。
    """
    d: Optional[Dict[str, Any]] = None
    if isinstance(raw_obj, dict):
        if "is_relevant" in raw_obj:
            d = raw_obj
        elif isinstance(raw_obj.get("incidents"), list) and raw_obj["incidents"]:
            first = raw_obj["incidents"][0]
            if isinstance(first, dict) and (first.get("title") or first.get("summary")):
                d = {
                    "is_relevant": True,
                    "content_type": "news",
                    "main_topic": str(first.get("title") or "")[:512],
                    "risk_domain": first.get("risk_domain", ""),
                    "risk_subdomains": _as_str_list(first.get("risk_subdomain")),
                    "entities": [str(first.get("entity") or "").strip()] if first.get("entity") else [],
                    "summary_structured": str(first.get("summary") or "")[:512],
                    "tags": _as_str_list(first.get("tags")),
                    "relevance_reason": "legacy incidents[0]",
                    "reject_reason": "",
                }
    elif isinstance(raw_obj, list) and raw_obj and isinstance(raw_obj[0], dict):
        first = raw_obj[0]
        if "is_relevant" in first:
            d = first
        elif first.get("title") or first.get("summary"):
            return _parse_article_obj({"incidents": raw_obj})
        else:
            d = None

    if not d or not isinstance(d, dict):
        return None

    rel = bool(d.get("is_relevant", False))
    if not rel:
        rr = str(d.get("reject_reason") or "").strip() or "no_ai_governance_content"
        return {"is_relevant": False, "reject_reason": rr[:255]}

    subs = _as_str_list(d.get("risk_subdomains"))
    if not subs and d.get("risk_subdomain"):
        subs = _as_str_list(d.get("risk_subdomain"))
    ents = _as_str_list(d.get("entities"))
    if not ents and d.get("entity"):
        ents = _as_str_list(str(d.get("entity")).replace("，", ",").split(","))

    summary = str(d.get("summary_structured") or d.get("summary") or "")[:512]
    tags = _as_str_list(d.get("tags"))

    out: Dict[str, Any] = {
        "is_relevant": True,
        "content_type": _normalize_content_type(d.get("content_type")),
        "main_topic": str(d.get("main_topic") or d.get("title") or "")[:512],
        "risk_domain": str(d.get("risk_domain") or "").strip(),
        "risk_subdomains": subs[:20],
        "entities": ents[:50],
        "summary_structured": summary,
        "tags": tags[:24],
        "relevance_reason": str(d.get("relevance_reason") or "")[:512],
        "reject_reason": "",
    }
    return out


def article_dict_to_incident_like(art: Dict[str, Any]) -> Dict[str, Any]:
    """将文章级抽取转为 RAG / SQLite / 事件匹配使用的单条「情报」dict。"""
    subs = art.get("risk_subdomains") or []
    if not isinstance(subs, list):
        subs = []
    sub_first = str(subs[0]).strip() if subs else "未指定子域"
    ents = art.get("entities") or []
    if not isinstance(ents, list):
        ents = []
    entity_str = ", ".join(str(e).strip() for e in ents if str(e).strip())
    tags = art.get("tags") if isinstance(art.get("tags"), list) else []
    tags = [str(t).strip() for t in tags if str(t).strip()]
    mt = str(art.get("main_topic") or "").strip()
    return {
        "title": mt,
        "entity": entity_str,
        "risk_level": "中",
        "risk_domain": art.get("risk_domain"),
        "risk_subdomain": sub_first,
        "summary": str(art.get("summary_structured") or "").strip(),
        "tags": tags,
        "action_type": "其他",
        "place": "",
        "stance": "未知",
        "topic_raw": sub_first if sub_first != "未指定子域" else mt[:160],
    }


def merge_article_with_rag(art: Dict[str, Any], inc_rag: Dict[str, Any]) -> Dict[str, Any]:
    """把 RAG 精炼后的主域/首子域写回文章级 dict，供 MySQL save_extraction。"""
    m = dict(art)
    m["risk_domain"] = str(inc_rag.get("risk_domain") or m.get("risk_domain") or "").strip()
    refined = str(inc_rag.get("risk_subdomain") or "").strip()
    subs = list(m.get("risk_subdomains") or [])
    if not isinstance(subs, list):
        subs = []
    subs = [str(s).strip() for s in subs if str(s).strip()]
    if refined and refined != "未指定子域":
        subs = [refined] + [s for s in subs if s != refined]
    elif refined == "未指定子域" and not subs:
        subs = []
    m["risk_subdomains"] = subs[:20]
    return m


def _build_backend(api_key: Optional[str], base_url: Optional[str]) -> OpenAICompatibleBackend:
    k = (api_key or "").strip() or API_KEY
    b = (base_url or "").strip() or BASE_URL
    return OpenAICompatibleBackend(api_key=k, base_url=b)


def extract_article_from_text(
    body_text: str,
    source_url: str = "",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    backend: Optional[OpenAICompatibleBackend] = None,
    temperature: float = 0.1,
    timeout: float = 120.0,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    debug: List[str] = []
    text = body_text.strip()
    if not text:
        debug.append(f"❌ 文本为空，跳过 [{source_url[:80]}]")
        return None, debug
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
        return None, debug

    art = _parse_article_obj(raw_obj)
    if art is None:
        debug.append(f"❌ 无法解析为文章级抽取 [{source_url[:60]}]")
        return None, debug
    debug.append(f"✓ 抽取完成，is_relevant:{art['is_relevant']} [{source_url[:60]}]")
    return art, debug


async def async_extract_article_from_text(
    body_text: str,
    source_url: str = "",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    backend: Optional[OpenAICompatibleBackend] = None,
    temperature: float = 0.1,
    timeout: float = 120.0,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    debug: List[str] = []
    text = body_text.strip()
    if not text:
        debug.append(f"❌ 文本为空，跳过 [{source_url[:80]}]")
        return None, debug
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
        return None, debug

    art = _parse_article_obj(raw_obj)
    if art is None:
        debug.append(f"❌ 无法解析为文章级抽取 [{source_url[:60]}]")
        return None, debug
    debug.append(f"✓ 异步抽取完成，is_relevant:{art['is_relevant']} [{source_url[:60]}]")
    return art, debug


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
    """兼容旧签名：内部走文章级抽取，仅当 is_relevant 时返回单元素列表（incident_like）。"""
    art, dbg = extract_article_from_text(
        body_text,
        source_url,
        api_key=api_key,
        base_url=base_url,
        backend=backend,
        temperature=temperature,
        timeout=timeout,
    )
    if not art or not art.get("is_relevant"):
        return [], dbg
    return [article_dict_to_incident_like(art)], dbg


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
    art, dbg = await async_extract_article_from_text(
        body_text,
        source_url,
        api_key=api_key,
        base_url=base_url,
        backend=backend,
        temperature=temperature,
        timeout=timeout,
    )
    if not art or not art.get("is_relevant"):
        return [], dbg
    return [article_dict_to_incident_like(art)], dbg

