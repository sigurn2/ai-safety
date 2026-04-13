"""
LLM 路由：在 RAG 候选基础上输出规范 risk_domain / risk_subdomain。

功能：把抽取模型的「参考主域/子域」与检索到的历史子域候选一并交给模型，优先复用已有子域字符串以抑制语义重复。
输入：事件字段、TaxonomyHit 候选列表、可选 LlmBackend（与爬虫侧 Key 一致时注入）。
输出：dict 含 risk_domain、risk_subdomain；副作用：一次 chat HTTP。
上下游：上游 pipeline；下游为入库前的 dict 覆盖；依赖 models.schema 主域枚举。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.llm_client import default_llm_backend
from core.llm_ports import LlmBackend
from models.schema import RISK_DOMAIN_CHOICES

from engine.rag_ingestion.retriever import TaxonomyHit


def route_incident_classification(
    *,
    title: str,
    summary: str,
    entity: str,
    risk_level: str,
    tags: List[str],
    hint_domain: str,
    hint_subdomain: str,
    candidates: List[TaxonomyHit],
    model: Optional[str] = None,
    backend: Optional[LlmBackend] = None,
) -> Dict[str, str]:
    """
    功能：生成最终三元主域 + 动态子域（JSON 契约）。
    输入：情报文本字段、RAG 候选、可选 chat model 名与 backend。
    输出：{"risk_domain", "risk_subdomain"}；若模型主域漂移则回退 hint 或默认第三主域。
    """
    cand_lines: List[str] = []
    for i, h in enumerate(candidates, 1):
        cand_lines.append(f"{i}. 主域: {h.domain} | 子域: {h.subdomain} | 相似度: {h.score:.4f}")

    cand_block = "\n".join(cand_lines) if cand_lines else "（知识库中尚无历史子域，请自行归纳一个简短、专业的子域名。）"

    allowed_domains = "\n".join(f"- {d}" for d in RISK_DOMAIN_CHOICES)

    user = (
        "请根据下列情报文本与「候选历史子域」完成分类。\n\n"
        f"【标题】{title}\n"
        f"【主体】{entity}\n"
        f"【风险等级】{risk_level}\n"
        f"【摘要】{summary}\n"
        f"【标签】{', '.join(tags)}\n\n"
        f"【爬虫/抽取模型给出的参考】主域: {hint_domain} | 子域: {hint_subdomain}\n\n"
        "【候选历史子域（RAG 检索，按相似度排序）】\n"
        f"{cand_block}\n\n"
        "【要求】\n"
        f"1. risk_domain 必须是以下三项之一（原样复制整行）：\n{allowed_domains}\n"
        "2. risk_subdomain：优先从候选中选择最匹配的「子域」字符串并原样复用；若无合适项，给出一个新的简短专业子类型（建议 2~12 个字，同类现象用语尽量与候选风格一致）。\n"
        "3. 只输出一个 JSON 对象，不要 Markdown，不要解释。格式："
        '{"risk_domain":"...","risk_subdomain":"..."}'
    )

    messages = [
        {"role": "system", "content": "你是 AI 治理与安全领域的分类专家，输出严格 JSON。"},
        {"role": "user", "content": user},
    ]
    be = backend or default_llm_backend()
    data: Any = be.chat_completion_json(messages, model=model, temperature=0.1)
    if not isinstance(data, dict):
        raise ValueError(f"router expected JSON object, got {type(data)}")
    rd = str(data.get("risk_domain", "")).strip()
    rs = str(data.get("risk_subdomain", "")).strip()
    if rd not in RISK_DOMAIN_CHOICES:
        # 模型偶发格式漂移：回退到抽取参考主域
        rd = hint_domain if hint_domain in RISK_DOMAIN_CHOICES else RISK_DOMAIN_CHOICES[2]
    if not rs:
        rs = hint_subdomain or "未指定子域"
    rs = rs[:160]
    return {"risk_domain": rd, "risk_subdomain": rs}


def format_router_debug(candidates: List[TaxonomyHit]) -> str:
    """
    功能：将候选列表序列化为紧凑 JSON 字符串，供爬虫 debug 日志展示。
    输入：TaxonomyHit 列表。
    输出：UTF-8 JSON 文本。
    """
    return json.dumps(
        [{"domain": h.domain, "subdomain": h.subdomain, "score": round(h.score, 4)} for h in candidates],
        ensure_ascii=False,
    )
