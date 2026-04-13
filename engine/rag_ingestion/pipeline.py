"""
RAG 入库流水线：对抽取结果逐条做「检索 + 路由」。

功能：在启用时，用向量检索历史子域并调用 LLM 统一主域/子域表述，降低子域碎片化成本。
输入：原始 incident dict 列表；可选 top_k、模型名、llm_backend（与爬虫传入的 API Key 对齐）。
输出：(更新后的 dict 列表, debug 字符串行列表)；失败单条跳过并记录原因，不中断整批。
上下游：上游 crawler 在 LLM 抽取完成后调用；下游 UI 仍用 core.db.incident_from_extraction 校验入库。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.config import RAG_ENABLED, RAG_TOP_K
from core.db import coerce_risk_domain
from core.llm_ports import LlmBackend

from engine.rag_ingestion.retriever import retrieve_similar_subdomains
from engine.rag_ingestion.router import format_router_debug, route_incident_classification


def apply_rag_to_incidents(
    incidents_data: List[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    enabled: Optional[bool] = None,
    llm_backend: Optional[LlmBackend] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    功能：逐条精炼 risk_domain / risk_subdomain（可选关闭）。
    输入：抽取阶段输出的 dict 列表；enabled=False 时原样返回。
    输出：(新列表, debug 行)；对异常单条保留原 dict 并追加 debug。
    """
    use = RAG_ENABLED if enabled is None else enabled
    k = RAG_TOP_K if top_k is None else top_k
    debug: List[str] = []
    if not use or not incidents_data:
        return incidents_data, debug

    out: List[Dict[str, Any]] = []
    for inc in incidents_data:
        d = dict(inc)
        title = str(d.get("title", "")).strip()
        summary = str(d.get("summary", "")).strip()
        entity = str(d.get("entity", "")).strip()
        risk_level = str(d.get("risk_level", "")).strip()
        tags = d.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip() for t in tags if str(t).strip()]
        hint_domain = coerce_risk_domain(d.get("risk_domain"))
        hint_sub = str(d.get("risk_subdomain", "")).strip()

        query = f"{title}\n{summary}\n{entity}\n{', '.join(tags)}"
        try:
            # 先在同一主域内检索，减少跨域候选干扰
            cands = retrieve_similar_subdomains(
                query,
                top_k=k,
                embedding_model=embedding_model,
                restrict_domain=hint_domain if hint_domain else None,
                backend=llm_backend,
            )
            if not cands and hint_domain:
                cands = retrieve_similar_subdomains(
                    query,
                    top_k=k,
                    embedding_model=embedding_model,
                    restrict_domain=None,
                    backend=llm_backend,
                )
            routed = route_incident_classification(
                title=title,
                summary=summary,
                entity=entity,
                risk_level=risk_level,
                tags=tags,
                hint_domain=hint_domain,
                hint_subdomain=hint_sub,
                candidates=cands,
                model=llm_model,
                backend=llm_backend,
            )
            d["risk_domain"] = routed["risk_domain"]
            d["risk_subdomain"] = routed["risk_subdomain"]
            debug.append(f"RAG: {title[:40]}… | hits={format_router_debug(cands)}")
        except Exception as e:
            debug.append(f"RAG skip ({title[:30]}…): {type(e).__name__}: {e}")
        out.append(d)

    return out, debug
