"""
基于 Chroma 向量检索的风险子域 Top-K 检索。

功能：从 risk_taxonomy 读出 (主域, 子域)，在 Chroma 中按 cosine 近邻检索，供路由 LLM 做「复用旧子域 / 新建子域」决策。
输入：查询文本、top_k、可选主域过滤、可选 LlmBackend（与爬虫侧 API Key 对齐时可注入）。
输出：TaxonomyHit 列表；副作用：可能调用嵌入 API、写 Chroma 持久化目录。
上下游：上游为 pipeline；下游为 router；依赖 core.db、core.chroma_taxonomy、embedder。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.chroma_taxonomy import (
    ensure_taxonomy_pairs_embedded,
    get_taxonomy_collection,
    query_similar_taxonomy,
)
from core.db import list_taxonomy_pairs
from core.llm_ports import LlmBackend

from engine.rag_ingestion.embedder import embed_query


@dataclass(frozen=True)
class TaxonomyHit:
    """功能：承载一次检索结果（主域、子域、相似度分数）。"""

    domain: str
    subdomain: str
    score: float


def retrieve_similar_subdomains(
    query_text: str,
    *,
    top_k: int = 8,
    embedding_model: Optional[str] = None,
    restrict_domain: Optional[str] = None,
    backend: Optional[LlmBackend] = None,
    persist_directory: Optional[str] = None,
) -> List[TaxonomyHit]:
    """
    功能：按与 query_text 的嵌入相似度在 Chroma 中检索 risk_taxonomy 子域，返回 Top-K。
    输入：查询文本；restrict_domain 非空时只在该主域内检索；persist_directory 供测试注入临时目录。
    输出：TaxonomyHit 列表。
    """
    pairs = list_taxonomy_pairs()
    if restrict_domain:
        rd = restrict_domain.strip()
        pairs = [(d, s) for d, s in pairs if d == rd]
    if not pairs:
        return []

    collection = get_taxonomy_collection(persist_directory)
    ensure_taxonomy_pairs_embedded(
        collection,
        pairs,
        embedding_model=embedding_model,
        backend=backend,
    )
    qv = embed_query(query_text, model=embedding_model, backend=backend)
    rows = query_similar_taxonomy(
        collection,
        qv,
        top_k=max(1, top_k),
        restrict_domain=restrict_domain.strip() if restrict_domain else None,
    )
    return [TaxonomyHit(domain=d, subdomain=s, score=sc) for d, s, sc in rows]
