"""
混合检索：Chroma 向量 + MySQL FULLTEXT，RRF 融合与按 article_id 截断。

依赖 article_chunks 上 FULLTEXT 索引（scripts/migrate_add_chunks_fulltext.sql）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from core.config import EMBEDDING_MODEL
from core.llm_ports import LlmBackend
from core.mysql_db import build_report_source_rows, list_article_ids_by_filters, search_chunks_fulltext
from engine.article_index.retriever import ArticleChunkHit, query_article_chunks

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class EvidenceHit:
    vector_id: str
    article_id: int
    chunk_text: str
    rrf_score: float
    metadata: Dict[str, Any]


def reciprocal_rank_fuse(ranked_id_lists: Sequence[Sequence[str]], *, k: int = DEFAULT_RRF_K) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ids in ranked_id_lists:
        for rank, vid in enumerate(ids):
            vs = str(vid or "").strip()
            if not vs:
                continue
            scores[vs] = scores.get(vs, 0.0) + 1.0 / (float(k) + float(rank) + 1.0)
    return scores


def _needs_article_filter(
    risk_domain: Optional[str],
    source: Optional[str],
    published_after: Optional[datetime],
    published_before: Optional[datetime],
) -> bool:
    if published_after is not None or published_before is not None:
        return True
    if risk_domain is not None and str(risk_domain).strip():
        return True
    if source is not None and str(source).strip():
        return True
    return False


def hybrid_retrieve(
    question: str,
    *,
    top_k: int = 12,
    rrf_k: int = DEFAULT_RRF_K,
    max_chunks_per_article: int = 2,
    embedding_model: Optional[str] = None,
    backend: Optional[LlmBackend] = None,
    persist_directory: Optional[str] = None,
    risk_domain: Optional[str] = None,
    source: Optional[str] = None,
    published_after: Optional[datetime] = None,
    published_before: Optional[datetime] = None,
    article_id_allowlist: Optional[List[int]] = None,
    vector_top_n: int = 24,
    sparse_top_n: int = 24,
) -> List[EvidenceHit]:
    aids: Optional[List[int]] = article_id_allowlist
    if aids is None and _needs_article_filter(risk_domain, source, published_after, published_before):
        aids = list_article_ids_by_filters(
            risk_domain=risk_domain,
            source=source,
            published_after=published_after,
            published_before=published_before,
        )
        if not aids:
            return []

    vec_hits = query_article_chunks(
        question,
        top_k=vector_top_n,
        embedding_model=embedding_model or EMBEDDING_MODEL,
        backend=backend,
        persist_directory=persist_directory,
        article_id_allowlist=aids,
    )
    vec_ids = [h.vector_id for h in vec_hits]

    sparse_rows = search_chunks_fulltext(question, limit=sparse_top_n, article_ids=aids)
    sparse_ids = [str(r["vector_id"]) for r in sparse_rows if r.get("vector_id")]

    fused = reciprocal_rank_fuse([vec_ids, sparse_ids], k=rrf_k)
    ordered_vids = sorted(fused.keys(), key=lambda v: fused[v], reverse=True)

    vec_map: Dict[str, ArticleChunkHit] = {h.vector_id: h for h in vec_hits}
    sparse_map: Dict[str, Dict[str, Any]] = {
        str(r["vector_id"]): r for r in sparse_rows if r.get("vector_id")
    }

    hits: List[EvidenceHit] = []
    per_art: Dict[int, int] = {}
    for vid in ordered_vids:
        h = vec_map.get(vid)
        row = sparse_map.get(vid)
        if h:
            meta = dict(h.metadata)
            try:
                aid = int(meta.get("article_id") or 0)
            except (TypeError, ValueError):
                aid = 0
            text = h.document
        elif row:
            try:
                aid = int(row.get("article_id") or 0)
            except (TypeError, ValueError):
                aid = 0
            text = str(row.get("chunk_text") or "")
            meta = {"article_id": aid, "vector_id": vid}
        else:
            continue
        if aid <= 0:
            continue
        n = per_art.get(aid, 0)
        if n >= max_chunks_per_article:
            continue
        per_art[aid] = n + 1
        hits.append(
            EvidenceHit(
                vector_id=vid,
                article_id=aid,
                chunk_text=text,
                rrf_score=float(fused[vid]),
                metadata=meta,
            )
        )
        if len(hits) >= top_k:
            break

    return hits


def evidence_hits_to_report_sources(hits: List[EvidenceHit]) -> List[Dict[str, Any]]:
    if not hits:
        return []
    mx = max(h.rrf_score for h in hits) or 1.0
    rows: List[Dict[str, Any]] = []
    for i, h in enumerate(hits, 1):
        rows.append(
            {
                "vector_id": h.vector_id,
                "article_id": h.article_id,
                "rrf_score": max(0.0, min(1.0, h.rrf_score / mx)),
                "citation_label": f"来源 {i}",
            }
        )
    return build_report_source_rows(rows)
