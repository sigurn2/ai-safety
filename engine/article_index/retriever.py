"""
Chroma article_content 集合向量检索。

与 core.chroma_taxonomy（子域 taxonomy）分离；用于研究问题 → 相关 chunk。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from core.config import ARTICLE_CHROMA_DIR, EMBEDDING_MODEL
from core.llm_ports import LlmBackend
from engine.article_index.indexer import ARTICLE_COLLECTION
from engine.rag_ingestion.embedder import embed_query


def get_article_collection(
    persist_directory: Optional[str] = None,
) -> Collection:
    path = persist_directory if persist_directory is not None else ARTICLE_CHROMA_DIR
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=ARTICLE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _distance_to_score(distance: float) -> float:
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - d))


@dataclass(frozen=True)
class ArticleChunkHit:
    vector_id: str
    document: str
    metadata: Dict[str, Any]
    score: float


def query_article_chunks(
    question: str,
    *,
    top_k: int = 12,
    embedding_model: Optional[str] = None,
    backend: Optional[LlmBackend] = None,
    persist_directory: Optional[str] = None,
    where_filter: Optional[Dict[str, Any]] = None,
    article_id_allowlist: Optional[Sequence[int]] = None,
) -> List[ArticleChunkHit]:
    """
    对 article_content 做语义检索。若提供 article_id_allowlist，多取结果后在 Python 中过滤。
    """
    q = (question or "").strip()
    if not q:
        return []

    emodel = (embedding_model or EMBEDDING_MODEL or "").strip()
    qv = embed_query(q, model=emodel or None, backend=backend)
    collection = get_article_collection(persist_directory)

    n_fetch = max(1, top_k * 4) if article_id_allowlist else max(1, top_k)
    kwargs: Dict[str, Any] = {
        "query_embeddings": [qv],
        "n_results": n_fetch,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        kwargs["where"] = where_filter

    res = collection.query(**kwargs)
    ids_out = (res.get("ids") or [[]])[0]
    if not ids_out:
        return []

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    allow = None
    if article_id_allowlist is not None:
        allow = {int(x) for x in article_id_allowlist}

    hits: List[ArticleChunkHit] = []
    for vid, doc, meta, dist in zip(ids_out, docs, metas, dists):
        if not vid:
            continue
        meta = dict(meta or {})
        if allow is not None:
            aid = meta.get("article_id")
            try:
                if int(aid) not in allow:
                    continue
            except (TypeError, ValueError):
                continue
        hits.append(
            ArticleChunkHit(
                vector_id=str(vid),
                document=str(doc or ""),
                metadata=meta,
                score=_distance_to_score(dist),
            )
        )
        if len(hits) >= top_k:
            break

    return hits
