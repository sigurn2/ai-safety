"""
风险子域 (domain, subdomain) 向量在 Chroma 中的持久化与检索。

功能：替代 SQLite taxonomy_embeddings；用 PersistentClient 本地目录存储嵌入，支持按主域 metadata 过滤查询。
输入：来自 risk_taxonomy 的 (主域, 子域) 对；向量由 embedder + LlmBackend 生成。
输出：query 返回 (domain, subdomain, score)；score 由 Chroma cosine 距离映射为越大越相似。
上下游：仅被 engine.rag_ingestion.retriever 调用；主数据仍在 SQLite risk_taxonomy。
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, List, Optional, Tuple

import chromadb

from core.config import CHROMA_PERSIST_DIR
from core.llm_ports import LlmBackend

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

COLLECTION_NAME = "risk_taxonomy"


def taxonomy_pair_id(domain: str, subdomain: str) -> str:
    """稳定 ID：避免主域/子域字符串中的特殊字符影响 Chroma id。"""
    raw = f"{domain.strip()}\0{subdomain.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _taxonomy_document(domain: str, subdomain: str) -> str:
    return f"{domain.strip()} | {subdomain.strip()}"


def get_chroma_client(persist_directory: Optional[str] = None) -> chromadb.PersistentClient:
    path = persist_directory if persist_directory is not None else CHROMA_PERSIST_DIR
    return chromadb.PersistentClient(path=path)


def get_taxonomy_collection(
    persist_directory: Optional[str] = None,
) -> "Collection":
    """
    功能：获取或创建 cosine 空间的子域集合。
    输入：可选持久化目录（测试用临时目录）。
    输出：Chroma Collection。
    """
    client = get_chroma_client(persist_directory)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ensure_taxonomy_pairs_embedded(
    collection: "Collection",
    pairs: List[Tuple[str, str]],
    *,
    embedding_model: Optional[str] = None,
    backend: Optional[LlmBackend] = None,
) -> None:
    """
    功能：对 Chroma 中尚不存在的 id 批量嵌入并 upsert。
    输入：当前要参与检索的 (domain, subdomain) 列表；backend 为可选 LlmBackend。
    输出：无；副作用：可能调用嵌入 API、写 Chroma。
    """
    if not pairs:
        return
    from engine.rag_ingestion.embedder import embed_documents

    ids = [taxonomy_pair_id(d, s) for d, s in pairs]
    got = collection.get(ids=ids, include=[])
    have = set(got["ids"] or [])
    missing = [(d, s) for i, (d, s) in enumerate(pairs) if ids[i] not in have]
    if not missing:
        return
    documents = [_taxonomy_document(d, s) for d, s in missing]
    embeddings = embed_documents(documents, model=embedding_model, backend=backend)
    new_ids = [taxonomy_pair_id(d, s) for d, s in missing]
    metadatas = [{"domain": d.strip(), "subdomain": s.strip()} for d, s in missing]
    collection.upsert(ids=new_ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def _distance_to_score(distance: float) -> float:
    """Chroma cosine 距离通常与 (1 - cos_sim) 同阶；映射为越大越相似便于日志与 router 展示。"""
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - d))


def query_similar_taxonomy(
    collection: "Collection",
    query_embedding: List[float],
    *,
    top_k: int,
    restrict_domain: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """
    功能：按 query 向量检索 Top-K，可选 metadata 过滤主域。
    输入：已归一化由同一嵌入模型产生的查询向量。
    输出：(domain, subdomain, score) 列表，按相似度降序。
    """
    n = max(1, top_k)
    where = {"domain": restrict_domain.strip()} if (restrict_domain and restrict_domain.strip()) else None
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=where,
        include=["metadatas", "distances"],
    )
    ids_out = (res.get("ids") or [[]])[0]
    if not ids_out:
        return []
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out: List[Tuple[str, str, float]] = []
    for meta, dist in zip(metas, dists):
        if not meta:
            continue
        d = str(meta.get("domain", "")).strip()
        s = str(meta.get("subdomain", "")).strip()
        if not d or not s:
            continue
        out.append((d, s, _distance_to_score(dist)))
    return out
