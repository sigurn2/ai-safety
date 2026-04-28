"""
Index one article into the article Chroma collection and persist chunk rows in MySQL.

Uses ARTICLE_CHROMA_DIR (separate from risk_taxonomy Chroma). Requires chromadb.

Body chunking: paragraph-first merge with target/max character bounds and cross-chunk overlap.
Summary chunk: title + API summary + optional extraction fields (summary_structured, main_topic, tags, entities).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from core.config import (
    ARTICLE_CHROMA_DIR,
    EMBEDDING_MODEL,
    INDEX_CHUNK_MAX_CHARS,
    INDEX_CHUNK_OVERLAP_CHARS,
    INDEX_CHUNK_TARGET_CHARS,
    INDEX_SUMMARY_MAX_CHARS,
)
from core.llm_ports import LlmBackend
from core.mysql_db import save_article_chunk
from engine.rag_ingestion.embedder import embed_documents

ARTICLE_COLLECTION = "article_content"

_PARA_SPLIT = re.compile(r"\n\s*\n+")


def _chunk_uid(article_id: int, chunk_type: str, chunk_index: int) -> str:
    ct = "summary" if (chunk_type or "").strip().lower() == "summary" else "body"
    raw = f"{int(article_id)}\0{ct}\0{int(chunk_index)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _token_guess(text: str) -> int:
    return max(1, len((text or "").strip()) // 4)


def _split_paragraphs(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = _PARA_SPLIT.split(t)
    return [p.strip() for p in parts if p and p.strip()]


def _split_long_paragraph(para: str, max_chars: int, overlap: int) -> List[str]:
    """Fixed windows for a single paragraph exceeding max_chars."""
    if len(para) <= max_chars:
        return [para]
    out: List[str] = []
    start = 0
    while start < len(para):
        end = min(start + max_chars, len(para))
        piece = para[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(para):
            break
        start = max(0, end - overlap)
    return out


def _flatten_to_pieces(text: str, max_chars: int, overlap: int) -> List[str]:
    pieces: List[str] = []
    for para in _split_paragraphs(text):
        pieces.extend(_split_long_paragraph(para, max_chars, overlap))
    return pieces


def _merge_pieces_to_chunks(
    pieces: List[str],
    target_chars: int,
    max_chars: int,
    cross_overlap: int,
) -> List[str]:
    """Greedy merge of paragraph pieces; then prepend tail overlap between consecutive chunks."""
    if not pieces:
        return []
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

    for piece in pieces:
        sep = 2 if buf else 0
        add_len = sep + len(piece)
        if not buf:
            buf = [piece]
            buf_len = len(piece)
            if buf_len >= target_chars:
                flush()
            continue
        if buf_len + add_len <= max_chars:
            buf.append(piece)
            buf_len += add_len
            if buf_len >= target_chars:
                flush()
        else:
            flush()
            buf = [piece]
            buf_len = len(piece)
            if buf_len >= target_chars:
                flush()
    flush()

    if not chunks:
        return []
    if len(chunks) == 1:
        return chunks

    overlapped: List[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = overlapped[-1]
        tail_n = min(cross_overlap, len(prev))
        tail = prev[-tail_n:] if tail_n > 0 else ""
        nxt = chunks[i]
        if tail:
            overlapped.append(f"{tail}\n\n{nxt}".strip())
        else:
            overlapped.append(nxt)
    return overlapped


def _iter_body_chunks(
    text: str,
    *,
    target_chars: int = INDEX_CHUNK_TARGET_CHARS,
    max_chars: int = INDEX_CHUNK_MAX_CHARS,
    overlap: int = INDEX_CHUNK_OVERLAP_CHARS,
) -> List[str]:
    pieces = _flatten_to_pieces(text, max_chars, overlap)
    if not pieces:
        return []
    return _merge_pieces_to_chunks(pieces, target_chars, max_chars, overlap)


def _cap_block(s: str, max_chars: int) -> str:
    t = (s or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def _build_summary_text(
    title: str,
    api_summary: str,
    extraction_ctx: Optional[Dict[str, Any]],
    max_chars: int,
) -> str:
    """Single summary chunk: title, trail/summary, then structured extraction lines."""
    lines: List[str] = []
    title_s = (title or "").strip()
    api_s = (api_summary or "").strip()
    if title_s:
        lines.append(title_s)
    if api_s:
        lines.append(api_s)

    ctx = extraction_ctx or {}
    mt = str(ctx.get("main_topic") or "").strip()
    ss = str(ctx.get("summary_structured") or "").strip()
    if mt:
        lines.append(f"主题: {mt}")
    if ss:
        lines.append(f"摘要: {ss}")

    tags = ctx.get("tags") or []
    if isinstance(tags, list) and tags:
        tag_str = ", ".join(str(x).strip() for x in tags[:16] if str(x).strip())
        if tag_str:
            lines.append(f"标签: {tag_str}")
    ents = ctx.get("entities") or []
    if isinstance(ents, list) and ents:
        ent_str = ", ".join(str(x).strip() for x in ents[:20] if str(x).strip())
        if ent_str:
            lines.append(f"主体: {ent_str}")

    block = "\n\n".join(lines) if lines else ""
    return _cap_block(block, max_chars)


def _build_chunk_specs(
    article_id: int,
    title: str,
    summary: str,
    content: str,
    extraction_ctx: Optional[Dict[str, Any]],
) -> List[Tuple[str, int, str]]:
    specs: List[Tuple[str, int, str]] = []
    title_s = (title or "").strip()

    summary_block = _build_summary_text(
        title_s,
        summary,
        extraction_ctx,
        INDEX_SUMMARY_MAX_CHARS,
    )
    if summary_block:
        specs.append(("summary", 0, summary_block))

    for i, part in enumerate(_iter_body_chunks(content)):
        specs.append(("body", i, part))

    if not specs and title_s:
        specs.append(("body", 0, title_s))
    return specs


def index_article(
    *,
    article_id: int,
    title: str,
    summary: str,
    content: str,
    source: str = "",
    risk_domain: str = "",
    published_at: str = "",
    url: str = "",
    backend: Optional[LlmBackend] = None,
    embedding_model: Optional[str] = None,
    chroma_persist_dir: Optional[str] = None,
    extraction_ctx: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Embed article fragments, upsert into Chroma, and write article_chunks rows.
    Returns number of chunks indexed.

    extraction_ctx: optional dict aligned with merge_article_with_rag output (main_topic,
    summary_structured, tags, entities) to enrich the summary chunk only.
    """
    emodel = (embedding_model or EMBEDDING_MODEL or "").strip()
    specs = _build_chunk_specs(article_id, title, summary, content, extraction_ctx)
    if not specs:
        return 0

    texts = [s[2] for s in specs]
    embeddings = embed_documents(texts, model=emodel or None, backend=backend)

    path = chroma_persist_dir if chroma_persist_dir is not None else ARTICLE_CHROMA_DIR
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name=ARTICLE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids: List[str] = []
    metadatas: List[dict] = []
    for (ctype, idx, text) in specs:
        uid = _chunk_uid(article_id, ctype, idx)
        ids.append(uid)
        metadatas.append(
            {
                "article_id": int(article_id),
                "chunk_type": "summary" if ctype == "summary" else "body",
                "chunk_index": int(idx),
                "source": (source or "")[:128],
                "risk_domain": (risk_domain or "")[:128],
                "published_at": (published_at or "")[:32],
                "url": (url or "")[:1024],
                "title": (title or "")[:256],
            }
        )

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    for (ctype, idx, text), uid in zip(specs, ids):
        save_article_chunk(
            article_id,
            chunk_uid=uid,
            chunk_type="summary" if ctype == "summary" else "body",
            chunk_index=idx,
            chunk_text=text,
            token_estimate=_token_guess(text),
            embedding_model=emodel,
            vector_id=uid,
        )
    return len(specs)
