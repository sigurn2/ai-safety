"""
article_content Chroma 检索测试（伪向量，无需真实 API）。
"""

from __future__ import annotations

import tempfile
from typing import Any, List, Optional
from unittest.mock import patch

from engine.article_index.indexer import index_article
from engine.article_index.retriever import query_article_chunks


class _FakeLlmBackend:
    dim = 8

    def embed_texts(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            for i, b in enumerate(t.encode("utf-8")[:80]):
                v[i % self.dim] += float(b) / 255.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


def test_query_article_chunks_after_index() -> None:
    root = tempfile.mkdtemp()
    chroma_dir = f"{root}/chroma"
    be = _FakeLlmBackend()
    with patch("engine.article_index.indexer.save_article_chunk", return_value=1):
        n = index_article(
            article_id=42,
            title="EU AI Act enforcement",
            summary="Regulators discuss compliance timelines.",
            content="Paragraph one about governance.\n\nParagraph two about safety assessments.",
            source="test",
            risk_domain="Systemic & Ethical Risk (系统性与伦理风险)",
            published_at="2026-01-15",
            url="https://example.com/a",
            backend=be,
            chroma_persist_dir=chroma_dir,
            extraction_ctx={
                "main_topic": "AI regulation",
                "summary_structured": "Compliance focus",
                "tags": ["EU", "Act"],
                "entities": ["Commission"],
            },
        )
    assert n >= 1

    hits = query_article_chunks(
        "EU AI regulation compliance",
        top_k=5,
        backend=be,
        persist_directory=chroma_dir,
    )
    assert hits
    assert all(h.vector_id for h in hits)
    aid = {h.metadata.get("article_id") for h in hits}
    assert 42 in aid

    filtered = query_article_chunks(
        "governance",
        top_k=3,
        backend=be,
        persist_directory=chroma_dir,
        article_id_allowlist=[42],
    )
    assert filtered
    assert all(h.metadata.get("article_id") == 42 for h in filtered)
