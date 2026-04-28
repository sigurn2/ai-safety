"""Unit tests for RRF and hybrid retrieval glue (no MySQL/Chroma required)."""

from __future__ import annotations

from engine.rag_ingestion.hybrid_retrieval import reciprocal_rank_fuse


def test_reciprocal_rank_fuse_orders_by_sum() -> None:
    a = ["x", "y", "z"]
    b = ["y", "z", "w"]
    s = reciprocal_rank_fuse([a, b], k=60)
    assert s["y"] > s["x"]
    assert s["y"] > s["w"]
    assert "x" in s and "w" in s
