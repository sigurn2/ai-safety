#!/usr/bin/env python3
"""
混合检索冒烟：问题 → hybrid_retrieve → 打印证据（需 MySQL + Chroma + 可选 FULLTEXT）。

  python scripts/research_smoke.py "EU AI Act enforcement" --top-k 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.rag_ingestion.hybrid_retrieval import (  # noqa: E402
    evidence_hits_to_report_sources,
    hybrid_retrieve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval smoke (MySQL + Chroma).")
    parser.add_argument("question", help="Research question")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--risk-domain", type=str, default="")
    parser.add_argument("--source", type=str, default="")
    args = parser.parse_args()

    hits = hybrid_retrieve(
        args.question,
        top_k=args.top_k,
        risk_domain=args.risk_domain or None,
        source=args.source or None,
    )
    if not hits:
        print("No hits (empty index, filters too strict, or FULLTEXT/Chroma unavailable).")
        return
    for i, h in enumerate(hits, 1):
        preview = (h.chunk_text or "").replace("\n", " ")[:240]
        print(f"{i}. vector_id={h.vector_id} article_id={h.article_id} rrf={h.rrf_score:.4f}")
        print(f"   {preview}…")

    sources = evidence_hits_to_report_sources(hits)
    print("report_source_preview:", [{"article_id": s["article_id"], "chunk_id": s.get("chunk_id")} for s in sources])


if __name__ == "__main__":
    main()
