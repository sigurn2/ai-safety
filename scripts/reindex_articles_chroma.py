#!/usr/bin/env python3
"""
对 MySQL 中已有文章重跑 index_article（Chroma + article_chunks upsert）。

需配置 .env 中 MYSQL_*、嵌入 API；已装 chromadb。

  python scripts/reindex_articles_chroma.py --limit 50
  python scripts/reindex_articles_chroma.py --article-id 42
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import LLM_MODEL  # noqa: E402
from core.mysql_db import mysql_conn  # noqa: E402
from core.llm_client import OpenAICompatibleBackend  # noqa: E402
from core.config import API_KEY, BASE_URL  # noqa: E402
from engine.article_index.indexer import index_article  # noqa: E402


def _as_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        try:
            obj = json.loads(val)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except json.JSONDecodeError:
            return []
    return []


def _row_to_extraction_ctx(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "main_topic": str(row.get("main_topic") or ""),
        "summary_structured": str(row.get("summary_structured") or ""),
        "tags": _as_list(row.get("tags_raw")),
        "entities": _as_list(row.get("entities_json")),
        "risk_domain": str(row.get("risk_domain") or ""),
    }


def reindex_one(
    article_id: int,
    *,
    backend: Optional[OpenAICompatibleBackend] = None,
) -> int:
    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.id, a.title_raw, a.summary_raw, a.content_raw, a.source,
                       a.published_at, a.normalized_url,
                       e.main_topic, e.summary_structured, e.tags_raw, e.entities_json, e.risk_domain
                FROM articles a
                LEFT JOIN article_extractions e ON e.article_id = a.id
                WHERE a.id = %s
                """,
                (article_id,),
            )
            row = cur.fetchone()
    if not row:
        raise ValueError(f"article_id={article_id} not found")

    summary = str(row.get("summary_raw") or "")[:8192]
    content = str(row.get("content_raw") or "")
    if not content.strip():
        content = summary
    pub = row.get("published_at")
    pub_str = pub.strftime("%Y-%m-%d") if isinstance(pub, datetime) else ""

    ctx = None
    if row.get("main_topic") or row.get("summary_structured"):
        ctx = _row_to_extraction_ctx(row)
    rd = str(row.get("risk_domain") or "")

    be = backend or OpenAICompatibleBackend(api_key=API_KEY, base_url=BASE_URL)
    return index_article(
        article_id=int(row["id"]),
        title=str(row.get("title_raw") or ""),
        summary=summary,
        content=content,
        source=str(row.get("source") or ""),
        risk_domain=rd,
        published_at=pub_str,
        url=str(row.get("normalized_url") or ""),
        backend=be,
        embedding_model=(LLM_MODEL or "").strip(),
        extraction_ctx=ctx,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Max articles when scanning")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--article-id", type=int, default=0, help="Reindex a single id")
    args = parser.parse_args()

    if args.article_id > 0:
        n = reindex_one(args.article_id)
        print(f"article_id={args.article_id} chunks={n}")
        return

    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM articles ORDER BY id LIMIT %s OFFSET %s",
                (max(1, args.limit), max(0, args.offset)),
            )
            ids = [int(r["id"]) for r in (cur.fetchall() or [])]

    be = OpenAICompatibleBackend(api_key=API_KEY, base_url=BASE_URL)
    ok = 0
    for aid in ids:
        try:
            n = reindex_one(aid, backend=be)
            print(f"article_id={aid} chunks={n}")
            ok += 1
        except Exception as e:
            print(f"article_id={aid} ERROR {type(e).__name__}: {e}")
    print(f"done. processed={ok}/{len(ids)}")


if __name__ == "__main__":
    main()
