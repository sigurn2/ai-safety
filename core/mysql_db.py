"""
MySQL persistence layer for Phase 1 (MVP).

Provides CRUD helpers for:
- immutable raw articles
- one metadata row per article (article_extractions, upsert by article_id)
- article chunks (vector bridge)
- research reports and citations
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pymysql
import pymysql.err
from pymysql.cursors import DictCursor

from core.config import (
    MYSQL_CHARSET,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)
from core.db import coerce_risk_domain


def normalize_url(url: str) -> str:
    """
    Canonicalize URL for dedupe: lowercase host, drop fragment, sort query parameters.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    query = urlencode(sorted(query_pairs))
    netloc = parts.netloc.lower()
    path = re.sub(r"/+$", "", parts.path or "")
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def compute_content_hash(title: str, summary: str, content: str) -> str:
    raw = "\n".join([(title or "").strip(), (summary or "").strip(), (content or "").strip()])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@contextmanager
def mysql_conn():
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset=MYSQL_CHARSET,
        autocommit=False,
        cursorclass=DictCursor,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_json_array(items: Iterable[str]) -> str:
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    return json.dumps(cleaned, ensure_ascii=False)


def get_article_by_url(normalized_url: str) -> Optional[Dict[str, Any]]:
    if not normalized_url:
        return None
    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, normalized_url, title_raw, summary_raw, content_raw, published_at, content_hash
                FROM articles
                WHERE normalized_url = %s
                LIMIT 1
                """,
                (normalized_url,),
            )
            return cur.fetchone()


def save_article(
    url: str,
    title: str,
    summary: str,
    content: str,
    published_at: Optional[datetime],
    source: str,
) -> Tuple[int, bool]:
    """
    Insert immutable article if not exists. Returns (article_id, is_new).
    """
    normalized_url = normalize_url(url)
    if not normalized_url:
        raise ValueError("url is required")
    content_hash = compute_content_hash(title, summary, content)

    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM articles WHERE normalized_url = %s LIMIT 1", (normalized_url,))
            row = cur.fetchone()
            if row:
                return int(row["id"]), False

            cur.execute(
                """
                INSERT INTO articles (
                    normalized_url, source, title_raw, summary_raw, content_raw,
                    published_at, content_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    normalized_url,
                    (source or "").strip(),
                    (title or "").strip(),
                    (summary or "").strip(),
                    (content or "").strip(),
                    published_at,
                    content_hash,
                ),
            )
            return int(cur.lastrowid), True


def save_extraction(
    article_id: int,
    extraction_dict: Dict[str, Any],
    model_name: str,
) -> int:
    """
    Upsert one article-level extraction row (unique article_id).
    extraction_dict：与 ArticleExtractionPayload / merge_article_with_rag 对齐；多余键被忽略。
    """
    if article_id <= 0:
        raise ValueError("article_id must be positive")
    d = dict(extraction_dict or {})
    tags = d.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    subs = d.get("risk_subdomains") or []
    if not isinstance(subs, list):
        subs = []
    subs_clean = [str(x).strip() for x in subs if str(x).strip()]
    ents = d.get("entities") or []
    if not isinstance(ents, list):
        ents = []
    ents_clean = [str(x).strip() for x in ents if str(x).strip()]

    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO article_extractions (
                    article_id, model_name, content_type, main_topic,
                    risk_domain, risk_subdomains_json, entities_json,
                    summary_structured, tags_raw
                ) VALUES (
                    %s, %s, %s, %s, %s, CAST(%s AS JSON), CAST(%s AS JSON),
                    %s, CAST(%s AS JSON)
                )
                ON DUPLICATE KEY UPDATE
                    model_name = VALUES(model_name),
                    content_type = VALUES(content_type),
                    main_topic = VALUES(main_topic),
                    risk_domain = VALUES(risk_domain),
                    risk_subdomains_json = VALUES(risk_subdomains_json),
                    entities_json = VALUES(entities_json),
                    summary_structured = VALUES(summary_structured),
                    tags_raw = VALUES(tags_raw),
                    id = LAST_INSERT_ID(id)
                """,
                (
                    article_id,
                    (model_name or "").strip(),
                    str(d.get("content_type", "other") or "other").strip()[:32],
                    str(d.get("main_topic", "")).strip()[:512],
                    coerce_risk_domain(d.get("risk_domain"))[:128],
                    _ensure_json_array(subs_clean),
                    _ensure_json_array(ents_clean),
                    str(d.get("summary_structured", d.get("summary", ""))).strip()[:512],
                    _ensure_json_array(tags),
                ),
            )
            return int(cur.lastrowid)


def save_article_chunk(
    article_id: int,
    *,
    chunk_uid: str,
    chunk_type: str,
    chunk_index: int,
    chunk_text: str,
    token_estimate: int = 0,
    embedding_model: str = "",
    vector_id: str,
) -> int:
    """
    Persist chunk metadata for a Chroma vector. Returns article_chunks.id.
    The chunk text remains in MySQL for traceability and index rebuilds.
    """
    if article_id <= 0:
        raise ValueError("article_id must be positive")
    uid = (chunk_uid or "").strip()
    vid = (vector_id or "").strip()
    if not uid or not vid:
        raise ValueError("chunk_uid and vector_id are required")

    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO article_chunks (
                    article_id, chunk_uid, chunk_type, chunk_index, chunk_text,
                    token_estimate, embedding_model, vector_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    id = LAST_INSERT_ID(id),
                    chunk_text = VALUES(chunk_text),
                    token_estimate = VALUES(token_estimate),
                    embedding_model = VALUES(embedding_model),
                    vector_id = VALUES(vector_id)
                """,
                (
                    article_id,
                    uid,
                    (chunk_type or "body").strip(),
                    int(chunk_index),
                    (chunk_text or "").strip(),
                    max(0, int(token_estimate)),
                    (embedding_model or "").strip(),
                    vid,
                ),
            )
            return int(cur.lastrowid)


def get_chunk_ids_by_vector_ids(vector_ids: List[str]) -> Dict[str, int]:
    """
    Map Chroma vector ids back to article_chunks.id for report citations.
    """
    cleaned = [str(v).strip() for v in vector_ids if str(v).strip()]
    if not cleaned:
        return {}
    placeholders = ",".join(["%s"] * len(cleaned))
    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, vector_id FROM article_chunks WHERE vector_id IN ({placeholders})",
                cleaned,
            )
            rows = cur.fetchall() or []
    return {str(row["vector_id"]): int(row["id"]) for row in rows}


def _fulltext_boolean_terms(query: str) -> str:
    """Build a simple BOOLEAN MODE string (+term per token)."""
    raw = (query or "").strip()
    if not raw:
        return ""
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", raw, flags=re.UNICODE)
    if not tokens:
        return ""
    parts: List[str] = []
    for t in tokens[:24]:
        tt = t.strip()
        if not tt:
            continue
        parts.append(f"+{tt}")
    return " ".join(parts)


def search_chunks_fulltext(
    query: str,
    *,
    limit: int = 30,
    article_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    FULLTEXT search on article_chunks.chunk_text (requires ft_chunk_text index).
    Returns rows: id, article_id, vector_id, chunk_text, ft_score.
    """
    boolq = _fulltext_boolean_terms(query)
    if not boolq:
        return []
    lim = max(1, min(int(limit), 200))

    sql_parts = [
        """
        SELECT id, article_id, vector_id, chunk_text,
               MATCH(chunk_text) AGAINST(%s IN BOOLEAN MODE) AS ft_score
        FROM article_chunks
        WHERE MATCH(chunk_text) AGAINST(%s IN BOOLEAN MODE)
        """
    ]
    params: List[Any] = [boolq, boolq]
    if article_ids:
        aids = []
        for x in article_ids:
            try:
                xi = int(x)
                if xi > 0:
                    aids.append(xi)
            except (TypeError, ValueError):
                continue
        if not aids:
            return []
        ph = ",".join(["%s"] * len(aids))
        sql_parts.append(f" AND article_id IN ({ph})")
        params.extend(aids)
    sql_parts.append(" ORDER BY ft_score DESC LIMIT %s")
    params.append(lim)

    with mysql_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("".join(sql_parts), params)
            except pymysql.err.OperationalError:
                return []
            return list(cur.fetchall() or [])


def list_article_ids_by_filters(
    *,
    risk_domain: Optional[str] = None,
    source: Optional[str] = None,
    published_after: Optional[datetime] = None,
    published_before: Optional[datetime] = None,
    limit: int = 5000,
) -> List[int]:
    """
    Narrow articles for hybrid retrieval using articles + article_extractions.
    """
    lim = max(1, min(int(limit), 20000))
    joins = "FROM articles a"
    wheres: List[str] = ["1=1"]
    params: List[Any] = []

    if risk_domain and str(risk_domain).strip():
        joins += " INNER JOIN article_extractions e ON e.article_id = a.id"
        wheres.append("e.risk_domain = %s")
        params.append(str(risk_domain).strip())
    if source and str(source).strip():
        wheres.append("a.source = %s")
        params.append(str(source).strip())
    if published_after is not None:
        wheres.append("a.published_at >= %s")
        params.append(published_after)
    if published_before is not None:
        wheres.append("a.published_at <= %s")
        params.append(published_before)

    sql = f"SELECT DISTINCT a.id AS id {joins} WHERE " + " AND ".join(wheres) + " LIMIT %s"
    params.append(lim)

    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
    return [int(r["id"]) for r in rows]


def get_articles_brief_by_ids(article_ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
    """
    Fetch id → row for articles.title_raw / source / normalized_url（供深度调研报告引用上下文）。
    """
    ids_ordered: List[int] = []
    seen: set[int] = set()
    for x in article_ids:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if xi > 0 and xi not in seen:
            seen.add(xi)
            ids_ordered.append(xi)
    if not ids_ordered:
        return {}
    placeholders = ",".join(["%s"] * len(ids_ordered))
    sql = (
        f"SELECT id, title_raw, source, normalized_url FROM articles "
        f"WHERE id IN ({placeholders})"
    )
    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, ids_ordered)
            rows = cur.fetchall() or []
    return {int(r["id"]): dict(r) for r in rows}


def build_report_source_rows(
    hits: List[Dict[str, Any]],
    *,
    vector_id_key: str = "vector_id",
    article_id_key: str = "article_id",
    score_key: str = "rrf_score",
) -> List[Dict[str, Any]]:
    """
    Map hybrid evidence hits to save_research_report sources list (chunk_id resolved via vector_id).
    """
    if not hits:
        return []
    vids = [str(h.get(vector_id_key) or "").strip() for h in hits if h.get(vector_id_key)]
    idmap = get_chunk_ids_by_vector_ids([v for v in vids if v])
    out: List[Dict[str, Any]] = []
    for i, h in enumerate(hits, 1):
        vid = str(h.get(vector_id_key) or "").strip()
        if not vid:
            continue
        try:
            aid = int(h.get(article_id_key) or 0)
        except (TypeError, ValueError):
            aid = 0
        if aid <= 0:
            continue
        try:
            sc = float(h.get(score_key) or 0.0)
        except (TypeError, ValueError):
            sc = 0.0
        out.append(
            {
                "article_id": aid,
                "chunk_id": idmap.get(vid),
                "relevance_score": max(0.0, min(1.0, sc)),
                "citation_label": str(h.get("citation_label") or f"来源 {i}")[:64],
            }
        )
    return out


def save_research_report(
    question: str,
    filters: Dict[str, Any],
    report_markdown: str,
    *,
    model_name: str = "",
    sources: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Persist a generated research report and its article/chunk citations.
    Returns research_reports.id.
    """
    srcs = sources or []
    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_reports (question, filters_json, report_markdown, model_name)
                VALUES (%s, CAST(%s AS JSON), %s, %s)
                """,
                (
                    (question or "").strip(),
                    json.dumps(filters or {}, ensure_ascii=False, default=str),
                    (report_markdown or "").strip(),
                    (model_name or "").strip(),
                ),
            )
            report_id = int(cur.lastrowid)

            for i, src in enumerate(srcs, 1):
                article_id = int(src.get("article_id") or 0)
                if article_id <= 0:
                    continue
                chunk_id = src.get("chunk_id")
                try:
                    chunk_id_int = int(chunk_id) if chunk_id is not None else None
                except (TypeError, ValueError):
                    chunk_id_int = None
                try:
                    score = float(src.get("relevance_score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                label = str(src.get("citation_label") or f"来源 {i}").strip()
                cur.execute(
                    """
                    INSERT INTO research_report_sources (
                        report_id, article_id, chunk_id, relevance_score, citation_label
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        report_id,
                        article_id,
                        chunk_id_int,
                        max(0.0, min(1.0, score)),
                        label[:64],
                    ),
                )
        return report_id


def list_research_reports(limit: int = 20) -> List[Dict[str, Any]]:
    """Recent research_reports rows（id / question / model_name / created_at），供看板列出历史。"""
    lim = max(1, min(int(limit), 200))
    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, model_name, created_at
                FROM research_reports
                ORDER BY id DESC
                LIMIT %s
                """,
                (lim,),
            )
            return list(cur.fetchall() or [])


def get_research_report_by_id(report_id: int) -> Optional[Dict[str, Any]]:
    """Load one report + optional sources summary（用于历史回放）。"""
    rid = int(report_id)
    if rid <= 0:
        return None
    with mysql_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, filters_json, report_markdown, model_name, created_at
                FROM research_reports WHERE id = %s LIMIT 1
                """,
                (rid,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                """
                SELECT article_id, chunk_id, citation_label, relevance_score
                FROM research_report_sources
                WHERE report_id = %s
                ORDER BY id ASC
                """,
                (rid,),
            )
            srcs = list(cur.fetchall() or [])
    out = dict(row)
    out["sources"] = srcs
    return out
