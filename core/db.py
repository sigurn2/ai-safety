"""
SQLite 持久化：事件、关键词池、风险分类演进。

功能：为 AI 治理监测提供本地单文件库；子域向量已迁移至 Chroma（见 core.chroma_taxonomy）。
输入：路径来自 core.config.DB_PATH；业务数据以 Pydantic AIIncident 等为契约。
输出：DataFrame / 元组 / 布尔等查询结果；写操作在同一事务内更新 incidents 与 risk_taxonomy。
上下游：上游为 crawler 与 UI 写入；下游为 engine.rag_ingestion.retriever（读 risk_taxonomy 列表 + Chroma 检索）。
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from core.config import DB_PATH
from models.schema import AIIncident, RISK_DOMAIN_CHOICES


def _table_column_names(conn: sqlite3.Connection, table: str) -> List[str]:
    """
    功能：读取表列名，用于迁移判断。
    输入：已打开连接与表名。
    输出：列名字符串列表；无副作用。
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def _migrate_incidents_schema(conn: sqlite3.Connection) -> None:
    cols = set(_table_column_names(conn, "incidents"))
    if "risk_level" not in cols:
        conn.execute("ALTER TABLE incidents ADD COLUMN risk_level TEXT")
    if "risk_domain" not in cols:
        conn.execute("ALTER TABLE incidents ADD COLUMN risk_domain TEXT")
    if "risk_subdomain" not in cols:
        conn.execute("ALTER TABLE incidents ADD COLUMN risk_subdomain TEXT")
    if "category" in cols:
        conn.execute(
            """
            UPDATE incidents
            SET risk_level = category
            WHERE (risk_level IS NULL OR trim(risk_level) = '')
              AND category IS NOT NULL AND trim(category) != ''
            """
        )


def _migrate_drop_taxonomy_embeddings(conn: sqlite3.Connection) -> None:
    """废除 SQLite 子域向量缓存表（向量改存 Chroma）。"""
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='taxonomy_embeddings'"
    )
    if cur.fetchone():
        conn.execute("DROP TABLE taxonomy_embeddings")


def init_db() -> None:
    """
    功能：创建 incidents / watched_keywords / risk_taxonomy 表并迁移 incidents 列；删除旧 taxonomy_embeddings。
    输入：无（使用 DB_PATH）。
    输出：无；副作用：写 SQLite 文件。
    上下游：应用启动（Streamlit main）与脚本应先调用。
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS incidents
                 (id TEXT PRIMARY KEY,
                 title TEXT,
                 category TEXT,
                 entity TEXT,
                 content TEXT,
                 url TEXT,
                 tags TEXT,
                 timestamp DATETIME)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS watched_keywords
                 (keyword TEXT PRIMARY KEY,
                 first_seen DATETIME,
                 count INTEGER DEFAULT 1)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS risk_taxonomy
                 (domain TEXT NOT NULL,
                  subdomain TEXT NOT NULL,
                  first_seen DATETIME,
                  count INTEGER DEFAULT 1,
                  PRIMARY KEY (domain, subdomain))"""
    )
    _migrate_drop_taxonomy_embeddings(conn)
    _migrate_incidents_schema(conn)
    conn.commit()
    conn.close()


def coerce_risk_domain(raw: Optional[str]) -> str:
    """
    功能：将模型返回的近似主域表述归一为 RISK_DOMAIN_CHOICES 中的规范字符串。
    输入：原始字符串或 None。
    输出：三条主域之一；无 IO。
    上下游：incident_from_extraction、RAG pipeline 的 hint 归一均依赖此函数。
    """
    if not raw or not isinstance(raw, str):
        return RISK_DOMAIN_CHOICES[2]
    v = raw.strip()
    if v in RISK_DOMAIN_CHOICES:
        return v
    low = v.lower()
    if "恶意" in v or "malicious" in low or "abuse" in low or "攻击" in v:
        return RISK_DOMAIN_CHOICES[0]
    if "意外" in v or "失效" in v or "鲁棒" in v or "accidental" in low or "failure" in low or "halluc" in low:
        return RISK_DOMAIN_CHOICES[1]
    if "系统" in v or "伦理" in v or "systemic" in low or "ethical" in low or "bias" in low or "偏见" in v:
        return RISK_DOMAIN_CHOICES[2]
    return RISK_DOMAIN_CHOICES[2]


def incident_from_extraction(d: dict) -> AIIncident:
    """
    功能：将爬虫/LLM 单条 dict 清洗为 AIIncident（主域归一、子域截断）。
    输入：含 title、risk_domain、risk_subdomain 等键的 dict。
    输出：校验后的 AIIncident；不写库。
    上下游：UI 入库前调用；依赖 coerce_risk_domain。
    """
    data = dict(d)
    data["risk_domain"] = coerce_risk_domain(data.get("risk_domain"))
    sub = data.get("risk_subdomain")
    if sub is None or str(sub).strip() == "":
        data["risk_subdomain"] = "未指定子域"
    else:
        data["risk_subdomain"] = str(sub).strip()[:160]
    return AIIncident(**data)


def _bump_risk_taxonomy_cursor(c: sqlite3.Cursor, domain: str, subdomain: str) -> bool:
    """
    功能：在同一事务内更新 risk_taxonomy 计数或插入新 (主域, 子域)。
    输入：活动游标与主域、子域字符串。
    输出：True 表示本次为全新组合；False 表示已存在仅自增 count。
    副作用：仅写当前事务，需调用方 commit。
    """
    domain = (domain or "").strip()
    subdomain = (subdomain or "").strip()
    if not domain or not subdomain:
        return False
    c.execute(
        "SELECT count FROM risk_taxonomy WHERE domain = ? AND subdomain = ?",
        (domain, subdomain),
    )
    row = c.fetchone()
    if row:
        c.execute(
            "UPDATE risk_taxonomy SET count = count + 1 WHERE domain = ? AND subdomain = ?",
            (domain, subdomain),
        )
        return False
    c.execute(
        "INSERT INTO risk_taxonomy (domain, subdomain, first_seen, count) VALUES (?, ?, ?, 1)",
        (domain, subdomain, datetime.now()),
    )
    return True


def get_stats() -> Tuple[int, int, int]:
    """
    功能：汇总看板指标。
    输入：无。
    输出：(事件条数, 去重标签数, risk_taxonomy 行数)；只读连接。
    上下游：Streamlit 顶部 metric。
    """
    conn = sqlite3.connect(DB_PATH)
    count = int(pd.read_sql_query("SELECT COUNT(*) as total FROM incidents", conn).iloc[0]["total"])
    tags = pd.read_sql_query("SELECT tags FROM incidents", conn)
    try:
        tax_n = int(pd.read_sql_query("SELECT COUNT(*) AS n FROM risk_taxonomy", conn).iloc[0]["n"])
    except Exception:
        tax_n = 0
    conn.close()
    unique_tags: set = set()
    if not tags.empty and "tags" in tags.columns:
        for sublist in tags["tags"].dropna().astype(str).str.split(","):
            if sublist is not None:
                unique_tags.update(t for t in sublist if t)
    return count, len(unique_tags), tax_n


def get_risk_taxonomy_df() -> pd.DataFrame:
    """
    功能：读取完整风险分类演进表。
    输入：无。
    输出：按主域、count 排序的 DataFrame；只读。
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT domain, subdomain, count, first_seen FROM risk_taxonomy ORDER BY domain, count DESC",
            conn,
        )
    except Exception:
        df = pd.DataFrame(columns=["domain", "subdomain", "count", "first_seen"])
    conn.close()
    return df


def list_taxonomy_pairs() -> List[Tuple[str, str]]:
    """
    功能：列出 risk_taxonomy 中所有 (domain, subdomain) 对。
    输入：无。
    输出：元组列表；供 RAG 检索全量候选。
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT domain, subdomain FROM risk_taxonomy")
    rows = [(str(r[0]), str(r[1])) for r in cur.fetchall()]
    conn.close()
    return rows


def save_incident(incident: AIIncident, source_url: str = "") -> Tuple[bool, bool]:
    """
    功能：插入一条 incidents 并同步 bump risk_taxonomy。
    输入：AIIncident 与来源 URL。
    输出：(是否成功插入 incidents, 是否首次出现该主域+子域组合)；失败时回滚。
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    inc_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(incident.title.encode()).hexdigest()[:6]}"
    tag_str = ",".join(incident.tags)
    now = datetime.now()
    try:
        c.execute(
            """INSERT INTO incidents
               (id, title, risk_level, risk_domain, risk_subdomain, entity, content, url, tags, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                inc_id,
                incident.title,
                incident.risk_level,
                incident.risk_domain,
                incident.risk_subdomain,
                incident.entity,
                incident.summary,
                source_url,
                tag_str,
                now,
            ),
        )
        tax_new = _bump_risk_taxonomy_cursor(c, incident.risk_domain, incident.risk_subdomain)
        conn.commit()
        return True, tax_new
    except sqlite3.IntegrityError:
        conn.rollback()
        return False, False
    finally:
        conn.close()


def get_watched_keywords() -> pd.DataFrame:
    """
    功能：读取待观察关键词池 Top 60（按 count）。
    输入：无。
    输出：DataFrame；只读。
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT keyword, count FROM watched_keywords ORDER BY count DESC LIMIT 60", conn
        )
    except Exception:
        df = pd.DataFrame(columns=["keyword", "count"])
    conn.close()
    return df


def update_watched_keywords(new_tags: list) -> list:
    """
    功能：合并新标签入关键词池，新词插入、旧词 count+1。
    输入：标签字符串列表。
    输出：本次全新入库的标签列表；副作用：写 watched_keywords。
    上下游：爬虫 run 结束阶段调用。
    """
    if not new_tags:
        return []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        existing_df = pd.read_sql_query("SELECT keyword FROM watched_keywords", conn)
        existing = set(existing_df["keyword"].str.strip().tolist())
    except Exception:
        existing = set()

    newly_added = []
    for tag in new_tags:
        tag = tag.strip()
        if not tag:
            continue
        if tag in existing:
            c.execute("UPDATE watched_keywords SET count = count + 1 WHERE keyword = ?", (tag,))
        else:
            c.execute("INSERT OR IGNORE INTO watched_keywords VALUES (?, ?, 1)", (tag, datetime.now()))
            newly_added.append(tag)
            existing.add(tag)
    conn.commit()
    conn.close()
    return newly_added
