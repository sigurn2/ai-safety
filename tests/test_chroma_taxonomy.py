"""
Chroma 子域 RAG 测试（无需真实 API Key）。

运行：
  ./venv/bin/pytest tests/test_chroma_taxonomy.py -q

手动冒烟（需 Key 与可用嵌入接口）：
  配置 .env 后启动 streamlit，执行一次爬取；或临时脚本调用 retrieve_similar_subdomains。
"""

from __future__ import annotations

import sqlite3
import tempfile
from typing import Any, List, Optional

import pytest

from core.db import init_db
from engine.rag_ingestion.retriever import retrieve_similar_subdomains


class _FakeLlmBackend:
    """确定性伪向量：同字符串得到同一单位向量，便于 Chroma 检索稳定。"""

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
            for i, b in enumerate(t.encode("utf-8")[: 50]):
                v[i % self.dim] += float(b) / 255.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out

    def chat_completion_json(
        self,
        messages: List[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> Any:
        raise NotImplementedError


@pytest.fixture
def isolated_db_and_chroma(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    root = tempfile.mkdtemp()
    db_path = f"{root}/test.db"
    chroma_dir = f"{root}/chroma"
    monkeypatch.setattr("core.db.DB_PATH", db_path)
    init_db()
    return db_path, chroma_dir


def test_retrieve_top_k_with_fake_embed(isolated_db_and_chroma: tuple[str, str]) -> None:
    db_path, chroma_dir = isolated_db_and_chroma
    dom = "Malicious Use (恶意滥用)"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO risk_taxonomy (domain, subdomain, first_seen, count) VALUES (?, ?, datetime('now'), 1)",
        (dom, "越狱攻击"),
    )
    conn.execute(
        "INSERT INTO risk_taxonomy (domain, subdomain, first_seen, count) VALUES (?, ?, datetime('now'), 1)",
        (dom, "数据投毒"),
    )
    conn.commit()
    conn.close()

    be = _FakeLlmBackend()
    hits = retrieve_similar_subdomains(
        "越狱 攻击 模型安全",
        top_k=2,
        backend=be,
        persist_directory=chroma_dir,
    )
    assert len(hits) >= 1
    assert all(h.domain == dom for h in hits)
    subs = {h.subdomain for h in hits}
    assert "越狱攻击" in subs or "数据投毒" in subs


def test_restrict_domain_filters(isolated_db_and_chroma: tuple[str, str]) -> None:
    db_path, chroma_dir = isolated_db_and_chroma
    d1 = "Malicious Use (恶意滥用)"
    d2 = "Accidental Failure (意外失效)"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO risk_taxonomy (domain, subdomain, first_seen, count) VALUES (?, ?, datetime('now'), 1)",
        (d1, "A类"),
    )
    conn.execute(
        "INSERT INTO risk_taxonomy (domain, subdomain, first_seen, count) VALUES (?, ?, datetime('now'), 1)",
        (d2, "B类"),
    )
    conn.commit()
    conn.close()

    be = _FakeLlmBackend()
    hits = retrieve_similar_subdomains(
        "测试查询",
        top_k=4,
        restrict_domain=d2,
        backend=be,
        persist_directory=chroma_dir,
    )
    assert hits
    assert all(h.domain == d2 for h in hits)
    assert all(h.subdomain == "B类" for h in hits)
