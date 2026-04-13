"""
嵌入门面：把「查询 / 文档」文本交给 LlmBackend 做向量化。

功能：对 RAG 检索层提供稳定入口；具体厂商差异由 core.llm_client.OpenAICompatibleBackend 等实现屏蔽。
输入：字符串或列表；可选 model 与可注入 backend（单测或用户侧 key）。
输出：float 向量；副作用：调用嵌入 HTTP（经 backend）。
上下游：仅被 engine.rag_ingestion.retriever 调用；不访问数据库。
"""

from __future__ import annotations

from typing import List, Optional

from core.llm_client import default_llm_backend
from core.llm_ports import LlmBackend


def embed_query(text: str, *, model: Optional[str] = None, backend: Optional[LlmBackend] = None) -> List[float]:
    """
    功能：单条查询文本转向量（用于与 taxonomy 标签比相似度）。
    输入：非空推荐；backend 缺省用进程默认 OpenAI 兼容后端。
    输出：一维 float 列表。
    """
    b = backend or default_llm_backend()
    return b.embed_texts([text.strip()], model=model)[0]


def embed_documents(
    texts: List[str],
    *,
    model: Optional[str] = None,
    backend: Optional[LlmBackend] = None,
) -> List[List[float]]:
    """
    功能：批量嵌入（如「主域 | 子域」合成标签），顺序与输入一致。
    输入：字符串列表；跳过空串由调用方保证列表语义。
    输出：向量列表。
    """
    cleaned = [t.strip() for t in texts]
    if not cleaned:
        return []
    b = backend or default_llm_backend()
    return b.embed_texts(cleaned, model=model)
