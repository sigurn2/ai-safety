"""
RAG 子域路由包：嵌入检索 + LLM 归类。

功能：对外导出 apply_rag_to_incidents；内部模块分工见各文件顶部说明。
"""

from engine.rag_ingestion.pipeline import apply_rag_to_incidents

__all__ = ["apply_rag_to_incidents"]
