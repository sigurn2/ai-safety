"""
环境与运行配置（单一事实来源）。

功能：集中从 .env / 环境变量读取 LLM、嵌入、SQLite、爬虫与 RAG 开关，避免业务模块散落 os.getenv。
输入：进程环境；部分项支持非法数字时回退默认值。
输出：模块级常量（字符串/整型/布尔）；副作用：首次 import 时 load_dotenv(override=True)。
上下游：被 core.db、core.llm_client、crawler、engine.rag_ingestion 读取；Streamlit 侧展示 API 状态时可复用。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# --- LLM（OpenAI 兼容 Chat）---
API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# --- 嵌入（可与 Chat 同 base_url / 同 key；模型名独立配置）---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- 持久化 ---
DB_PATH: str = os.getenv("DB_PATH", "ai_governance.db")
# Chroma 子域向量库（本地目录，非独立服务）
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "chroma_data")

# --- Crawl4ai / Playwright ---
try:
    CRAWL_PAGE_TIMEOUT_MS: int = int(os.getenv("CRAWL_PAGE_TIMEOUT_MS", "90000"))
except ValueError:
    CRAWL_PAGE_TIMEOUT_MS = 90000
CRAWL_WAIT_UNTIL: str = (os.getenv("CRAWL_WAIT_UNTIL", "commit") or "commit").strip()

# --- RAG 子域路由（高频路径）---
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "8"))
RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
