"""
爬虫子系统：基于 Crawl4AI 的页面抓取与 LLM 抽取。

功能：对外导出异步入口 run_agentic_crawl；内部组合 core / engine.rag_ingestion。
输入/输出：见 crawler.agentic_crawl 模块级说明。
上下游：由 Streamlit app 或脚本调用；不直接依赖 UI。
"""

from crawler.agentic_crawl import run_agentic_crawl

__all__ = ["run_agentic_crawl"]
