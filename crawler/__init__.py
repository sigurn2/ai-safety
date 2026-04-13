"""
爬虫子系统：基于 Crawl4AI 的页面抓取与 LLM 抽取；信源适配见 crawler.sources。

功能：对外导出异步入口 run_agentic_crawl（延迟导入，避免仅使用 sources 时拉取 crawl4ai）。
输入/输出：见 crawler.agentic_crawl 模块级说明。
上下游：由 Streamlit app 或脚本调用；crawler.sources 可被脚本单独引用而不安装 Playwright 栈。
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_agentic_crawl"]


def __getattr__(name: str) -> Any:
    if name == "run_agentic_crawl":
        from crawler.agentic_crawl import run_agentic_crawl

        return run_agentic_crawl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
