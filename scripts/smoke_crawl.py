#!/usr/bin/env python3
"""
不调用 LLM 的爬取冒烟测试：验证 Crawl4AI 能否完成导航（与 app.py 中导航参数对齐）。

用法（在项目根目录）:
  ./venv/bin/python scripts/smoke_crawl.py [URL]

环境变量（与 app 一致）:
  CRAWL_PAGE_TIMEOUT_MS  默认 90000
  CRAWL_WAIT_UNTIL       默认 commit
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 保证可从项目根导入（若需要）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig  # noqa: E402


async def main(url: str) -> int:
    try:
        page_timeout = int(os.getenv("CRAWL_PAGE_TIMEOUT_MS", "90000"))
    except ValueError:
        page_timeout = 90000
    wait_until = (os.getenv("CRAWL_WAIT_UNTIL", "commit") or "commit").strip()
    session_id = f"smoke_{os.getpid()}"

    cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_until=wait_until,
        page_timeout=page_timeout,
        session_id=session_id,
    )
    print(f"url={url!r} wait_until={wait_until!r} page_timeout_ms={page_timeout}")
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=cfg)
    ok = bool(getattr(result, "success", False))
    print(f"success={ok}")
    if not ok:
        err = getattr(result, "error_message", "") or ""
        print("error_message:\n", err[:800])
        return 1
    html = getattr(result, "html", "") or ""
    print(f"html_length={len(html)}")
    return 0


if __name__ == "__main__":
    u = sys.argv[1] if len(sys.argv) > 1 else "https://cset.georgetown.edu/news/"
    raise SystemExit(asyncio.run(main(u)))
