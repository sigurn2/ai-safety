#!/usr/bin/env python3
"""
卫报 Content API 冒烟：验证 GUARDIAN_API_KEY 与默认 AI 治理/安全检索。

功能：调用 crawler.sources.search_articles 打印一页摘要；不调用 LLM、不写库。
输入：命令行可选 --pages N（默认 1）；环境变量见 core.config。
输出：stdout 打印条数与标题/URL；退出码 0/1。
上下游：独立脚本；依赖 .env 中 GUARDIAN_API_KEY。

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from crawler.sources import (  # noqa: E402
    GuardianAPIError,
    search_articles,
    search_articles_multipage,
)

# 验证卫报 Content API 是否正常工作
def main() -> int:
    parser = argparse.ArgumentParser(description="Guardian API smoke test (AI safety/governance default query)")
    parser.add_argument("--pages", type=int, default=1, help="number of API pages to fetch (>=1)")
    parser.add_argument("--page-size", type=int, default=5, help="page size (1-50)")
    parser.add_argument("--section", type=str, default="", help="optional Guardian section e.g. technology")
    parser.add_argument("--query", type=str, default="", help="override search query (default: AI governance preset)")
    args = parser.parse_args()

    query = args.query.strip() or None
    section = args.section.strip() or None

    try:
        if args.pages <= 1:
            page = search_articles(
                query=query,
                page=1,
                page_size=args.page_size,
                section=section,
            )
            articles = page.articles
            print(f"status={page.status} total={page.total} pages={page.pages} current_page={page.current_page}")
        else:
            articles = search_articles_multipage(
                query=query,
                max_pages=args.pages,
                page_size=args.page_size,
                section=section,
            )
            print(f"fetched_articles={len(articles)} (multipage, max_pages={args.pages})")
    except GuardianAPIError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if e.status_code:
            print(f"  status_code={e.status_code}", file=sys.stderr)
        return 1

    for i, a in enumerate(articles, 1):
        print(f"\n--- {i} ---")
        print(a.title)
        print(a.web_url)
        if a.section_name:
            print(f"section={a.section_name}")
        if a.trail_text:
            print(f"lead: {a.trail_text[:200]}{'...' if len(a.trail_text) > 200 else ''}")
        if a.body_text:
            print(f"body_len={len(a.body_text)}")

    print(f"\nTotal articles listed: {len(articles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
