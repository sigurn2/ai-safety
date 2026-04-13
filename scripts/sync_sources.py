#!/usr/bin/env python3
"""
信源同步入口（CLI）：从卫报拉取 AI 治理/安全新闻并入库。

功能：调用 crawler.orchestrator.sync_guardian，打印摘要与 debug 日志；适合 cron 或手动触发。
输入：命令行参数；GUARDIAN_API_KEY 从 .env 或环境变量读取。
输出：stdout 日志；exit 0/1；副作用：SQLite 写入。

用法:
  ./venv/bin/python scripts/sync_sources.py
  ./venv/bin/python scripts/sync_sources.py --pages 3 --page-size 8
  ./venv/bin/python scripts/sync_sources.py --no-rag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import init_db  # noqa: E402
from crawler.orchestrator import sync_guardian  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Guardian AI safety/governance news to local DB")
    parser.add_argument("--pages", type=int, default=2, help="max API pages (default 2)")
    parser.add_argument("--page-size", type=int, default=10, help="page size 1-50 (default 10)")
    parser.add_argument("--section", type=str, default="", help="Guardian section filter e.g. technology")
    parser.add_argument("--query", type=str, default="", help="override search query")
    parser.add_argument("--no-rag", action="store_true", help="disable RAG refinement")
    args = parser.parse_args()

    init_db()

    r = sync_guardian(
        query=args.query.strip() or None,
        max_pages=args.pages,
        page_size=args.page_size,
        section=args.section.strip() or None,
        rag_enabled=False if args.no_rag else None,
    )

    for line in r.debug_log:
        print(line)

    print(f"\n--- 汇总 ---")
    print(f"入库: {r.saved}")
    print(f"已有（跳过）: {r.skipped_url_dup}")
    print(f"无关（跳过）: {r.skipped_no_incident}")
    print(f"失败: {r.failed}")
    if r.new_keywords:
        print(f"新关键词: {', '.join(r.new_keywords[:10])}{'...' if len(r.new_keywords)>10 else ''}")
    if r.new_subdomains:
        print(f"新子域: {', '.join(r.new_subdomains)}")

    return 1 if (r.failed > 0 and r.saved == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
