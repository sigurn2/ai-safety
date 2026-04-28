#!/usr/bin/env python3
"""
Smoke test: write one synthetic article + one article_extractions row via core.mysql_db.

Usage (from repo root, venv optional):
  python scripts/mysql_smoke_write.py

Requires .env with MYSQL_* and tables from scripts/init_mysql.sql.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import LLM_MODEL, MYSQL_DATABASE, MYSQL_HOST, MYSQL_USER  # noqa: E402
from core.mysql_db import get_article_by_url, normalize_url, save_article, save_extraction  # noqa: E402


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    url = f"https://smoke-test.local/mysql-write/{stamp}"
    title = f"MySQL smoke write {stamp}"
    summary = "Synthetic summary for connectivity test."
    body = "Synthetic body paragraph. Safe to delete manually if desired."

    extraction = {
        "is_relevant": True,
        "content_type": "news",
        "main_topic": title[:256],
        "risk_domain": "Systemic & Ethical Risk (系统性与伦理风险)",
        "risk_subdomains": ["合规", "测试"],
        "entities": ["SmokeTestOrg"],
        "summary_structured": summary,
        "tags": ["smoke", "mysql"],
    }

    print(f"Target DB: {MYSQL_USER}@{MYSQL_HOST} / {MYSQL_DATABASE}")
    article_id, is_new = save_article(
        url=url,
        title=title,
        summary=summary,
        content=body,
        published_at=datetime.now(timezone.utc).replace(tzinfo=None),
        source="mysql_smoke_write",
    )
    print(f"save_article -> article_id={article_id}, is_new={is_new}")

    ext_id = save_extraction(
        article_id=article_id,
        extraction_dict=extraction,
        model_name=(LLM_MODEL or "smoke").strip(),
    )
    print(f"save_extraction -> extraction row id (last insert)={ext_id}")

    row = get_article_by_url(normalize_url(url))
    if not row:
        print("ERROR: get_article_by_url returned nothing after write.")
        sys.exit(1)
    print("verify: article row keys id, title_raw =", row["id"], row.get("title_raw", "")[:60])
    print("OK — MySQL write path succeeded.")


if __name__ == "__main__":
    main()
