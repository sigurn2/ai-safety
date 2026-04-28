#!/usr/bin/env python3
"""
将 article_extractions.risk_domain 批量规范为 models.schema.RISK_DOMAIN_CHOICES 中的完整字符串。

用于修正历史数据（模型曾输出「恶意使用」、重复括号等）。新项目写入已在 save_extraction 内调用 coerce_risk_domain。

用法（项目根目录）:
  python3 scripts/normalize_mysql_risk_domains.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.db import coerce_risk_domain
from core.mysql_db import mysql_conn


def main() -> None:
    updated = 0
    with mysql_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, risk_domain FROM article_extractions")
        rows = cur.fetchall() or []
        for row in rows:
            rid = int(row["id"])
            old = str(row.get("risk_domain") or "")
            new = coerce_risk_domain(old)
            if new != old:
                cur.execute(
                    "UPDATE article_extractions SET risk_domain = %s WHERE id = %s",
                    (new, rid),
                )
                updated += 1
    print(f"article_extractions: 已更新 {updated} 行的 risk_domain。")


if __name__ == "__main__":
    main()
