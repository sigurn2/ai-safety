"""
问答式深度调研：在混合检索证据基础上调用 LLM 生成 Markdown 报告。

功能：拼证据块 → chat 补全；引用角标 [来源 n] 与 hybrid 证据顺序一致。
第一轮增强：system 约束 6～8 节、每节展开篇幅、综合/局限/参考文献；单块材料上限 5200 字。
上下游：Streamlit 深度调研 Tab、scripts 可复用；依赖 core.mysql_db、core.llm_client。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.config import LLM_MODEL
from core.llm_client import OpenAICompatibleBackend
from core.mysql_db import get_articles_brief_by_ids
from engine.rag_ingestion.hybrid_retrieval import EvidenceHit

_SYSTEM = (
    "你是 AI 治理与安全领域的调研员。根据用户问题与给定证据片段撰写**丰满、可读**的 Markdown 调研报告（面向决策与内参阅读）。\n"
    "硬性要求：\n"
    "- 使用 Markdown：正文前须有一级标题；主体至少 **6～8 个二级标题（##）**，按问题自拟小标题，避免只有 2～3 节。\n"
    "- 每个二级标题下：先 **2～4 句** 概括要点，再 **至少一段**（5～8 句）展开背景、主体、时间线、争议点或政策含义；勿仅用一句话结束该节。\n"
    "- **引用**：重要论断须带 [来源 n]（n 与材料「来源 n」一致）；鼓励一节内综合多段材料时使用多个角标，如 [来源 2][来源 5]。\n"
    "- **忠于证据**：不得编造材料中不存在的事实、数据或引语；材料不足处如实写「证据未涉及」。\n"
    "- **综合节**：在主体之后增一节「## 综合与交叉观察」（或相近标题），横向对比或归纳 2～3 点，并引用至少 **3** 个不同来源编号。\n"
    "- **局限**：另起一节「## 证据局限与未覆盖」，说明检索片段的片面性、语料缺口或矛盾处。\n"
    "- **参考文献**：最后一节「## 参考文献」，逐条列出本次使用过的 [来源 n] + 标题（可附 URL），与正文引用对应。\n"
    "- 全文建议 **不少于 2000 汉字**（若证据极短则尽量写满分析，并诚实说明篇幅受限原因）。\n"
)


def _pack_evidence(question: str, hits: List[EvidenceHit], briefs: Dict[int, Dict[str, Any]]) -> str:
    lines: List[str] = [
        f"## 问题\n{question.strip()}",
        "",
        "## 证据材料",
        "",
    ]
    for i, h in enumerate(hits, 1):
        b = briefs.get(h.article_id) or {}
        title = str(b.get("title_raw") or "").strip() or f"(article_id={h.article_id})"
        src = str(b.get("source") or "").strip()
        url = str(b.get("normalized_url") or "").strip()
        lines.append(f"### 来源 {i}")
        lines.append(f"- 标题：{title}")
        if src:
            lines.append(f"- 信源：{src}")
        if url:
            lines.append(f"- URL：{url}")
        lines.append(f"- article_id：{h.article_id}")
        lines.append("")
        body = (h.chunk_text or "").strip()
        if len(body) > 5200:
            body = body[:5200] + "…"
        lines.append(body)
        lines.append("")
    return "\n".join(lines)


def generate_research_report_markdown(
    question: str,
    hits: List[EvidenceHit],
    *,
    backend: Optional[OpenAICompatibleBackend] = None,
    model: Optional[str] = None,
    temperature: float = 0.45,
    timeout: float = 240.0,
) -> str:
    """
    输入：研究问题 + hybrid_retrieve 的 EvidenceHit 列表。
    输出：Markdown 正文；无证据时返回说明段，不发起 LLM。
    """
    q = (question or "").strip()
    if not q:
        return "（问题为空，请输入研究问题。）"
    if not hits:
        return (
            "## 深度调研报告\n\n"
            "**未检索到可用证据片段。** 可能原因：Chroma 文章向量库未建、MySQL 中尚无分块、"
            "或主域/信源筛选过严。可先同步新闻并完成向量化入库，再重试。\n"
        )
    ids = list({h.article_id for h in hits})
    briefs = get_articles_brief_by_ids(ids)
    packed = _pack_evidence(q, hits, briefs)
    be = backend or OpenAICompatibleBackend()
    m = model or LLM_MODEL
    user_msg = (
        "请**直接输出完整 Markdown 报告正文**（不要前言如「以下为报告」、不要用代码围栏包裹全文）。\n"
        "材料中已按「来源 1、来源 2…」编号；正文引用务必与编号一致。\n\n"
        + packed
    )
    raw = be.chat_completion(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        model=m,
        temperature=temperature,
        timeout=timeout,
    ).strip()
    return raw
