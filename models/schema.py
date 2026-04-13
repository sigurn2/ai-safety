"""
AI 治理事件的结构化模式定义。

本模块定义「意图与来源」三元风险模型：
- 三个固定主域（risk_domain）：便于统计与治理框架对齐；
- 动态子域（risk_subdomain）：由 LLM 从内容中归纳，写入 risk_taxonomy 表做自增长演进。
"""

from pydantic import BaseModel, Field
from typing import List, Literal

# ---------------------------------------------------------------------------
# 三元模型：三大主域（与 LLM 提示词、SQLite 中存储的字符串保持一致）
# ---------------------------------------------------------------------------
RISK_DOMAIN_CHOICES = (
    "Malicious Use (恶意滥用)",
    "Accidental Failure (意外失效)",
    "Systemic & Ethical Risk (系统性与伦理风险)",
)

# 用于 Pydantic / JSON Schema，约束 LLM 输出枚举值
RiskDomainLiteral = Literal[
    "Malicious Use (恶意滥用)",
    "Accidental Failure (意外失效)",
    "Systemic & Ethical Risk (系统性与伦理风险)",
]


class AIIncident(BaseModel):
    """
    单条 AI 治理/安全情报（入库与 API 的契约形状）。

    功能：统一 title/主体/三元风险/摘要/标签字段，供 Pydantic 校验与 SQLite 落库。
    输入：通常来自 LLM 抽取 dict，经 core.db.incident_from_extraction 转换。
    输出：不可变业务对象；写库由 save_incident 完成。
    上下游：crawl 抽取 →（可选 RAG）→ incident_from_extraction → incidents 表。
    """

    title: str = Field(..., description="事件或会议标题")
    entity: str = Field(..., description="涉及主体（如 OpenAI、欧盟、中国工信部）")
    risk_level: str = Field(..., description="风险等级：高 / 中 / 低")
    # 固定主域：三元模型之一（LLM 必须择一；解析层会对近似表述做归一）
    risk_domain: RiskDomainLiteral = Field(
        default="Systemic & Ethical Risk (系统性与伦理风险)",
        description="风险来源主域：恶意滥用 / 意外失效 / 系统性与伦理风险（三选一，字符串与 RISK_DOMAIN_CHOICES 完全一致）",
    )
    # 动态子域：如「越狱攻击」「严重幻觉」「算法偏见」等，可随语料自动扩充
    risk_subdomain: str = Field(
        default="未指定子域",
        description="该主域下的具体风险子类型，简短专业术语，便于后续聚类与图谱扩展",
    )
    summary: str = Field(..., description="一句话核心内容摘要，不超过 60 字")
    tags: List[str] = Field(default_factory=list, description="关键词标签列表，3~8 个中英文词")


class ExtractionResult(BaseModel):
    """
    Crawl4AI LLMExtractionStrategy 的根 JSON 结构（{"incidents": [...]}）。

    功能：为抽取策略提供 model_json_schema，约束 incidents 数组元素形状。
    输入：由 crawl4ai 根据网页内容调用 LLM 生成。
    输出：校验后的对象树；下游解析兼容 list-only 等畸形输出在 crawler 内处理。
    """

    incidents: List[AIIncident]
