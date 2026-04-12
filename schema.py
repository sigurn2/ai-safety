"""
AI 治理事件的结构化模式定义。

本模块定义「意图与来源」三元风险模型：
- 三个固定主域（risk_domain）：便于统计与治理框架对齐；
- 动态子域（risk_subdomain）：由 LLM 从内容中归纳，写入 risk_taxonomy 表做自增长演进。
"""

from pydantic import BaseModel, Field
from typing import List, Literal

# ---------------------------------------------------------------------------
# 三元模型：三大主域（与 app.py 中 LLM 提示词、SQLite 中存储的字符串保持一致）
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
    """单条 AI 治理/安全相关情报的结构化表示。"""

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
    """LLM 批量提取时的根对象（对应 JSON 里的 incidents 数组）。"""

    incidents: List[AIIncident]
