"""
AI 治理事件的结构化模式定义。

本模块定义「意图与来源」三元风险模型与**文章级** LLM 抽取载荷（MVP：一篇文章一条 extraction，多条细粒度情报后续用独立表）。
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 三元模型：三大主域（与 LLM 提示词、SQLite 中存储的字符串保持一致）
# 三条的详细释义见 crawler.extraction.RISK_DOMAIN_LLM_GUIDANCE（与 herein 字符串须一致）
# ---------------------------------------------------------------------------
RISK_DOMAIN_CHOICES = (
    "Malicious Use (恶意滥用)",
    "Accidental Failure (意外失效)",
    "Systemic & Ethical Risk (系统性与伦理风险)",
)

RiskDomainLiteral = Literal[
    "Malicious Use (恶意滥用)",
    "Accidental Failure (意外失效)",
    "Systemic & Ethical Risk (系统性与伦理风险)",
]

# 与 MySQL article_extractions.content_type 及 prompt 一致（勿与 incident 混在同表）
ContentTypeLiteral = Literal["news", "meeting", "report", "policy", "opinion", "other"]


class AIIncident(BaseModel):
    """
    本地 SQLite「一条展示/统计行」的契约（由文章级抽取派生一条，非多 incident 主存储）。
    """

    model_config = ConfigDict(extra="ignore")

    title: str = Field(..., description="事件或会议标题")
    entity: str = Field(..., description="涉及主体（如 OpenAI、欧盟、中国工信部）")
    risk_level: str = Field(..., description="风险等级：高 / 中 / 低")
    risk_domain: RiskDomainLiteral = Field(
        default="Systemic & Ethical Risk (系统性与伦理风险)",
        description="风险来源主域（三选一，字符串与 RISK_DOMAIN_CHOICES 完全一致）",
    )
    risk_subdomain: str = Field(
        default="未指定子域",
        description="该主域下的具体风险子类型",
    )
    summary: str = Field(..., description="一句话核心内容摘要，不超过 60 字")
    tags: List[str] = Field(default_factory=list, description="关键词标签列表，3~8 个中英文词")


class ArticleExtractionPayload(BaseModel):
    """
    每篇文章**一个** JSON 对象（Crawl4AI / extraction 共用 schema）。

    落库 MySQL `article_extractions` 的字段映射：
    content_type, main_topic, risk_domain, risk_subdomains→JSON, entities→JSON,
    summary_structured, tags→JSON。`model_name` 由服务端写入，勿出现在 JSON 根上。

    仅流程控制、不落 extraction 表：``is_relevant``、``reject_reason``、``relevance_reason``。
    """

    model_config = ConfigDict(extra="ignore")

    is_relevant: bool = Field(description="是否与 AI 治理/安全/政策/监管实质相关")
    reject_reason: str = Field(
        default="",
        max_length=255,
        description="不相关时的原因代码或短语，如 no_ai_governance_content",
    )
    content_type: ContentTypeLiteral = Field(
        default="other",
        description="与 article_extractions.content_type 一致：news/meeting/report/policy/opinion/other",
    )
    main_topic: str = Field(
        default="",
        max_length=512,
        description="一句话主旨；法案/会议名等政策线索一并写进本条（对应表字段 main_topic）",
    )
    risk_domain: RiskDomainLiteral = Field(
        default="Systemic & Ethical Risk (系统性与伦理风险)",
        description="三元主域之一，须与枚举字符串完全一致（对应 risk_domain）",
    )
    risk_subdomains: List[str] = Field(
        default_factory=list,
        description="治理/风险议题子标签（落库 risk_subdomains_json）",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="主要机构、公司、政府、人物（落库 entities_json）",
    )
    summary_structured: str = Field(
        default="",
        max_length=512,
        description="一句话摘要（落库 summary_structured）",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="检索关键词（落库 tags_raw），建议 3–8 个",
    )
    relevance_reason: str = Field(
        default="",
        max_length=512,
        description="为何判定相关；仅供调试，不落 article_extractions",
    )


# Crawl4AI 旧名兼容
ExtractionResult = ArticleExtractionPayload
