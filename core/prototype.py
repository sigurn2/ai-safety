from datetime import datetime
from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field

# ==========================================
# 0. 爬取层 (Raw Layer)
# ==========================================
class RawArticle(BaseModel):
    """当前爬虫原始数据结构，直接从来源（如 Guardian API）获取的基础字段"""
    web_url: str = Field(..., description="原始文章链接")
    title: str = Field(..., description="原始文章标题 (如 Guardian 的 webTitle)")
    trail_text: Optional[str] = Field(None, description="原始导语或摘要")
    body_text: Optional[str] = Field(None, description="原始正文全文")
    web_publication_date: Optional[str] = Field(None, description="原始发布时间 (通常为 ISO 8601 字符串)")
    section_name: Optional[str] = Field(None, description="所属新闻版块")
    api_url: Optional[str] = Field(None, description="API 请求链接")
    guardian_id: Optional[str] = Field(None, description="文章在 Guardian 的唯一 ID")


# ==========================================
# 1. 结构化抽取层 (Base & Extraction Layer)
# ==========================================
class Article(BaseModel):
    """正样本文章主表 (articles)：保存命中 AI 治理/安全相关内容的文章原文"""
    id: Optional[int] = Field(None, description="文章主键，数据库自动生成")
    normalized_url: str = Field(..., description="归一化后的 URL，用于全局去重")
    source: str = Field(default="guardian", description="信源名称，当前固定为 guardian")
    title_raw: str = Field(..., description="原始标题")
    summary_raw: Optional[str] = Field(None, description="原始导语/摘要；若无则可截取正文前 512 字")
    content_raw: Optional[str] = Field(None, description="原始正文全文；若无则退化为 summary_raw")
    published_at: Optional[datetime] = Field(None, description="原文发布时间")
    content_hash: str = Field(..., description="基于(标题+摘要+正文)计算的哈希值，用于内容级去重")
    created_at: datetime = Field(default_factory=datetime.now, description="系统入库时间")
    rejected: bool = Field(True, description="是否和主题无关， 默认为True")
    rejected_reason: str = Field(...,description="被拒绝的原因和阶段")


# ==========================================
# 2. 信息抽取层， 用于incident tracking
# ==========================================
class ArticleExtraction(BaseModel):
    """结构化抽取表 (article_extractions)：保存 LLM 从正样本中抽取的 AI 治理/安全情报"""
    id: Optional[int] = Field(None, description="抽取记录主键")
    article_id: int = Field(..., description="关联的正样本文章主键 (articles.id)")
    model_name: str = Field(default="", description="执行抽取任务的 LLM 模型名称")
    content_type: str = Field(default="other", description="内容类型：news, meeting, report 等")
    main_topic: str = Field(default="", description="文章级核心主题")
    risk_domain: str = Field(default="", description="风险主域（意图与来源三元模型）")
    risk_subdomains_json: List[str] = Field(..., description="风险/治理子域列表，存入 MySQL 时需序列化为 JSON")
    entities_json: List[str] = Field(..., description="涉及主体（机构、公司等）列表，存入 MySQL 时需序列化为 JSON")
    summary_structured: str = Field(default="", description="LLM 生成的结构化摘要")
    tags_raw: List[str] = Field(..., description="原始标签列表，存入 MySQL 时需序列化为 JSON")
    created_at: datetime = Field(default_factory=datetime.now, description="抽取入库时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最后一次重抽或覆盖的更新时间")


# 
# 3. 生成层， 用于 report generation log system
# 
class ResearchReport(BaseModel):
    """研究报告表 (research_reports)：保存用户提问及生成的最终报告"""
    id: Optional[int] = Field(None, description="报告主键")
    question: str = Field(..., description="用户输入的研究问题")
    filters_json: Dict[str, Any] = Field(..., description="检索的过滤条件（如时间范围、风险域等），MySQL 存 JSON")
    related_articles: List[int] = Field(..., description="检索到的相关文章之id")
    report_markdown: str = Field(..., description="LLM 最终生成并返回的 Markdown 格式报告正文")
    model_name: str = Field(default="", description="生成该报告所使用的 LLM 模型名称")
    created_at: datetime = Field(default_factory=datetime.now, description="报告生成与入库时间")

