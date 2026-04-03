from pydantic import BaseModel, Field
from typing import List


class AIIncident(BaseModel):
    title: str = Field(..., description="事件或会议标题")
    entity: str = Field(..., description="涉及主体（如 OpenAI、欧盟、中国工信部）")
    risk_level: str = Field(..., description="风险等级：高 / 中 / 低")
    summary: str = Field(..., description="一句话核心内容摘要，不超过 60 字")
    tags: List[str] = Field(default_factory=list, description="关键词标签列表，3~8 个中英文词")


class ExtractionResult(BaseModel):
    incidents: List[AIIncident]
