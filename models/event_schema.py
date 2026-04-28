"""
MySQL Phase 1 storage schemas used by this repo (MVP).

Defines normalized entities for:
- immutable article records
- one row per article in article_extractions
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Article(BaseModel):
    normalized_url: str = Field(..., description="Normalized URL, unique in storage.")
    source: str = Field(default="", description="Source system label.")
    title_raw: str = Field(..., description="Raw article title.")
    summary_raw: str = Field(default="", description="Raw summary or trail text.")
    content_raw: str = Field(default="", description="Raw article body.")
    published_at: Optional[datetime] = Field(default=None, description="Source publish time.")
    content_hash: str = Field(..., description="SHA-256 hash of canonicalized content.")


class ArticleExtraction(BaseModel):
    """Aligns with MySQL article_extractions (one row per article_id)."""

    article_id: int
    model_name: str = Field(default="", description="Model used for extraction.")
    content_type: str = Field(
        default="other",
        description="news | meeting | report | policy | opinion | other",
    )
    main_topic: str = Field(default="", description="One-line topic / title-like label.")
    risk_domain: str = Field(default="", description="Normalized risk domain.")
    risk_subdomains: List[str] = Field(default_factory=list, description="Subdomain labels (JSON in MySQL).")
    entities: List[str] = Field(default_factory=list, description="Entity mentions (JSON in MySQL).")
    summary_structured: str = Field(default="", description="Structured one-line summary.")
    tags_raw: List[str] = Field(default_factory=list, description="Raw extracted tags.")
