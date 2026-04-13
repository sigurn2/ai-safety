"""
信源适配器包：将第三方 API / RSS 等统一为 RawArticle 等形状。

功能：导出卫报 Content API 类型与入口，供编排器与脚本使用。
输入：无（包级说明）。
输出：见 __all__。
上下游：crawler.orchestrator、scripts；下游 LLM 抽取见 crawler.extraction（待接）。
"""

from crawler.sources.guardian import (
    DEFAULT_AI_GOVERNANCE_QUERY,
    GuardianAPIError,
    GuardianSearchPage,
    RawArticle,
    map_result_to_raw_article,
    raw_article_to_llm_context,
    search_articles,
    search_articles_multipage,
)

__all__ = [
    "DEFAULT_AI_GOVERNANCE_QUERY",
    "GuardianAPIError",
    "GuardianSearchPage",
    "RawArticle",
    "map_result_to_raw_article",
    "raw_article_to_llm_context",
    "search_articles",
    "search_articles_multipage",
]
