"""Article full-text indexing: Chroma vectors + MySQL article_chunks metadata."""

from engine.article_index.indexer import index_article
from engine.article_index.retriever import ArticleChunkHit, get_article_collection, query_article_chunks

__all__ = [
    "index_article",
    "ArticleChunkHit",
    "get_article_collection",
    "query_article_chunks",
]
