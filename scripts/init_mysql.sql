-- Phase 1 MySQL schema: articles, extractions, chunks, research reports (MVP).
-- MySQL 8.0+ required for JSON and utf8mb4 support.

CREATE DATABASE IF NOT EXISTS ai_governance
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE ai_governance;

CREATE TABLE IF NOT EXISTS articles (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  normalized_url VARCHAR(1024) NOT NULL,
  source VARCHAR(128) NOT NULL DEFAULT '',
  title_raw VARCHAR(1024) NOT NULL,
  summary_raw TEXT NULL,
  content_raw MEDIUMTEXT NULL,
  published_at DATETIME NULL,
  content_hash CHAR(64) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_articles_normalized_url (normalized_url(768)),
  KEY idx_articles_published_at (published_at),
  KEY idx_articles_source_time (source, published_at),
  KEY idx_articles_content_hash (content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;



CREATE TABLE IF NOT EXISTS unmatched_articles (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  normalized_url VARCHAR(1024) NOT NULL,
  source VARCHAR(128) NOT NULL DEFAULT '',
  title_raw VARCHAR(1024) NOT NULL DEFAULT '',
  summary_raw TEXT NULL,
  content_preview MEDIUMTEXT NULL,
  published_at DATETIME NULL,
  section_name VARCHAR(255) NOT NULL DEFAULT '',
  content_hash CHAR(64) NOT NULL DEFAULT '',
  reject_stage ENUM('fetch', 'llm', 'dedup', 'manual') NOT NULL DEFAULT 'llm',
  reject_reason VARCHAR(255) NOT NULL DEFAULT '',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_unmatched_normalized_url (normalized_url(768)),
  KEY idx_unmatched_source_time (source, published_at),
  KEY idx_unmatched_stage (reject_stage),
  KEY idx_unmatched_content_hash (content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


CREATE TABLE IF NOT EXISTS article_extractions (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  article_id BIGINT UNSIGNED NOT NULL,
  model_name VARCHAR(128) NOT NULL DEFAULT '',
  content_type VARCHAR(32) NOT NULL DEFAULT 'other',
  main_topic VARCHAR(512) NOT NULL DEFAULT '',
  risk_domain VARCHAR(128) NOT NULL DEFAULT '',
  risk_subdomains_json JSON NOT NULL,
  entities_json JSON NOT NULL,
  summary_structured VARCHAR(512) NOT NULL DEFAULT '',
  tags_raw JSON NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_extractions_article (article_id),
  KEY idx_extractions_domain (risk_domain),
  KEY idx_extractions_main_topic (main_topic(191)),
  CONSTRAINT fk_extractions_article
    FOREIGN KEY (article_id)
    REFERENCES articles (id)
    ON DELETE CASCADE
    ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


CREATE TABLE IF NOT EXISTS article_chunks (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  article_id BIGINT UNSIGNED NOT NULL,
  chunk_uid CHAR(64) NOT NULL,
  chunk_type ENUM('summary', 'body') NOT NULL DEFAULT 'body',
  chunk_index INT NOT NULL DEFAULT 0,
  chunk_text MEDIUMTEXT NOT NULL,
  token_estimate INT NOT NULL DEFAULT 0,
  embedding_model VARCHAR(128) NOT NULL DEFAULT '',
  vector_id CHAR(64) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_article_chunk_uid (chunk_uid),
  KEY idx_chunks_article_id (article_id),
  KEY idx_chunks_vector_id (vector_id),
  FULLTEXT KEY ft_chunk_text (chunk_text),
  CONSTRAINT fk_chunks_article
    FOREIGN KEY (article_id)
    REFERENCES articles (id)
    ON DELETE CASCADE
    ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS research_reports (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  question TEXT NOT NULL,
  filters_json JSON NOT NULL,
  report_markdown MEDIUMTEXT NOT NULL,
  model_name VARCHAR(128) NOT NULL DEFAULT '',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_reports_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS research_report_sources (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  report_id BIGINT UNSIGNED NOT NULL,
  article_id BIGINT UNSIGNED NOT NULL,
  chunk_id BIGINT UNSIGNED NULL,
  relevance_score DECIMAL(6, 5) NOT NULL DEFAULT 0.00000,
  citation_label VARCHAR(64) NOT NULL DEFAULT '',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_report_sources_report (report_id),
  KEY idx_report_sources_article (article_id),
  KEY idx_report_sources_chunk (chunk_id),
  CONSTRAINT fk_report_sources_report
    FOREIGN KEY (report_id)
    REFERENCES research_reports (id)
    ON DELETE CASCADE
    ON UPDATE RESTRICT,
  CONSTRAINT fk_report_sources_article
    FOREIGN KEY (article_id)
    REFERENCES articles (id)
    ON DELETE CASCADE
    ON UPDATE RESTRICT,
  CONSTRAINT fk_report_sources_chunk
    FOREIGN KEY (chunk_id)
    REFERENCES article_chunks (id)
    ON DELETE SET NULL
    ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

