-- Optional: add FULLTEXT on article_chunks.chunk_text for sparse (BOOLEAN MODE) retrieval.
-- Skip if you already created the DB from scripts/init_mysql.sql after FULLTEXT was added to that file.
-- Run once, e.g.: mysql -u USER -p ai_governance < scripts/migrate_add_chunks_fulltext.sql

USE ai_governance;

ALTER TABLE article_chunks
  ADD FULLTEXT INDEX ft_chunk_text (chunk_text);
