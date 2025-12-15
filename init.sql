CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS documentos;

CREATE TABLE documentos (
    id SERIAL PRIMARY KEY,
    norma TEXT NOT NULL,
    clausula TEXT,
    contenido TEXT NOT NULL,
    embedding vector(384)  -- ajustado al modelo de embeddings
);

-- Índice vectorial usando cosine
CREATE INDEX documentos_embedding_idx
ON documentos
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);