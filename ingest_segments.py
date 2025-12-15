import os
import json
from glob import glob

import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIGURACIÓN DB ---
DB_HOST = "localhost"      # o el host de tu contenedor si no estás en la misma máquina
DB_PORT = 5432             # o 5433 si cambiaste el mapeo
DB_NAME = "normas_db"
DB_USER = "vector_user"
DB_PASS = "vector_pass"

# Carpeta donde tienes tus JSON segmentados
SEGMENTS_DIR = "segments"

# Modelo de embeddings (384 dim)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


def ensure_extension_and_table():
    """Por si no usas init.sql: crea extensión y tabla si no existen."""
    create_sql = """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS documentos (
        id SERIAL PRIMARY KEY,
        norma TEXT NOT NULL,
        clausula TEXT,
        contenido TEXT NOT NULL,
        embedding vector(384)
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
        conn.commit()


def load_segments(directory: str):
    """Carga todos los JSON de la carpeta de segmentos."""
    pattern = os.path.join(directory, "*.json")
    files = glob(pattern)
    files.sort()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Esperamos algo tipo:
        # { "norma": "ISO-9001", "clausula": "4.1.1", "texto": "..." }
        norma = data.get("norma")
        clausula = data.get("clausula")
        texto = data.get("texto")
        if not (norma and texto):
            continue
        yield norma, clausula, texto


def main():
    print("Cargando modelo de embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Conectando a la base de datos...")
    ensure_extension_and_table()

    segments = list(load_segments(SEGMENTS_DIR))
    print(f"Encontrados {len(segments)} segmentos en '{SEGMENTS_DIR}'")

    batch_size = 64

    with get_connection() as conn:
        with conn.cursor() as cur:
            for i in tqdm(range(0, len(segments), batch_size), desc="Insertando"):
                batch = segments[i : i + batch_size]
                textos = [s[2] for s in batch]
                # Embeddings -> lista de listas (vectores float)
                embeddings = model.encode(textos, show_progress_bar=False)

                rows = []
                for (norma, clausula, texto), emb in zip(batch, embeddings):
                    # psycopg2 acepta lista de floats para vector
                    rows.append((norma, clausula, texto, emb.tolist()))

                execute_batch(
                    cur,
                    """
                    INSERT INTO documentos (norma, clausula, contenido, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    rows,
                )
            conn.commit()

    print("✅ Ingestión completada.")


if __name__ == "__main__":
    main()