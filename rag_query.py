import psycopg2
from sentence_transformers import SentenceTransformer

# Configuración base de datos
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "normas_db"
DB_USER = "vector_user"
DB_PASS = "vector_pass"

# Modelo de embeddings (384 dimensiones)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RAGRetriever:
    def __init__(self):
        print("Cargando modelo de embeddings...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("Estableciendo conexión a PGVector...")
        self.conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
        )

    def embed(self, text: str):
        """Genera el embedding de la pregunta."""
        return self.model.encode(text).tolist()

    def retrieve(self, query: str, top_k=5):
        """Devuelve los fragmentos más relevantes usando PGVector."""
        embedding = self.embed(query)

        # Convertimos la lista de floats al formato texto que entiende PGVector: [0.1,0.2,...]
        emb_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

        sql = """
            SELECT 
                norma,
                clausula,
                contenido,
                embedding <-> %s AS distancia
            FROM documentos
            ORDER BY embedding <-> %s
            LIMIT %s;
        """

        with self.conn.cursor() as cur:
            cur.execute(sql, (emb_str, emb_str, top_k))
            rows = cur.fetchall()

        results = []
        for norma, clausula, contenido, distancia in rows:
            results.append({
                "norma": norma,
                "clausula": clausula,
                "contenido": contenido,
                "distancia": distancia
            })

        return results

