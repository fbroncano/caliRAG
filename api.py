from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Literal
import os
import io
import requests

import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# --------------------------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------------------------

# LMStudio: servidor OpenAI-compatible en modo chat
LMSTUDIO_URL = os.getenv(
    "LMSTUDIO_URL",
    "http://localhost:1234/v1/chat/completions"
)

# PostgreSQL + PGVector
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = int(os.getenv("PG_PORT", "5432"))
DB_NAME = os.getenv("PG_DB", "normas_db")
DB_USER = os.getenv("PG_USER", "vector_user")
DB_PASS = os.getenv("PG_PASS", "vector_pass")

# Modelo de embeddings (384 dimensiones, coherente con vector(384) en BD)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Importa tu lógica de RAG ya implementada
from rag_query import RAGRetriever
from rag_prompt import build_rag_prompt

# --------------------------------------------------------------------
# INICIALIZACIÓN
# --------------------------------------------------------------------

app = FastAPI(
    title="API RAG Normativas de Calidad",
    description="FastAPI para RAG sobre PGVector + LMStudio (Llama 3/4).",
    version="1.1.0",
)

rag_retriever = RAGRetriever()
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --------------------------------------------------------------------
# MODELOS Pydantic
# --------------------------------------------------------------------

class QueryRequest(BaseModel):
    pregunta: str
    top_k: int = 10


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: int = 10
    use_rag: bool = True


class Segment(BaseModel):
    norma: str
    clausula: Optional[str] = None
    contenido: str


# --------------------------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


def call_lmstudio_chat(messages: List[dict], reasoner: bool=False) -> str:
    """
    Llama a LMStudio (chat completions).
    Transparente al modelo (Llama 3 / Llama 4) mientras el endpoint sea el mismo.
    """

    payload = {
        "model": "mistralai/ministral-3-8b-reasoning" if reasoner else "meta-llama-3-8b-instruct",  # LMStudio lo ignora, pero es obligatorio
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    try:
        r = requests.post(LMSTUDIO_URL, json=payload)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error llamando a LMStudio: {e}")

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Respuesta inesperada de LMStudio")


def insert_segments(segments: List[Segment]):
    """
    Inserta segmentos en la tabla `documentos`,
    generando embeddings con el mismo modelo que el pipeline previo.
    """
    if not segments:
        return

    textos = [s.contenido for s in segments]
    vectors = embedding_model.encode(textos, show_progress_bar=False)

    rows = []
    for seg, emb in zip(segments, vectors):
        rows.append((
            seg.norma,
            seg.clausula,
            seg.contenido,
            emb.tolist(),
        ))

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_batch(
                cur,
                """
                INSERT INTO documentos (norma, clausula, contenido, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """
    Trocea texto largo en segmentos con solapamiento simple.
    """
    text = text.replace("\r", " ")
    text = " ".join(text.split())  # normalizar espacios

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]

        # evitar cortar palabras de forma fea
        if end < n:
            last_space = chunk.rfind(" ")
            if last_space != -1 and last_space > max_chars * 0.6:
                chunk = chunk[:last_space]
                end = start + last_space

        chunks.append(chunk)
        start = max(0, end - overlap)

        if end == n:
            break

    return chunks


def extract_segments_from_pdf(
    file_bytes: bytes,
    norma_label: str
) -> List[Segment]:
    """
    Extrae texto de un PDF, lo trocea en segmentos y los devuelve como Segment[].
    `norma_label` se usa para rellenar el campo `norma` (por ejemplo, nombre del fichero).
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text() or ""
        full_text += page_text + "\n"

    chunks = chunk_text(full_text, max_chars=1000, overlap=200)

    segments: List[Segment] = []
    for i, ch in enumerate(chunks, start=1):
        segments.append(
            Segment(
                norma=norma_label,
                clausula=f"pdf_chunk_{i}",
                contenido=ch,
            )
        )
    return segments


# --------------------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "mensaje": "API RAG Normativas de Calidad",
        "endpoints": ["/query", "/chat", "/upload_docs"],
    }


@app.post("/query")
def query_rag(q: QueryRequest):
    """
    /query → RAG de un turno:
    - Recupera contexto en PGVector.
    - Construye el prompt RAG.
    - Llama a LMStudio (Llama 3/4).
    """
    # 1. Recuperar segmentos relevantes con RAG
    segmentos = rag_retriever.retrieve(q.pregunta, top_k=q.top_k)

    # 2. Construir prompt RAG
    prompt = build_rag_prompt(q.pregunta, segmentos)

    # 3. Llamar a LMStudio como una única interacción
    mensajes = [
        {"role": "user", "content": prompt}
    ]
    respuesta = call_lmstudio_chat(mensajes, reasoner=False)

    return {
        "pregunta": q.pregunta,
        "segmentos": segmentos,
        "prompt_usado": prompt,
        "respuesta": respuesta,
    }


@app.post("/chat")
def chat_llama(req: ChatRequest):
    """
    /chat → chat con historial.
    - Recibe una lista de mensajes tipo chat (system/user/assistant).
    - Toma el ÚLTIMO mensaje de usuario.
    - Si use_rag=True:
        * Hace RAG sobre ese último mensaje.
        * Construye un nuevo prompt RAG.
        * Sustituye el contenido del último mensaje de usuario por ese prompt.
    - Envía TODO el historial (con el último mensaje transformado) a LMStudio.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="No se han enviado mensajes.")

    # localizar el último mensaje de usuario
    last_user_idx = None
    for idx in range(len(req.messages) - 1, -1, -1):
        if req.messages[idx].role == "user":
            last_user_idx = idx
            break

    if last_user_idx is None:
        raise HTTPException(status_code=400, detail="No hay mensajes de usuario en el historial.")

    messages = [m.dict() for m in req.messages]

    if req.use_rag:
        last_user_msg = req.messages[last_user_idx].content

        # 1. RAG con el último mensaje de usuario
        segmentos = rag_retriever.retrieve(last_user_msg, top_k=req.top_k)

        # 2. Construir prompt RAG
        prompt_rag = build_rag_prompt(last_user_msg, segmentos)

        # 3. Sustituir contenido del último mensaje de usuario por el prompt RAG
        messages[last_user_idx]["content"] = prompt_rag
    else:
        segmentos = []

    # 4. Llamar a LMStudio con el historial completo
    respuesta = call_lmstudio_chat(messages, reasoner=False)

    return {
        "messages_enviados": messages,
        "segmentos_usados": segmentos,
        "respuesta": respuesta,
    }


@app.post("/upload_docs")
async def upload_docs(
    file: UploadFile = File(...),
    norma: Optional[str] = Form(None),
):
    """
    /upload_docs → acepta un PDF, lo trocea y lo sube a la BD.
    - file: PDF subido vía multipart/form-data.
    - norma: etiqueta opcional para el campo `norma` (por defecto, nombre del fichero).
    """
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Sólo se aceptan PDFs.")

    file_bytes = await file.read()
    norma_label = norma or file.filename or "DOCUMENTO_SIN_NOMBRE"

    try:
        segments = extract_segments_from_pdf(file_bytes, norma_label)
        insert_segments(segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el PDF: {e}")

    return {
        "norma": norma_label,
        "segmentos_creados": len(segments),
        "detalle": f"PDF {norma_label}  procesado y segmentos insertados en la base vectorial.",
    }