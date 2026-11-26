from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from rag_query import RAGRetriever
from rag_prompt import build_rag_prompt
import requests

# Configuración LM Studio
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

app = FastAPI(
    title="Chatbot Normativo ISO - RAG API",
    version="1.0.0",
    description="API para consultas normativas usando PGVector + LMStudio (RAG)."
)

class Query(BaseModel):
    pregunta: str
    top_k: int = 5

rag = RAGRetriever()

def ask_lmstudio(prompt: str, model: str = "meta-llama-3-8b-instruct") -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    r = requests.post(LMSTUDIO_URL, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


@app.post("/query")
def query_api(q: Query):
    # 1. Recuperar contexto RAG
    segmentos = rag.retrieve(q.pregunta, top_k=q.top_k)

    # 2. Generar prompt RAG
    prompt = build_rag_prompt(q.pregunta, segmentos)

    # 3. Pasar prompt a LM Studio
    respuesta = ask_lmstudio(prompt)

    return {
        "pregunta": q.pregunta,
        "prompt_usado": prompt,
        "segmentos": segmentos,
        "respuesta": respuesta
    }


if __name__ == "__main__":
    uvicorn.run("api_rag:app", host="0.0.0.0", port=8000)