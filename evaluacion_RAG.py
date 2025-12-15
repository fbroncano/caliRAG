import json
import re
from typing import Any, Dict, List

from rag_query import RAGRetriever
from rag_prompt import build_rag_prompt
from api_rag import ask_lmstudio

# Preguntas del Hito 2
PREGUNTAS = [
    "¿Qué debe incluir un sistema de gestión de la calidad según ISO 9001?",
    "¿Qué criterios establece ISO 19011 para la competencia de los auditores?",
    "¿Cómo abordan ISO 9004 e ISO/IEC 15504 la mejora continua?",
    "¿Qué ocurre si los equipos de medición no tienen calibración vigente según ISO 10012?",
    "¿Cómo debe tratarse la falta de seguimiento de auditorías anteriores según ISO 9001 e ISO 19011?"
]


MODELOS = ["meta-llama-3-8b-instruct", "mistralai/ministral-3-8b-reasoning"]
KS = [5, 10]
ETIQUETAS = ["Muy alta", "Alta", "Media", "Baja", "Muy baja"]


def _extract_label(text: str) -> str:
    """
    Extrae una etiqueta válida desde la salida del LLM (por si no devuelve JSON limpio).
    """
    for lab in ETIQUETAS:
        if re.search(rf"\b{re.escape(lab)}\b", text, flags=re.IGNORECASE):
            # normaliza capitalización exacta
            return next(x for x in ETIQUETAS if x.lower() == lab.lower())
    return "Medio"  # fallback conservador


def judge_with_llm(pregunta: str, respuesta: str, model: str) -> Dict[str, Any]:
    """
    Pide al LLM que evalúe la respuesta y devuelva:
      - label: Muy bueno|Bueno|Medio|Mal|Muy mal
      - reason: breve justificación (1-2 frases)
    """
    judge_prompt = f"""
Eres un evaluador de calidad de respuestas sobre normativas ISO.

TAREA:
Valora la respuesta del asistente a la pregunta del usuario.

CRITERIOS:
- Exactitud normativa (no inventar requisitos, no afirmar sin base).
- Claridad y estructura.
- Utilidad práctica.
- Coherencia con la pregunta.

SALIDA OBLIGATORIA (JSON válido, sin texto extra):
{{
  "label": "Muy Alta|Alta|Media|Baja|Muy baja",
  "reason": "1 frase de 50 caracteres de máximo."
}}

PREGUNTA:
{pregunta}

RESPUESTA:
{respuesta}
""".strip()

    raw = ask_lmstudio(judge_prompt, model)

    # Intenta parsear JSON
    try:
        data = json.loads(raw)
        label = data.get("label", "").strip()
        reason = data.get("reason", "").strip()
        if label not in ETIQUETAS:
            label = _extract_label(raw)
        if not reason:
            reason = raw.strip()[:300]
        return {"label": label, "reason": reason, "raw": raw}
    except Exception:
        # fallback: extraer etiqueta a mano
        return {"label": _extract_label(raw), "reason": raw.strip()[:300], "raw": raw}


def main():
    rag = RAGRetriever()
    resultados: List[Dict[str, Any]] = []

    for pregunta in PREGUNTAS:
        print(f"\n=== Pregunta ===\n{pregunta}")

        for k in KS:
            for evaluado in MODELOS:
                for evaluador in MODELOS:

                    print(f"-> Ejecutando RAG con top_k={k} sobre {evaluado} y evaluado por {evaluador}")

                    # 1) Recuperación RAG
                    segmentos = rag.retrieve(pregunta, top_k=k)

                    # 2) Prompt RAG
                    prompt = build_rag_prompt(pregunta, segmentos)

                    # 3) Respuesta del LLM
                    respuesta = ask_lmstudio(prompt, evaluador)

                    # 4) Evaluación por el propio LLM (self-judge)
                    eval_llm = judge_with_llm(pregunta, respuesta, evaluador)

                    resultados.append({
                        "pregunta": pregunta,
                        "evaluado": evaluado,
                        "evaluador": evaluador,
                        "top_k": k,
                        "respuesta": respuesta,
                        "segmentos": segmentos,
                        "evaluacion_llm": {
                            "label": eval_llm["label"],
                            "reason": eval_llm["reason"],
                            "raw": eval_llm["raw"],
                        }
                    })

                    print(f"   Evaluación: {eval_llm['label']}")

    with open("evaluacion_rag_llm.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\nEvaluación completada. Resultados en evaluacion_rag_llm.json")


if __name__ == "__main__":
    main()
