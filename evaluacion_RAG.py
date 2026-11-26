import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

from rag_query import RAGRetriever
from rag_prompt import build_rag_prompt
from api_rag import LMSTUDIO_URL

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

SALIDA = "evaluacion_rag_cruzada.json"
MAX_TOKENS = 6000      # evita bucles de generación sin fin
TIMEOUT_S = 900        # tiempo máximo por llamada (15 min)
SIN_RESPUESTA = "Sin respuesta"
SIN_VALORACION = "Sin valoración"


def llamar_llm(prompt: str, model: str) -> Optional[str]:
    """
    Llama a LM Studio con límite de tokens y de tiempo.
    Devuelve None si la generación no termina correctamente
    (límite de tokens alcanzado, timeout o error del servidor).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    try:
        r = requests.post(LMSTUDIO_URL, json=payload, timeout=TIMEOUT_S)
        r.raise_for_status()
        choice = r.json()["choices"][0]
    except (requests.RequestException, KeyError, IndexError, ValueError) as e:
        print(f"   [fallo de generación en {model}: {e}]")
        return None
    if choice.get("finish_reason") == "length":
        print(f"   [{model} alcanzó el límite de {MAX_TOKENS} tokens]")
        return None
    return choice["message"]["content"]


def quitar_razonamiento(text: str) -> str:
    """Elimina el bloque de razonamiento [THINK]...[/THINK] de los modelos reasoning."""
    if "[/THINK]" in text:
        text = text.split("[/THINK]")[-1]
    return text.strip()


def _extract_label(text: str) -> str:
    """
    Extrae una etiqueta válida desde la salida del LLM (por si no devuelve JSON limpio).
    """
    for lab in ETIQUETAS:
        if re.search(rf"\b{re.escape(lab)}\b", text, flags=re.IGNORECASE):
            # normaliza capitalización exacta
            return next(x for x in ETIQUETAS if x.lower() == lab.lower())
    return SIN_VALORACION  # no se ha podido extraer una etiqueta válida


def judge_with_llm(pregunta: str, respuesta: str, model: str) -> Dict[str, Any]:
    """
    Pide al LLM que evalúe la respuesta y devuelva:
      - label: Muy alta|Alta|Media|Baja|Muy baja
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

    raw = llamar_llm(judge_prompt, model)
    if raw is None:
        return {"label": SIN_VALORACION, "reason": "", "raw": None}

    # Solo se analiza la salida final, sin el razonamiento interno
    final = quitar_razonamiento(raw)

    # Intenta parsear el último objeto JSON de la salida final
    bloques = re.findall(r"\{[^{}]*\}", final, flags=re.DOTALL)
    for bloque in reversed(bloques):
        try:
            data = json.loads(bloque)
        except ValueError:
            continue
        label = _extract_label(str(data.get("label", "")))
        reason = str(data.get("reason", "")).strip() or final[:300]
        return {"label": label, "reason": reason, "raw": raw}

    # fallback: extraer etiqueta de la salida final
    return {"label": _extract_label(final), "reason": final[:300], "raw": raw}


def guardar(resultados: List[Dict[str, Any]]) -> None:
    tmp = SALIDA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SALIDA)


def main():
    rag = RAGRetriever()

    # Reanudación: se cargan los resultados ya guardados
    resultados: List[Dict[str, Any]] = []
    if os.path.exists(SALIDA):
        with open(SALIDA, encoding="utf-8") as f:
            resultados = json.load(f)
        print(f"Reanudando: {len(resultados)} evaluaciones ya guardadas")

    def buscar(pregunta, k, evaluado, evaluador=None):
        for r in resultados:
            if (r["pregunta"], r["top_k"], r["evaluado"]) == (pregunta, k, evaluado) and \
               (evaluador is None or r["evaluador"] == evaluador):
                return r
        return None

    for pregunta in PREGUNTAS:
        print(f"\n=== Pregunta ===\n{pregunta}")

        for k in KS:
            # 1) Recuperación RAG (común a todos los modelos para esta pregunta y k)
            segmentos = rag.retrieve(pregunta, top_k=k)

            # 2) Prompt RAG
            prompt = build_rag_prompt(pregunta, segmentos)

            for evaluado in MODELOS:
                # 3) Respuesta generada por el modelo evaluado (o la ya guardada)
                previo = buscar(pregunta, k, evaluado)
                if previo is not None:
                    respuesta_raw = previo["respuesta_raw"]
                else:
                    print(f"-> Generando respuesta con {evaluado} (top_k={k})")
                    respuesta_raw = llamar_llm(prompt, evaluado)
                respuesta = quitar_razonamiento(respuesta_raw) if respuesta_raw else None

                # 4) La misma respuesta la valoran ambos modelos
                #    (evaluador == evaluado: autoevaluación; distinto: evaluación cruzada)
                for evaluador in MODELOS:
                    if buscar(pregunta, k, evaluado, evaluador) is not None:
                        continue

                    if respuesta is None:
                        eval_llm = {"label": SIN_RESPUESTA, "reason": "", "raw": None}
                    else:
                        eval_llm = judge_with_llm(pregunta, respuesta, evaluador)

                    resultados.append({
                        "pregunta": pregunta,
                        "evaluado": evaluado,
                        "evaluador": evaluador,
                        "top_k": k,
                        "respuesta": respuesta,
                        "respuesta_raw": respuesta_raw,
                        "segmentos": segmentos,
                        "evaluacion_llm": {
                            "label": eval_llm["label"],
                            "reason": eval_llm["reason"],
                            "raw": eval_llm["raw"],
                        }
                    })
                    guardar(resultados)

                    print(f"   {evaluado} evaluado por {evaluador}: {eval_llm['label']}")

    print(f"\nEvaluación completada. Resultados en {SALIDA}")


if __name__ == "__main__":
    main()
