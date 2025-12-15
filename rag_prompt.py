from textwrap import dedent
from rag_query import RAGRetriever  # si lo pones en otro fichero, ajusta el import


# Prompt base (resumido) del asistente normativo ISO
BASE_SYSTEM_PROMPT = dedent("""
Eres un asistente técnico especializado en sistemas de gestión de la calidad y en normas ISO.

Tu propósito es ofrecer respuestas precisas, verificables y normativamente fundamentadas
basadas exclusivamente en el marco ISO autorizado (normas de la familia ISO 9000, ISO 10000,
ISO 19011, ISO/IEC 12207, ISO/IEC 15288, ISO/IEC 15504, ISO/IEC 25000, etc.).

Debes actuar con el rigor de un auditor o consultor técnico experto. Tu comunicación debe
ser formal, clara y orientada a la correcta interpretación de requisitos y principios normativos.
No inventes cláusulas ni números de norma.
""").strip()


def build_rag_prompt(question: str, context_segments: list[dict], max_chars_per_segment: int = 900) -> str:
    """
    Construye un prompt RAG en texto plano:
    - Incluye instrucciones del sistema
    - Añade fragmentos normativos recuperados
    - Formula la pregunta del usuario
    - Refuerza la estructura de respuesta
    """

    lines: list[str] = []

    # 1) Instrucciones base del sistema
    lines.append("### INSTRUCCIONES DEL SISTEMA")
    lines.append(BASE_SYSTEM_PROMPT)

    # 2) Contexto normativo recuperado
    lines.append("\n### CONTEXTO NORMATIVO RECUPERADO")
    lines.append("Los siguientes fragmentos proceden de normas ISO y deben considerarse como única fuente válida:")

    if not context_segments:
        lines.append("\n(No se ha recuperado contexto normativo relevante.)")
    else:
        for i, seg in enumerate(context_segments, start=1):
            norma = seg.get("norma", "N/D")
            clausula = seg.get("clausula") or "N/D"
            texto = seg.get("contenido", "").replace("\n", " ")

            if len(texto) > max_chars_per_segment:
                texto = texto[:max_chars_per_segment] + "..."

            lines.append(f"\n[Fragmento {i}] Norma: {norma} | Cláusula: {clausula}")
            lines.append(texto)

    # 3) Pregunta del usuario
    lines.append("\n### CONSULTA DEL USUARIO")
    lines.append(question.strip())

    # 4) Instrucciones de salida (estructura de respuesta)
    lines.append("\n### INSTRUCCIONES PARA LA RESPUESTA")
    lines.append(dedent("""
    Responde únicamente utilizando la información contenida en los fragmentos recuperados.
    Si la información no aparece en ellos, indica explícitamente:
    "No existe referencia normativa directa en los fragmentos proporcionados".

    Estructura SIEMPRE la respuesta en los siguientes apartados numerados:

    1. Resumen normativo breve
       - Expón la idea principal de la norma o cláusula relevante.

    2. Explicación técnica
       - Desarrolla la interpretación práctica dentro del marco ISO.

    3. Citas normativas
       - Lista las normas y cláusulas exactas utilizadas.
         Ejemplo:
         - ISO 9001:2015 — 7.1.5.2 “Trazabilidad de las mediciones”
         - ISO 19011:2018 — 7.2 “Competencia de los auditores”

    4. Observaciones o notas prácticas (opcional)
       - Solo si aportan contexto útil habitual en auditoría o consultoría.

    No incluyas información ajena a las normas ni inventes numeraciones.
    """).strip())

    return "\n".join(lines)


if __name__ == "__main__":
    # Ejemplo de uso integrando RAGRetriever + build_rag_prompt
    rag = RAGRetriever()

    pregunta = "¿Qué exige ISO 9001 respecto a la trazabilidad metrológica y la calibración de los equipos de medición?"
    segmentos = rag.retrieve(pregunta, top_k=5)

    prompt = build_rag_prompt(pregunta, segmentos)

    print("========== PROMPT GENERADO ==========\n")
    print(prompt)

