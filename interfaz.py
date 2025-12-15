import streamlit as st
import requests
import os

# -------------------------------------------------------------------
# Configuración de la API backend
# -------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload_docs"

# -------------------------------------------------------------------
# Estado inicial del chat
# -------------------------------------------------------------------
if "messages" not in st.session_state:
    # Lista de mensajes estilo OpenAI/LMStudio: {role: "user"/"assistant"/"system", content: "..."}
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente conversacional especializado en normativas de calidad "
                "y normas ISO. Respondes de forma clara, técnica y estructurada."
            ),
        }
    ]

# -------------------------------------------------------------------
# Layout general
# -------------------------------------------------------------------
st.set_page_config(page_title="Chat Normativas de Calidad", page_icon="📚", layout="wide")

st.title("📚 Chat Normativas de Calidad (RAG + LMStudio)")

col_chat, col_tools = st.columns([2, 1])

# ==========================
# COLUMNA IZQUIERDA: CHAT
# ==========================
with col_chat:
    st.subheader("Chat")

    # Mostrar historial de mensajes
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            with st.expander("Mensaje de sistema", expanded=False):
                st.markdown(msg["content"])
        else:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])

    # Entrada de nuevo mensaje
    user_input = st.chat_input("Escribe tu mensaje sobre normativas ISO...")

    if user_input:
        # Añadir mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Construir payload para /chat
        payload = {
            "messages": st.session_state.messages,
            "top_k": 5,      # puedes exponerlo en la UI si quieres
            "use_rag": True  # RAG activado
        }

        try:
            resp = requests.post(CHAT_ENDPOINT, json=payload)
            resp.raise_for_status()
            data = resp.json()
            respuesta = data.get("respuesta", "")

            # Añadir respuesta del asistente al historial
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

            # Mostrar respuesta inmediatamente
            with st.chat_message("assistant"):
                st.markdown(respuesta)

        except requests.RequestException as e:
            st.error(f"Error al llamar a la API de chat: {e}")

# ==========================
# COLUMNA DERECHA: TOOLS
# ==========================
with col_tools:
    st.subheader("📄 Subir documento")

    st.markdown(
        "Sube un PDF para que sus contenidos se indexen en la base vectorial "
        "(PostgreSQL + pgvector) y puedan usarse en el RAG."
    )

    uploaded_file = st.file_uploader("Selecciona un PDF", type=["pdf"])
    norma_label = st.text_input("Etiqueta para 'norma' (opcional)", value="DOCUMENTO_STREAMLIT")

    if uploaded_file is not None:
        if st.button("Subir e indexar PDF"):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
                }
                data = {
                    "norma": norma_label or uploaded_file.name,
                }
                resp = requests.post(UPLOAD_ENDPOINT, files=files, data=data)
                resp.raise_for_status()
                info = resp.json()
                st.success(
                    f"PDF procesado correctamente.\n\n"
                    f"Norma: {info.get('norma')}\n"
                    f"Segmentos creados: {info.get('segmentos_creados')}"
                )
            except requests.RequestException as e:
                st.error(f"Error al subir el PDF: {e}")

    st.markdown("---")
    st.subheader("⚙️ Opciones de sesión")

    if st.button("Reiniciar chat"):
        st.session_state.messages = [
            st.session_state.messages[0]  # mantiene sólo el mensaje de sistema
        ]
        st.experimental_rerun()

    st.caption(
        "Backend: FastAPI en http://localhost:8000\n\n"
        "Modelo LLM: LMStudio en http://localhost:1234/v1/chat/completions.\n"
        "Base vectorial: PostgreSQL + pgvector (tabla `documentos`)."
    )