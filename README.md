# caliRAG: Asistente conversacional RAG para normativas de calidad (ISO)

caliRAG es un proyecto académico que implementa un asistente conversacional especializado en normativas de calidad (ISO 9001, ISO 19011, ISO 9004, ISO/IEC 15504, ISO 10012, etc.) mediante una arquitectura de Generación Aumentada por Recuperación (RAG). El sistema se ejecuta íntegramente en entorno local e integra una base de datos vectorial, un backend de servicios, modelos LLM y una interfaz web interactiva.

## Arquitectura general
El sistema está compuesto por los siguientes bloques:
- PostgreSQL + pgvector: almacenamiento de fragmentos normativos y embeddings.
- Backend FastAPI: API REST que orquesta la recuperación, generación y carga de documentos.
- LMStudio: servidor local compatible con OpenAI API para ejecutar modelos LLM.
- Streamlit: interfaz web para chat conversacional y gestión de documentos.
- Módulo de evaluación: scripts para evaluar automáticamente la calidad de las respuestas.

## Requisitos
- Docker y Docker Compose
- Python 3.10+
- LMStudio (https://lmstudio.ai/)
- Git

## Puesta en marcha

### 1. Despliegue de PostgreSQL + pgvector (Docker)

Desde la raíz del proyecto:``docker compose up -d``

Esto levanta:
- PostgreSQL con la extensión pgvector
- La base de datos y usuario configurados para el sistema

Para comprobar el estado: ``docker ps``

Para acceder manualmente a la base de datos: ``psql -h localhost -U vector_user -d normas_db``

### 2. Inicialización de la base de datos

Ejecuta el script de inicialización: ``psql -h localhost -U vector_user -d normas_db -f init.sql``

Este script:
- Crea el esquema de datos
- Define la tabla documentos con columna vectorial
- Habilita la extensión pgvector

### 3. Configuración y arranque de LMStudio
1.	Abre LMStudio
2.	Descarga y carga uno de los modelos soportados:
    - Llama 3 8B Instruct
    - MiniMistral 3 8B Reasoning
3.	Inicia el servidor local con API OpenAI-compatible:
    - Endpoint: ``http://localhost:1234/v1/chat/completions``

El cambio de modelo (por ejemplo, Llama 3 → Llama 4) es transparente para la aplicación.

### 4. Backend FastAPI

1. Instala dependencias: ``pip install -r requirements.txt``

2. Arranca la API: ``uvicorn api2:app --reload --host 0.0.0.0 --port 8000``

3. Endpoints disponibles:
    - POST /query — consulta RAG de un turno
    - POST /chat — chat conversacional con historial + RAG
    - POST /upload_docs — carga e indexación de documentos PDF

Documentación interactiva: ``http://localhost:8000/docs``

### 5. Interfaz web (Streamlit)

Arranca la aplicación: ``streamlit run streamlit_app.py``

Funcionalidades:
- Chat conversacional con historial
- Subida de documentos PDF
- Indexación dinámica en la base vectorial
- Interacción directa con el backend FastAPI

## Gestión de la base vectorial (pgvector)

### 1. Inserción de documentos

Los documentos pueden añadirse de dos formas:
- Mediante el endpoint /upload_docs
- Mediante scripts de ingestión (ingest_segments.py)

Cada documento se:
1.	Trocea en fragmentos
2.	Convierte a embeddings
3.	Inserta en la tabla documentos

### 2.Consulta semántica
Las consultas RAG utilizan:
- Distancia coseno sobre embeddings
- Parámetro top_k configurable (por defecto 5 o 10)

## Evaluación del sistema

El proyecto incluye scripts de evaluación automática: ``python evaluate_rag_llm.py``

La evaluación:
- Reutiliza un conjunto fijo de preguntas normativas
- Compara configuraciones RAG (top_k = 5 vs top_k = 10)
- Utiliza LLMs como evaluadores
- Clasifica las respuestas en una escala ordinal:
    - Muy alta
    - Alta
    - Media
    - Baja
    - Muy baja

Los resultados se almacenan en formato JSON para su análisis posterior.


## Consideraciones importantes
- Todo el sistema se ejecuta en local, sin dependencias de servicios cloud.
- La estabilidad del modelo LLM es crítica:
    - Algunos modelos de razonamiento pueden generar salidas no controladas.
    - Se recomienda limitar max_tokens y usar stop sequences.
- El RAG actual se basa en similitud semántica; puede mejorarse incorporando:
    - detección explícita de normas citadas,
    - filtrado híbrido semántico + simbólico,
    - ponderación por norma.
