# interface/app_streamlit.py

import streamlit as st
from pathlib import Path
import json
import faiss
import numpy as np
from anthropic import Anthropic
import os

import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Agrega la raíz del proyecto al path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from llm_integration.rag_prompt import build_rag_prompt

# ----------------------------
# Configuración Claude
# ----------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
MAX_TOKENS = 4000
TEMPERATURE = 0.1

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ----------------------------
# Rutas vector DB
# ----------------------------
VECTOR_DB_DIR = Path(__file__).resolve().parents[1] / "embeddings" / "vector_db"
INDEX_PATH = VECTOR_DB_DIR / "index.faiss"
META_PATH = VECTOR_DB_DIR / "metadata.json"

# ----------------------------
# Cargar FAISS + metadata
# ----------------------------
@st.cache_resource
def load_vector_db():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        st.error("No se encontró index.faiss o metadata.json — genera embeddings primero.")
        st.stop()

    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# ----------------------------
# Generar embedding del query
# ----------------------------
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(query: str):
    vec = model.encode(query)                  # Genera embedding
    vec = vec / np.linalg.norm(vec)           # Normaliza
    return vec.reshape(1, -1)                 # Forma (1, dim)


# ----------------------------
# Búsqueda vectorial
# ----------------------------
def search_chunks(query: str, top_k: int = 4):
    index, metadata = load_vector_db()
    q_emb = embed_query(query)
    distances, indices = index.search(q_emb, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "id": idx,
            "text": metadata[idx]["text"],
            "source": metadata[idx]["source"],
            "distance": float(dist)
        })
    return results

# ----------------------------
# Generar respuesta con Claude
# ----------------------------
def generate_answer(query: str, retrieved_chunks: list):
    prompt = build_rag_prompt(query, retrieved_chunks)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    return response.content[0].text

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="RAG Térmicas", page_icon="", layout="wide")
st.title("Manual de funcionamiento y fallas de una Central Termoeléctrica")
st.caption("Búsqueda + Razonamiento con Claude")

query = st.text_input("Escribe tu pregunta sobre los manuales:", "")
top_k = st.slider("Número de fragmentos a recuperar", min_value=1, max_value=10, value=4)

if st.button("Consultar"):
    if not query.strip():
        st.warning("Escribe una pregunta antes de continuar.")
        st.stop()

    st.info("Buscando información relevante...")
    chunks = search_chunks(query, top_k=top_k)

    st.info("Generando respuesta con Claude...")
    answer = generate_answer(query, chunks)

    st.subheader("Respuesta")
    st.write(answer)

    st.subheader("Fragmentos utilizados")
    for idx, ch in enumerate(chunks, start=1):
        with st.expander(f"Fragmento {idx} — {ch['source']}"):
            st.write(ch["text"])
            st.caption(f"Chunk ID: {ch['id']}")

st.markdown("Sistema RAG optimizado para documentación basado en Centrales Térmicas")
