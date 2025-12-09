#!/usr/bin/env python3
"""
utils.py — Utilidades del módulo de retrieval
----------------------------------------------
• Cargar FAISS
• Cargar metadata
• Embeddings de queries (Anthropic)
• Normalización
• Funciones auxiliares
"""

import os
import json#!/usr/bin/env python3
"""
utils.py — Utilidades del módulo de retrieval (Sentence Transformers)
----------------------------------------------------------------------

• Cargar FAISS
• Cargar metadata
• Embeddings de queries
• Normalización
• Funciones auxiliares
"""

import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------- CONFIG -------------------
ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = ROOT / "embeddings" / "vector_db"

INDEX_PATH = VECTOR_DB_PATH / "index.faiss"
METADATA_PATH = VECTOR_DB_PATH / "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ------------------- LOADERS -------------------
def load_faiss_index():
    """Carga el índice FAISS desde vector_db/index.faiss"""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No existe el índice FAISS en: {INDEX_PATH}")

    print(f"[INFO] Cargando índice FAISS: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))
    return index


def load_metadata() -> List[Dict]:
    """Carga el metadata.json que contiene información de cada chunk."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"No existe metadata.json en: {METADATA_PATH}")

    print(f"[INFO] Cargando metadata: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- EMBEDDINGS QUERY ----------------
def embed_query(text: str) -> np.ndarray:
    """
    Genera un embedding de una consulta usando Sentence Transformers
    """
    vec = model.encode([text], normalize_embeddings=True)
    return np.array(vec, dtype="float32").reshape(1, -1)


# ------------------- UTILIDADES -------------------
def validate_vector_db():
    """Verifica que FAISS + metadata existan."""
    if not INDEX_PATH.exists():
        raise RuntimeError("No se encontró el índice FAISS. Debes correr generate_embeddings.py primero.")

    if not METADATA_PATH.exists():
        raise RuntimeError(" No se encontró metadata.json. Falta correr generate_embeddings.py.")

    print("✔ Vector DB OK | FAISS + metadata disponibles.")


def get_chunk_text(metadata: List[Dict], idx: int) -> Dict:
    """
    Devuelve un chunk completo del metadata dado un índice del vector.
    Útil para ver más información del chunk devuelto por FAISS.
    """
    return metadata[idx]

from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from anthropic import Anthropic

# ------------------- CONFIG -------------------
ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = ROOT / "embeddings" / "vector_db"

INDEX_PATH = VECTOR_DB_PATH / "index.faiss"
METADATA_PATH = VECTOR_DB_PATH / "metadata.json"

EMBED_MODEL = "claude-3-haiku-20240307"

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ------------------- LOADERS -------------------
def load_faiss_index():
    """Carga el índice FAISS desde vector_db/index.faiss"""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No existe el índice FAISS en: {INDEX_PATH}")

    print(f"[INFO] Cargando índice FAISS: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))
    return index


def load_metadata() -> List[Dict]:
    """Carga el metadata.json que contiene información de cada chunk."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"No existe metadata.json en: {METADATA_PATH}")

    print(f"[INFO] Cargando metadata: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- EMBEDDINGS QUERY ----------------
def embed_query(text: str) -> np.ndarray:
    """
    Genera un embedding de una consulta usando Anthropic (mismo modelo que generate_embeddings.py)
    """
    resp = anthropic_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    vec = np.array(resp.data[0].embedding, dtype="float32")

    # Normalizamos para similitud por producto punto
    vec = vec / np.linalg.norm(vec)

    return vec.reshape(1, -1)


# ------------------- UTILIDADES -------------------
def validate_vector_db():
    """Verifica que FAISS + metadata existan."""
    if not INDEX_PATH.exists():
        raise RuntimeError("No se encontró el índice FAISS. Debes correr generate_embeddings.py primero.")

    if not METADATA_PATH.exists():
        raise RuntimeError("No se encontró metadata.json. Falta correr generate_embeddings.py.")

    print("✔ Vector DB OK | FAISS + metadata disponibles.")


def get_chunk_text(metadata: List[Dict], idx: int) -> Dict:
    """
    Devuelve un chunk completo del metadata dado un índice del vector.
    Útil para ver más información del chunk devuelto por FAISS.
    """
    return metadata[idx]
