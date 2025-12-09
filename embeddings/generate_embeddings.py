#!/usr/bin/env python3
"""
generate_embeddings.py — Embeddings con Sentence Transformers
"""

import os
import sys
import json
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUALS_DIR = PROJECT_ROOT / "data" / "manuals"

VECTOR_DB_DIR = Path(__file__).resolve().parent / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHUNK_TOKENS = 450
DEFAULT_CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # modelo compacto y rápido

# ---------- PDF extraction ----------
def extract_text_from_pdf(pdf_path: Path) -> str:
    text_pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_pages.append(t)
    return "\n".join(text_pages)


def load_pdfs(folder: Path):
    pdfs = [p for p in folder.iterdir() if p.suffix.lower() == ".pdf"]
    if not pdfs:
        print(f"No se encontraron PDFs en {folder}")
        sys.exit(1)
    docs = []
    for pdf in pdfs:
        print(f"Extrayendo: {pdf.name}")
        docs.append({"source": pdf.name, "text": extract_text_from_pdf(pdf)})
    return docs


# ---------- Chunking ----------
def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_TOKENS, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    tokens = text.split()  # división simple por palabras
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def create_chunks(docs: List[dict], chunk_size: int, overlap: int):
    final = []
    for doc in docs:
        chs = chunk_text(doc["text"], chunk_size, overlap)
        for idx, c in enumerate(chs):
            final.append({
                "id": f"{doc['source']}__chunk_{idx}",
                "source": doc["source"],
                "chunk_index": idx,
                "text": c,
                "token_count": len(c.split())
            })
    print(f"🔹 Total chunks generados: {len(final)}")
    return final


# ---------- Sentence Transformers Embeddings ----------
def embed_sentencetransformer(text_list: List[str]) -> np.ndarray:
    print(f"Generando embeddings con {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(text_list, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


# ---------- FAISS ----------
def build_faiss(emb: np.ndarray):
    index = faiss.IndexFlatIP(emb.shape[1])  # similaridad coseno
    index.add(emb)
    return index


def save_metadata(chunks: List[dict], path: Path):
    md = []
    for i, c in enumerate(chunks):
        md.append({
            "vector_index": i,
            "id": c["id"],
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "token_count": c["token_count"],
            "text": c["text"],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(md, f, indent=2, ensure_ascii=False)


# ---------- MAIN ----------
def main(save_embeddings: bool = False):
    if not MANUALS_DIR.exists():
        print(f"Falta la carpeta {MANUALS_DIR}")
        sys.exit(1)

    docs = load_pdfs(MANUALS_DIR)
    chunks = create_chunks(docs, DEFAULT_CHUNK_TOKENS, DEFAULT_CHUNK_OVERLAP)
    texts = [c["text"] for c in chunks]

    emb = embed_sentencetransformer(texts)

    index = build_faiss(emb)
    faiss.write_index(index, str(VECTOR_DB_DIR / "index.faiss"))
    print(f"Índice guardado en {VECTOR_DB_DIR / 'index.faiss'}")

    save_metadata(chunks, VECTOR_DB_DIR / "metadata.json")
    print(f"Metadata guardada en {VECTOR_DB_DIR / 'metadata.json'}")

    if save_embeddings:
        np.save(str(VECTOR_DB_DIR / "embeddings.npy"), emb)
        print("Embeddings guardados en embeddings.npy")

    print("\nCOMPLETADO — Pipeline listo con Sentence Transformers")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-embeddings", action="store_true", help="Guardar embeddings en .npy")
    args = parser.parse_args()

    main(save_embeddings=args.save_embeddings)
