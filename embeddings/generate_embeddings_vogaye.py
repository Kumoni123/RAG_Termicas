#!/usr/bin/env python3
"""
generate_embeddings.py — Versión con Voyage AI
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
import pdfplumber
import faiss
import voyageai

from dotenv import load_dotenv

load_dotenv()  # carga las variables de .env

# Optional token-aware chunking
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except:
    TIKTOKEN_AVAILABLE = False

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUALS_DIR = PROJECT_ROOT / "data" / "manuals"

VECTOR_DB_DIR = Path(__file__).resolve().parent / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHUNK_TOKENS = 450
DEFAULT_CHUNK_OVERLAP = 50
VOYAGE_MODEL = "voyage-3.5"

# ⚡ Instanciar cliente Voyage AI
voyage_api_key = os.getenv("VOYAGE_AI_API_KEY")
if not voyage_api_key:
    raise ValueError("No se encontró la variable de entorno VOYAGE_API_KEY")
voyage_client = voyageai.Client(api_key=voyage_api_key)


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
    docs = []
    for pdf in pdfs:
        print(f"Extrayendo: {pdf.name}")
        docs.append({"source": pdf.name, "text": extract_text_from_pdf(pdf)})
    return docs


# ---------- Chunking ----------
def count_tokens(text, model="gpt-4o-mini"):
    if TIKTOKEN_AVAILABLE:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    return len(text) // 4


def chunk_text(text, chunk_size, overlap, model="gpt-4o-mini"):
    if TIKTOKEN_AVAILABLE:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        chunks = []
        i = 0
        while i < len(tokens):
            sub = tokens[i : i + chunk_size]
            chunks.append(enc.decode(sub))
            i += chunk_size - overlap
        return chunks
    # fallback simple
    approx = chunk_size * 4
    ov = overlap * 4
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + approx])
        i += approx - ov
    return chunks


def create_chunks(docs, chunk_size, overlap, token_model):
    final = []
    for doc in docs:
        chs = chunk_text(doc["text"], chunk_size, overlap, token_model)
        for idx, c in enumerate(chs):
            final.append(
                {
                    "id": f"{doc['source']}__chunk_{idx}",
                    "source": doc["source"],
                    "chunk_index": idx,
                    "text": c,
                    "token_count": count_tokens(c, token_model),
                }
            )
    print(f"🔹 Total chunks generados: {len(final)}")
    return final


# ---------- Voyage AI Embeddings ----------
def embed_voyage(text_list: List[str]) -> np.ndarray:
    embeddings = []
    print(f"Generando embeddings con {VOYAGE_MODEL}...")

    for i in tqdm(range(0, len(text_list), 10)):  # batch de 10 para no saturar
        batch = text_list[i : i + 10]
        resp = voyage_client.embed(
            batch,
            model=VOYAGE_MODEL,
            input_type="document"
        )
        embeddings.extend([np.array(e, dtype="float32") for e in resp.embeddings])

    return np.vstack(embeddings)


# ---------- FAISS ----------
def build_faiss(emb):
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def save_metadata(chunks, path):
    md = []
    for i, c in enumerate(chunks):
        md.append(
            {
                "vector_index": i,
                "id": c["id"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
                "text": c["text"],
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(md, f, indent=2, ensure_ascii=False)


# ---------- MAIN ----------
def main(args):
    if not MANUALS_DIR.exists():
        print("Falta la carpeta data/manuals")
        sys.exit(1)

    docs = load_pdfs(MANUALS_DIR)
    chunks = create_chunks(docs, args.chunk_size, args.overlap, args.token_model)
    texts = [c["text"] for c in chunks]

    # Embeddings Voyage AI
    emb = embed_voyage(texts)

    index = build_faiss(emb)
    faiss.write_index(index, str(VECTOR_DB_DIR / "index.faiss"))
    print("Índice guardado en vector_db/index.faiss")

    save_metadata(chunks, VECTOR_DB_DIR / "metadata.json")
    print("Metadata guardada en vector_db/metadata.json")

    if args.save_embeddings:
        np.save(str(VECTOR_DB_DIR / "embeddings.npy"), emb)
        print("Embeddings guardados en embeddings.npy")

    print("\nCOMPLETADO — Pipeline Voyage AI listo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--token-model", default="gpt-4o-mini")
    parser.add_argument("--save-embeddings", action="store_true")
    args = parser.parse_args()

    main(args)
