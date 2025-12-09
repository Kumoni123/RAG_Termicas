import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorSearch:
    """
    Clase encargada de cargar la base vectorial (FAISS) y realizar búsquedas semánticas
    usando los embeddings generados con Sentence Transformers.
    """

    def __init__(self, index_path: str, metadata_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = SentenceTransformer(model_name)

        # ---------------------------
        # Cargar FAISS index
        # ---------------------------
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index no encontrado: {index_path}")
        print(f"[INFO] Cargando FAISS index desde: {index_path}")
        self.index = faiss.read_index(index_path)

        # ---------------------------
        # Cargar metadata
        # ---------------------------
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata JSON no encontrado: {metadata_path}")
        print(f"[INFO] Cargando metadata desde: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    # ---------------------------
    # Generar embedding del query
    # ---------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """
        Genera embedding del query usando Sentence Transformers.
        """
        print("[INFO] Generando embedding del query…")
        emb = self.model.encode([query], normalize_embeddings=True)
        return np.array(emb, dtype="float32")

    # ---------------------------
    # Búsqueda vectorial
    # ---------------------------
    def search(self, query: str, top_k: int = 5):
        """
        Busca los chunks más similares al query.
        Retorna una lista de diccionarios con:
        {
            "text": chunk de texto,
            "source": archivo origen,
            "distance": similitud (menor = más cercano)
        }
        """
        print(f"[INFO] Buscando top-{top_k} resultados…")
        query_embedding = self.embed_query(query).reshape(1, -1)

        # FAISS search
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in range(top_k):
            idx = int(indices[0][i])
            score = float(distances[0][i])

            if idx < 0 or idx >= len(self.metadata):
                continue

            results.append({
                "text": self.metadata[idx]["text"],
                "source": self.metadata[idx]["source"],
                "distance": score
            })

        return results


# ---------------------------
# Prueba rápida
# ---------------------------
if __name__ == "__main__":
    index_path = "embeddings/vector_db/index.faiss"
    metadata_path = "embeddings/vector_db/metadata.json"

    vs = VectorSearch(index_path, metadata_path)
    query = "Explica cómo funciona la turbina de vapor."
    results = vs.search(query, top_k=3)

    for i, r in enumerate(results, 1):
        print(f"Resultado {i}:")
        print(f"Fuente: {r['source']}")
        print(f"Texto: {r['text'][:200]}…")
        print(f"Distancia: {r['distance']}")
        print("="*50)
