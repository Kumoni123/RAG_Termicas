# llm_integration/rag_prompt.py

def build_rag_prompt(user_query: str, retrieved_chunks: list) -> str:
    """
    Construye el prompt para RAG usando los chunks recuperados desde la base vectorial.

    Params:
        - user_query: consulta del usuario
        - retrieved_chunks: lista de dicts: [{ "text": "...", "source": "...", "page": 12 }]
    
    Retorna:
        - prompt final string
    """

    # Construcción de texto con las fuentes relevantes
    context_text = ""
    for i, ch in enumerate(retrieved_chunks, start=1):
        context_text += (
            f"【Fragmento {i} — Fuente: {ch.get('source', 'N/A')} — Página: {ch.get('page', '?')}】\n"
            f"{ch['text']}\n\n"
        )

    prompt = f"""
Eres un asistente experto en documentación técnica empresarial.

Los fragmentos proporcionados pueden estar en **español o inglés**.
Tu tarea es responder usando **únicamente** la información de los fragmentos proporcionados.
Responde siempre en el mismo idioma en el que se hizo la consulta.

Si la respuesta no está en la documentación, responde:
"Lo siento, no encuentro información suficiente en los documentos."

### Consulta del usuario:
{user_query}

---

### Fragmentos relevantes:
{context_text}

---

### Reglas importantes:
- No inventes información.
- Usa solamente los fragmentos dados.
- Si hay contradicciones, indica cuál fragmento es más confiable y por qué.
- Cita los fragmentos usando el formato: (Fragmento X)
- Responde de forma clara, profesional y estructurada.

### Respuesta:
"""

    return prompt
