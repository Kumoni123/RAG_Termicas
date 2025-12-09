from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()  # carga las variables de .env

client = Anthropic(api_key="TEST")
print(client)

from anthropic import Anthropic
import os

# Asegúrate que ANTHROPIC_API_KEY está en tu entorno
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Falta ANTHROPIC_API_KEY en variables de entorno")

client = Anthropic(api_key=api_key)
print("Cliente cargado OK:", client)

try:
    print("Probando llamada simple a Cclaude-opus-4-5-20251101laude...")

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",  # << MODELO COMPATIBLE
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Dime un dato curioso."}
        ]
    )

    print("Respuesta de Claude 3 Sonnet:")
    print(response.content[0].text)

except Exception as e:
    print("Error durante la prueba:")
    print(e)
