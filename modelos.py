import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Esto listará tus modelos disponibles (según la versión de la librería)
print(client.models.list())