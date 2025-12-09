import os
import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()


def load_config(config_path="config.yaml"):
    """Carga la configuración general del proyecto."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LLMClient:
    """
    Cliente para manejar la interacción con modelos Anthropic (Claude).
    Usa .env para la API key y config.yaml para parámetros.
    """

    def __init__(self, config_path="config.yaml"):
        cfg = load_config(config_path)

        # Prioridad: .env > config.yaml
        api_key = os.getenv("ANTHROPIC_API_KEY") or cfg["llm"]["api_key"]
        if not api_key:
            raise ValueError("No se encontró la API key de Anthropic (env o config.yaml)")

        self.client = Anthropic(api_key=api_key)
        self.model = cfg["llm"]["model"]
        self.max_tokens = cfg["llm"].get("max_tokens", 4000)
        self.temperature = cfg["llm"].get("temperature", 0.0)

    def ask(self, prompt: str) -> str:
        """
        Envía un prompt y devuelve solo el texto de la respuesta.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error en la llamada al LLM: {e}")
            return "Error al generar la respuesta."

    def ask_raw(self, prompt: str):
        """Devuelve la respuesta completa sin procesar (útil para debug)."""
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )


if __name__ == "__main__":
    client = LLMClient()
    print(client.ask("Hola, ¿qué puedes hacer?"))
