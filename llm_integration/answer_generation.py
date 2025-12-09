# llm_integration/answer_generator.py

import yaml
from anthropic import Anthropic
from llm_integration.rag_prompt import build_rag_prompt
import os
from dotenv import load_dotenv
load_dotenv()


def load_config(config_path="config.yaml"):
    """Carga archivo de configuración."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AnswerGenerator:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)

        api_key = os.getenv(self.config["llm"]["api_key_env"])
        self.client = Anthropic(api_key=api_key)
        #self.client = Anthropic(api_key=self.config["llm"]["api_key"])

        self.model = self.config["llm"]["model"]
        self.max_tokens = self.config["llm"].get("max_tokens", 4000)
        self.temperature = self.config["llm"].get("temperature", 0.0)

    def generate(self, user_query: str, retrieved_chunks: list) -> str:
        """
        Genera una respuesta final usando el modelo Claude con los chunks recuperados.
        """

        prompt = build_rag_prompt(user_query, retrieved_chunks)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Anthropic devuelve message.content como lista de bloques
        result = response.content[0].text
        return result


if __name__ == "__main__":
    # Ejemplo manual
    test_query = "¿Cuáles son los procedimientos de emergencia?"
    test_chunks = [
        {"text": "Los procedimientos de emergencia incluyen..."},
    ]

    generator = AnswerGenerator()
    answer = generator.generate(test_query, test_chunks)
    print("\n======= RESPUESTA =======\n")
    print(answer)
