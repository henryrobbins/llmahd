import logging
import os

from dataclasses import dataclass
from utils.llm_client.openai import OpenAIClient, OpenAIClientConfig

logger = logging.getLogger(__name__)


@dataclass
class LlamaAPIClientConfig(OpenAIClientConfig):
    pass


class LlamaAPIClient(OpenAIClient):

    def __init__(
        self,
        config: LlamaAPIClientConfig,
    ) -> None:
        if config.api_key is None:
            config.api_key = os.getenv("LLAMA_API_KEY", None)
            assert (
                config.api_key
            ), "Please provide llama API key via environment variable LLAMA_API_KEY"
        config.base_url = config.base_url or "https://api.llama-api.com"

        super().__init__(config=config)

    def _chat_completion_api(
        self, messages: list[dict], temperature: float, n: int = 1
    ) -> list[dict]:
        assert n == 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
            max_tokens=1024,
            timeout=100,
        )
        return response.choices
