from dataclasses import dataclass
import logging
from utils.llm_client.base import BaseClient, BaseLLMClientConfig

try:
    from openai import OpenAI
except ImportError:
    OpenAI = "openai"


logger = logging.getLogger(__name__)


@dataclass
class OpenAIClientConfig(BaseLLMClientConfig):
    base_url: str | None = None
    api_key: str | None = None


class OpenAIClient(BaseClient):

    ClientClass = OpenAI

    def __init__(
        self,
        config: OpenAIClientConfig,
    ) -> None:
        super().__init__(config=config)

        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)

        self.client = self.ClientClass(api_key=config.api_key, base_url=config.base_url)

    def _chat_completion_api(
        self, messages: list[dict], temperature: float, n: int = 1
    ) -> list[dict]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            n=n,
            stream=False,
        )
        return response.choices
