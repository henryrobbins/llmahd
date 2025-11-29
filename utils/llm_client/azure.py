from dataclasses import dataclass
import logging
from utils.llm_client.base import BaseClient, BaseLLMClientConfig

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = "openai"


logger = logging.getLogger(__name__)


@dataclass
class AzureOpenAIClientConfig(BaseLLMClientConfig):
    endpoint: str | None = None
    deployment: str | None = None
    api_key: str | None = None
    api_version: str = "2024-12-01-preview"


class AzureOpenAIClient(BaseClient):

    ClientClass = AzureOpenAI

    def __init__(
        self,
        config: AzureOpenAIClientConfig,
        **kwargs: dict,
    ) -> None:
        super().__init__(config=config)

        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)

        self.client = self.ClientClass(
            azure_endpoint=config.endpoint,
            azure_deployment=config.deployment,
            api_key=config.api_key,
            api_version=config.api_version,
            **kwargs,
        )

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
