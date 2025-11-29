from dataclasses import dataclass
import logging
from utils.llm_client.base import BaseClient, BaseLLMClientConfig

try:
    from litellm import completion
except ImportError:
    completion = None

logger = logging.getLogger(__name__)


@dataclass
class LiteLLMClientConfig(BaseLLMClientConfig):
    pass


class LiteLLMClient(BaseClient):

    def __init__(
        self,
        config: LiteLLMClientConfig,
    ) -> None:
        super().__init__(config=config)

        if completion is None:
            logger.fatal(f"Package `litellm` is required")
            exit(-1)

        from litellm import validate_environment

        validity = validate_environment(self.model)
        if not validity["keys_in_environment"]:
            logger.fatal(
                f"Missing environment variables: {repr(validity['missing_keys'])}"
            )
            exit(-1)

    def _chat_completion_api(
        self, messages: list[dict], temperature: float, n: int = 1
    ) -> list[dict]:
        assert n == 1
        response = completion(
            model=self.model, messages=messages, temperature=temperature
        )
        return response.choices
