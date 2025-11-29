from dataclasses import dataclass
from utils.llm_client.openai import OpenAIClient, OpenAIClientConfig

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = "zhipuai"


@dataclass
class ZhipuAIClientConfig(OpenAIClientConfig):
    pass


class ZhipuAIClient(OpenAIClient):

    ClientClass = ZhipuAI

    def _chat_completion_api(
        self, messages: list[dict], temperature: float, n: int = 1
    ) -> list[dict]:
        assert n == 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=min(temperature, 1.0),
        )
        return response.choices
