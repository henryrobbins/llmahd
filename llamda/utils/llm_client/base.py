import os
import time
import logging
import concurrent
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class BaseLLMClientConfig:
    model: str
    temperature: float = 1.0


class BaseClient(object):
    def __init__(
        self,
        config: BaseLLMClientConfig,
    ) -> None:
        self.model = config.model
        self.temperature = config.temperature

    def _chat_completion_api(
        self, messages: list[dict], temperature: float, n: int = 1
    ) -> list[dict]:
        raise NotImplementedError

    def chat_completion(
        self, n: int, messages: list[dict], temperature: float | None = None
    ) -> list[dict]:
        """
        Generate n responses using OpenAI Chat Completions API
        """
        temperature = temperature or self.temperature

        for attempt in range(3):
            try:
                return self._chat_completion_api(messages, temperature, n)
            except Exception as e:
                if attempt == 2:  # Last attempt
                    logger.error("All 3 attempts failed")
                    raise
                logger.warning(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)

    def multi_chat_completion(
        self,
        messages_list: list[list[dict]],
        n: int = 1,
        temperature: float | None = None,
    ) -> list[str]:
        """
        Generate multiple chat completions in parallel.
        param: n: number of responses to generate for each message in messages_list
        """
        # If messages_list is not a list of list (i.e., only one conversation),
        # convert it to a list of list
        assert isinstance(messages_list, list), "messages_list should be a list."
        try:
            if not isinstance(messages_list[0], list):
                messages_list = [messages_list]
        except:
            print(messages_list)
            raise IndexError("Something is wrong.")

        if len(messages_list) > 1:
            assert n == 1, "Currently, only n=1 is supported for multi-chat completion."

        num_workers = os.cpu_count()
        if "gpt" not in self.model:
            # Transform messages if n > 1
            messages_list *= n
            n = 1
            num_workers = 2

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            args = [
                dict(n=n, messages=messages, temperature=temperature)
                for messages in messages_list
            ]
            choices = executor.map(lambda p: self.chat_completion(**p), args)

        contents: list[str] = []
        for choice in choices:
            for c in choice:
                contents.append(c.message.content)
        return contents
