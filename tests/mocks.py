import json
import logging
import os
from pathlib import Path
from typing import Any, TypeVar

from llamda.evaluate import Evaluator
from llamda.individual import Individual
from llamda.llm_client.base import BaseClient, BaseLLMClientConfig
from llamda.problem import BaseProblem

logger = logging.getLogger(__name__)


class MockResponse:

    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:

    def __init__(self, content: str):
        self.content = content


class MockClient(BaseClient):

    def __init__(
        self,
        config: BaseLLMClientConfig,
        responses_dir: str,
    ) -> None:

        super().__init__(config=config)
        self.responses_dir = responses_dir
        self.call_count = 0
        self.call_history: list[dict[str, Any]] = []

        self._load_responses()

    def _load_responses(self) -> None:
        """Load all response files from the responses directory."""
        self.responses: list[str] = []
        i = 0
        while True:
            filepath = os.path.join(self.responses_dir, f"{i}.txt")
            if not os.path.exists(filepath):
                break
            with open(filepath, "r") as f:
                self.responses.append(f.read())
            i += 1

        if not self.responses:
            raise ValueError(f"No response files found in {self.responses_dir}")

    def _chat_completion_api(
        self, messages: list[dict], temperature: float, n: int = 1
    ) -> list:
        """Return pre-recorded responses instead of making real API calls."""
        if n != 1:
            raise NotImplementedError("MockClient only supports n=1")

        # Log the call for debugging/verification
        self.call_history.append(
            {
                "call_index": self.call_count,
                "messages": messages,
                "temperature": temperature,
                "n": n,
            }
        )

        # Check if we have more responses available
        if self.call_count >= len(self.responses):
            raise IndexError(
                f"No more responses available. Call count: {self.call_count}, "
                f"Available responses: {len(self.responses)}"
            )

        # Get the next response
        response_content = self.responses[self.call_count]
        self.call_count += 1

        # Return mock response in the expected format
        return [MockResponse(response_content)]

    def reset(self) -> None:
        """Reset call counter and history for reuse."""
        self.call_count = 0
        self.call_history = []


T = TypeVar("T", bound=Individual)


class MockEvaluator(Evaluator):

    def __init__(
        self,
        problem: BaseProblem,
        evaluation_path: str | Path,
        timeout: int = 30,
    ) -> None:

        self.problem = problem
        self.timeout = timeout
        self.evaluation_path = Path(evaluation_path)
        self.function_evals = 0
        self.iteration = 0

        # Load cache
        self._load_cache()

    def _load_cache(self) -> None:
        """Load the evaluation cache from JSON file."""
        if not self.evaluation_path.exists():
            raise FileNotFoundError(f"Cache file not found: {self.evaluation_path}")

        with open(self.evaluation_path, "r") as f:
            self.cache = json.load(f)

    def mark_invalid_individual(self, individual: T, traceback_msg: str) -> T:
        """Mark an individual as invalid."""
        individual.exec_success = False
        individual.obj = float("inf")
        individual.traceback_msg = traceback_msg
        return individual

    def batch_evaluate(self, population: list[T], iteration: int = 0) -> list[T]:
        """Evaluate population by looking up results in the cache."""
        self.iteration = iteration

        for response_id, individual in enumerate(population):
            self.function_evals += 1

            # Skip if response is invalid
            if individual.code is None:
                population[response_id] = self.mark_invalid_individual(
                    individual, "Invalid response!"
                )
                continue

            logger.info(
                f"Iteration {self.iteration}: Evaluating Code {response_id} from cache"
            )

            # Look up result in cache
            code_key = individual.code
            if code_key not in self.cache:
                logger.warning(
                    f"Code not found in cache for response_id {response_id}. "
                    f"Marking as invalid."
                )
                population[response_id] = self.mark_invalid_individual(
                    individual, "Code not found in evaluation cache"
                )
                continue

            # Retrieve cached results
            cached_result = self.cache[code_key]

            individual.exec_success = cached_result["exec_success"]
            individual.traceback_msg = cached_result.get("traceback_msg")

            if cached_result["exec_success"]:
                # Apply objective type transformation (min vs max)
                obj_value = cached_result["obj"]
                individual.obj = (
                    -obj_value if self.problem.obj_type == "max" else obj_value
                )
            else:
                individual.obj = float("inf")

            logger.info(
                f"Iteration {self.iteration}, response_id {response_id}: "
                f"Objective value: {individual.obj} (from cache)"
            )

        return population
