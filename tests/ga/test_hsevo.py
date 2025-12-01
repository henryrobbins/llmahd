from importlib.resources import files
import logging
import numpy as np
from pathlib import Path
import pytest

from llamda.ga.hsevo.hsevo import HSEvo, HSEvoConfig
from llamda.llm_client.base import BaseLLMClientConfig
from llamda.problem import Problem

from tests.common import EVALUATIONS_PATH, RESPONSES_PATH
from tests.mocks import MockClient, MockEvaluator


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_hsevo(problem_name: str, tmp_path: Path) -> None:
    np.random.seed(42)

    client = MockClient(
        config=BaseLLMClientConfig(model="mock", temperature=1.0),
        responses_dir=str(RESPONSES_PATH / "hsevo"),
    )
    problem = Problem.load_problem(str(files("llamda.prompts.problems") / problem_name))
    evaluator = MockEvaluator(
        problem, evaluation_path=str(EVALUATIONS_PATH / "hsevo.json")
    )

    hsevo = HSEvo(
        config=HSEvoConfig(init_pop_size=5, max_fe=15),
        problem=problem,
        evaluator=evaluator,
        llm_client=client,
        output_dir=str(tmp_path / "test_hsevo"),
    )

    best_code_overall, best_code_path_overall = hsevo.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")
