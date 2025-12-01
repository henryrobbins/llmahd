from importlib.resources import files
import logging
from pathlib import Path
import pytest

from llamda.llm_client.base import BaseLLMClientConfig
from llamda.problem import Problem
from llamda.ga.reevo.reevo import ReEvo, ReEvoConfig

from tests.common import EVALUATIONS_PATH, RESPONSES_PATH
from tests.mocks import MockClient, MockEvaluator


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_reevo(problem_name: str, tmp_path: Path) -> None:

    client = MockClient(
        config=BaseLLMClientConfig(model="mock", temperature=1.0),
        responses_dir=str(RESPONSES_PATH / "reevo"),
    )
    problem = Problem.load_problem(
        path=str(files("llamda.prompts.problems") / problem_name)
    )
    evaluator = MockEvaluator(
        problem, evaluation_path=str(EVALUATIONS_PATH / "reevo.json")
    )

    reevo = ReEvo(
        config=ReEvoConfig(init_pop_size=5, max_fe=15),
        problem=problem,
        evaluator=evaluator,
        output_dir=str(tmp_path / "test_reevo"),
        llm_client=client,
    )

    best_code_overall, _ = reevo.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
