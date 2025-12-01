import logging
import numpy as np
from pathlib import Path
import pytest

from llamda.ga.hsevo.hsevo import HSEvo, HSEvoConfig
from llamda.llm_client.base import BaseLLMClientConfig
from llamda.problem import Problem, ProblemName

from tests.common import EVALUATIONS_PATH, RESPONSES_PATH
from tests.mocks import MockClient, MockEvaluator

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("problem_name", [ProblemName.TSP_ACO])
def test_hsevo(problem_name: ProblemName, tmp_path: Path) -> None:
    np.random.seed(42)

    client = MockClient(
        config=BaseLLMClientConfig(model="mock", temperature=1.0),
        responses_dir=str(RESPONSES_PATH / "hsevo"),
    )
    problem = Problem.load_builtin(problem_name)
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
    logger.info(f"Best Code Overall: {best_code_overall}")
    logger.info(f"Best Code Path Overall: {best_code_path_overall}")
