import logging
from pathlib import Path
import pytest

import numpy as np

from llamda.llm_client.base import BaseLLMClientConfig
from llamda.problem import Problem, ProblemName, adapt_prompt
from llamda.ga.mcts.mcts_ahd import AHDConfig
from llamda.ga.mcts.mcts_ahd import MCTS_AHD as LHH

from tests.common import EVALUATIONS_PATH, RESPONSES_PATH
from tests.mocks import MockClient, MockEvaluator

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("problem_name", [ProblemName.TSP_ACO])
def test_mcts(problem_name: ProblemName, tmp_path: Path) -> None:
    np.random.seed(2024)

    client = MockClient(
        config=BaseLLMClientConfig(model="mock", temperature=1.0),
        responses_dir=str(RESPONSES_PATH / "mcts"),
    )

    problem = Problem.load_builtin(problem_name)
    eoh_problem = adapt_prompt(problem)
    evaluator = MockEvaluator(
        eoh_problem, evaluation_path=str(EVALUATIONS_PATH / "mcts.json")
    )

    lhh = LHH(
        config=AHDConfig(init_size=5, fe_max=15),
        problem=eoh_problem,
        evaluator=evaluator,
        output_dir=str(tmp_path / "test_mcts"),
        llm_client=client,
    )

    best_code_overall, _ = lhh.run()
    logger.info(f"Best Code Overall: {best_code_overall}")
