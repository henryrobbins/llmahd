from importlib.resources import files
import logging
from pathlib import Path
import pytest

from llamda.llm_client.base import BaseLLMClientConfig
from llamda.problem import Problem, adapt_prompt
from llamda.ga.mcts.mcts_ahd import AHDConfig
from llamda.ga.mcts.mcts_ahd import MCTS_AHD as LHH

from tests.common import EVALUATIONS_PATH, RESPONSES_PATH
from tests.mocks import MockClient, MockEvaluator


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_mcts(problem_name: str, tmp_path: Path) -> None:

    client = MockClient(
        config=BaseLLMClientConfig(model="mock", temperature=1.0),
        responses_dir=str(RESPONSES_PATH / "mcts"),
    )

    problem = Problem.load_problem(
        path=str(files("llamda.prompts.problems") / problem_name)
    )
    eoh_problem = adapt_prompt(problem)
    evaluator = MockEvaluator(
        eoh_problem, evaluation_path=str(EVALUATIONS_PATH / "mcts.json")
    )

    lhh = LHH(
        config=AHDConfig(init_size=5, ec_fe_max=15),
        problem=eoh_problem,
        evaluator=evaluator,
        output_dir=str(tmp_path / "test_mcts"),
        llm_client=client,
    )

    best_code_overall, _ = lhh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")
