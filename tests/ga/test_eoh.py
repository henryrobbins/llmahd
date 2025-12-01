from importlib.resources import files
import logging
from pathlib import Path
import pytest

from llamda.utils.llm_client.base import BaseLLMClientConfig
from llamda.utils.problem import ProblemPrompts, adapt_prompt
from llamda.ga.eoh.eoh import EOH, EoHConfig

from tests.common import EVALUATIONS_PATH, RESPONSES_PATH
from tests.mocks import MockClient, MockEvaluator


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_eoh(problem_name: str, tmp_path: Path) -> None:

    client = MockClient(
        config=BaseLLMClientConfig(model="mock", temperature=1.0),
        responses_dir=str(RESPONSES_PATH / "eoh"),
    )
    problem_prompts = ProblemPrompts.load_problem_prompts(
        str(files("llamda.prompts.problems") / problem_name)
    )
    eoh_problem_prompts = adapt_prompt(problem_prompts)
    evaluator = MockEvaluator(
        eoh_problem_prompts, evaluation_path=str(EVALUATIONS_PATH / "eoh.json")
    )

    llh = EOH(
        config=EoHConfig(ec_pop_size=3),
        problem=eoh_problem_prompts,
        evaluator=evaluator,
        llm_client=client,
        output_dir=str(tmp_path / "test_eoh"),
    )

    best_code_overall, _ = llh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")
