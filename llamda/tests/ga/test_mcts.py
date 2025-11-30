from importlib.resources import files
import logging
import os

import pytest

from llamda.utils.evaluate import Evaluator
from llamda.utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from llamda.utils.problem import ProblemPrompts, adapt_prompt
from llamda.utils.utils import get_output_dir
from llamda.ga.mcts.mcts_ahd import AHDConfig
from llamda.ga.mcts.mcts_ahd import MCTS_AHD as LHH

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_mcts", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_mcts(problem_name: str) -> None:

    config = OpenAIClientConfig(
        model="gpt-3.5-turbo",
        temperature=1.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    client = OpenAIClient(config)

    problems_dir = files("llamda.prompts.problems")
    problem_prompts = ProblemPrompts.load_problem_prompts(
        path=str(problems_dir / problem_name),
    )
    eoh_problem_prompts = adapt_prompt(problem_prompts)

    lhh = LHH(
        config=AHDConfig(init_size=5, ec_fe_max=15),
        problem=eoh_problem_prompts,
        evaluator=Evaluator(eoh_problem_prompts),
        output_dir=output_dir,
        llm_client=client,
    )

    best_code_overall, _ = lhh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")
