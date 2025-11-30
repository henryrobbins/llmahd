from importlib.resources import files
import logging
import os

import pytest

from llamda.utils.evaluate import Evaluator
from llamda.utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from llamda.utils.problem import ProblemPrompts
from llamda.utils.utils import get_output_dir
from llamda.ga.reevo.reevo import ReEvo, ReEvoConfig

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_reevo", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_reevo(problem_name: str) -> None:

    client = OpenAIClient(
        config=OpenAIClientConfig(
            model="gpt-3.5-turbo",
            temperature=1.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )
    prompts = ProblemPrompts.load_problem_prompts(
        path=str(files("llamda.prompts.problems") / problem_name)
    )

    reevo = ReEvo(
        config=ReEvoConfig(init_pop_size=5, max_fe=15),
        problem=prompts,
        evaluator=Evaluator(prompts, timeout=20),
        output_dir=output_dir,
        llm_client=client,
    )

    best_code_overall, _ = reevo.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
