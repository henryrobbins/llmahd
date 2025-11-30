from importlib.resources import files
import logging
import os

import pytest

from llamda.ga.hsevo.hsevo import HSEvo, HSEvoConfig
from llamda.utils.evaluate import Evaluator
from llamda.utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from llamda.utils.problem import ProblemPrompts
from llamda.utils.utils import get_output_dir

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_hsevo", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_hsevo(problem_name: str) -> None:

    client = OpenAIClient(
        config=OpenAIClientConfig(
            model="gpt-4o-mini-2024-07-18",
            temperature=1.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )
    prompts = ProblemPrompts.load_problem_prompts(
        str(files("llamda.prompts.problems") / problem_name)
    )

    hsevo = HSEvo(
        config=HSEvoConfig(init_pop_size=5, max_fe=15),
        problem=prompts,
        evaluator=Evaluator(prompts, timeout=50),
        llm_client=client,
        output_dir=output_dir,
    )

    best_code_overall, best_code_path_overall = hsevo.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")
