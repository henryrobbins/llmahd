from importlib.resources import files
import logging
import os

import pytest

from llamda.utils.evaluate import Evaluator
from llamda.utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from llamda.utils.problem import ProblemPrompts, adapt_prompt
from llamda.utils.utils import get_output_dir
from llamda.ga.eoh.eoh import EOH, EoHConfig

ROOT_DIR = os.getcwd()
ouput_dir = get_output_dir("test_eoh", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize("problem_name", ["tsp_aco"])
def test_eoh(problem_name: str) -> None:

    client = OpenAIClient(
        config=OpenAIClientConfig(
            model="gpt-3.5-turbo",
            temperature=1.0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )
    problem_prompts = ProblemPrompts.load_problem_prompts(
        str(files("llamda.prompts.problems") / problem_name)
    )
    eoh_problem_prompts = adapt_prompt(problem_prompts)

    llh = EOH(
        config=EoHConfig(ec_pop_size=3),
        problem=eoh_problem_prompts,
        evaluator=Evaluator(eoh_problem_prompts, timeout=5),
        llm_client=client,
        output_dir=ouput_dir,
    )

    best_code_overall, _ = llh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")
