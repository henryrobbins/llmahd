from importlib.resources import files
import logging
import os
from pathlib import Path

from llamda.ga.hsevo.hsevo import HSEvo as LHH, HSEvoConfig
from llamda.utils.evaluate import Evaluator
from llamda.utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from llamda.utils.problem import ProblemPrompts
from llamda.utils.utils import get_output_dir

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_hsevo", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


def test_hsevo() -> None:
    problem_name = "tsp_aco"

    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    llm_config = OpenAIClientConfig(
        model="gpt-4o-mini-2024-07-18",
        temperature=1.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    client = OpenAIClient(llm_config)

    prompt_dir = files("llamda.prompts.problems")
    prompts = ProblemPrompts.load_problem_prompts(str(prompt_dir / problem_name))

    config = HSEvoConfig()

    evaluator = Evaluator(prompts)

    # Main algorithm
    lhh = LHH(
        config=config,
        problem_prompts=prompts,
        evaluator=evaluator,
        llm_client=client,
        temperature=1.0,
        output_dir=output_dir,
    )
    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")
