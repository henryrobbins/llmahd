from importlib.resources import files
import logging
import os
from pathlib import Path

from llamda.ga.mcts.mcts_ahd import MCTS_AHD, AHDConfig
from llamda.utils.evaluate import Evaluator
from llamda.utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from llamda.utils.problem import ProblemPrompts, adapt_prompt
from llamda.utils.utils import get_output_dir

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_mcts-ahd", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


def test_mcts() -> None:
    problem_name = "tsp_constructive"

    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    config = OpenAIClientConfig(
        model="gpt-3.5-turbo",
        temperature=1.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    client = OpenAIClient(config)

    # ========================================================================

    problem_prompts_dir = files("llamda.prompts.problems")
    problem_config = ProblemPrompts.load_problem_prompts(
        str(problem_prompts_dir / problem_name)
    )

    prompts = adapt_prompt(problem_config)

    evaluator = Evaluator(prompts)

    ahd_config = AHDConfig()

    llm_client = client

    # ========================================================================

    # Main algorithm
    lhh = MCTS_AHD(ahd_config, prompts, evaluator, llm_client, output_dir)
    best_code_overall, best_code_path_overall = lhh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")
