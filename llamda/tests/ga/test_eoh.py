import logging
import os
from pathlib import Path

from llamda.utils.evaluate import Evaluator
from llamda.utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from llamda.utils.problem import ProblemPrompts, adapt_prompt
from llamda.utils.utils import get_output_dir, print_hyperlink
from llamda.ga.eoh.eoh import EOH, EoHConfig

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


def test_eoh() -> None:
    problem_name = "tsp_aco"

    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")

    config = OpenAIClientConfig(
        model="gpt-3.5-turbo",
        temperature=1.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    client = OpenAIClient(config)

    # ========================================================================
    root_dir = ROOT_DIR
    ouput_dir = get_output_dir("test_eoh", root_dir)

    problem_config = ProblemPrompts.load_problem_prompts(
        f"{root_dir}/llamda/prompts/{problem_name}"
    )
    prompts = adapt_prompt(problem_config)

    evaluator = Evaluator(prompts, root_dir)

    eoh_config = EoHConfig()

    # ========================================================================

    # Main algorithm
    llh = EOH(eoh_config, prompts, evaluator, client, output_dir=ouput_dir)

    best_code_overall, best_code_path_overall = llh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")
