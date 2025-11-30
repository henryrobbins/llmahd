from importlib.resources import files
import logging
import os
from pathlib import Path

from llamda.utils.evaluate import Evaluator
from llamda.utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from llamda.utils.problem import ProblemPrompts
from llamda.utils.utils import get_output_dir, print_hyperlink
from llamda.ga.reevo.reevo import ReEvo as LHH, ReEvoConfig

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_reevo", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


def test_reevo() -> None:
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

    problems_dir = files("llamda.prompts.problems")
    prompts = ProblemPrompts.load_problem_prompts(
        path=str(problems_dir / problem_name),
    )

    reevo_config = ReEvoConfig()
    evaluator = Evaluator(prompts)

    lhh = LHH(
        reevo_config,
        prompts,
        evaluator=evaluator,
        output_dir=output_dir,
        generator_llm=client,
    )

    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    best_path = best_code_path_overall.replace(".py", ".txt").replace(
        "code", "response"
    )
    logging.info(
        f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}"
    )
