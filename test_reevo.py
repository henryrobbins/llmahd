import logging
import os
from pathlib import Path
from utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from utils.problem import ProblemPrompts
from utils.utils import get_output_dir, print_hyperlink

from ga.reevo.reevo import ReEvo as LHH

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_reevo", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


def main() -> None:
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

    prompt_dir = f"{ROOT_DIR}/prompts"
    prompts = ProblemPrompts.load_problem_prompts(
        path=f"{prompt_dir}/{problem_name}",
    )

    lhh = LHH(prompts, ROOT_DIR, output_dir=output_dir, generator_llm=client)

    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    best_path = best_code_path_overall.replace(".py", ".txt").replace(
        "code", "response"
    )
    logging.info(
        f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}"
    )


if __name__ == "__main__":
    main()
