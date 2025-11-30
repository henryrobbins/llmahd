import logging
import os
from pathlib import Path
from ga.eoh.config import Config
from utils.evaluate import Evaluator
from utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from utils.problem import ProblemPrompts, adapt_prompt
from utils.utils import get_output_dir, print_hyperlink

from ga.eoh.eoh import EOH, EoHConfig

ROOT_DIR = os.getcwd()
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

    # ========================================================================
    root_dir = ROOT_DIR
    ouput_dir = get_output_dir("test_eoh", root_dir)

    problem_config = ProblemPrompts.load_problem_prompts(
        f"{root_dir}/prompts/{problem_name}"
    )

    if problem_config.problem_type == "constructive":
        from utils.problem import TSP_CONSTRUCTIVE_PROMPTS

        prompts = TSP_CONSTRUCTIVE_PROMPTS
    elif problem_config.problem_type == "online":
        from utils.problem import BPP_ONLINE_PROMPTS

        prompts = BPP_ONLINE_PROMPTS
    else:
        prompts = adapt_prompt(problem_config)

    evaluator = Evaluator(prompts, root_dir)

    eoh_config = EoHConfig()

    paras = Config(
        ec_pop_size=eoh_config.pop_size,
        ec_n_pop=(eoh_config.max_fe - 2 * eoh_config.pop_size)
        // (4 * eoh_config.pop_size)
        + 1,  # total evals = 2 * pop_size + n_pop * 4 * pop_size; for pop_size = 10, n_pop = 5, total evals = 2 * 10 + 4 * 5 * 10 = 220
        exp_output_path="./",
    )

    # ========================================================================

    # Main algorithm
    llh = EOH(paras, prompts, evaluator, client, output_dir=ouput_dir)

    best_code_overall, best_code_path_overall = llh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")


if __name__ == "__main__":
    main()
