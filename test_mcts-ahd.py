import logging
import os
from pathlib import Path

from ga.mcts.config import Config
from ga.mcts.mcts_ahd import MCTS_AHD, AHDConfig
from utils.evaluate import Evaluator
from utils.llm_client.openai import OpenAIClient, OpenAIClientConfig
from utils.problem import ProblemPrompts, adapt_prompt
from utils.utils import get_output_dir

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_mcts-ahd", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


def main() -> None:
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

    root_dir = ROOT_DIR

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

    ahd_config = AHDConfig()

    paras = Config(
        init_size=ahd_config.init_pop_size,
        pop_size=ahd_config.pop_size,
        ec_fe_max=ahd_config.max_fe,
        exp_output_path=f"{workspace_dir}/",
    )

    llm_client = client

    # ========================================================================

    # Main algorithm
    lhh = MCTS_AHD(paras, prompts, evaluator, llm_client, output_dir)
    best_code_overall, best_code_path_overall = lhh.run()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")


if __name__ == "__main__":
    main()
