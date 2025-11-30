import logging
import os
from pathlib import Path

from ga.hsevo.hsevo import HSEvo as LHH
from utils.utils import get_output_dir

ROOT_DIR = os.getcwd()
output_dir = get_output_dir("test_hsevo", ROOT_DIR)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    problem_name = "tsp_aco"

    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    # Main algorithm
    lhh = LHH(
        problem_name=problem_name,
        model="openai/gpt-4o-mini-2024-07-18",
        temperature=1.0,
        root_dir=ROOT_DIR,
        output_dir=output_dir,
    )
    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")


if __name__ == "__main__":
    main()
