from dataclasses import dataclass

from ga.mcts.mcts_ahd import MCTS_AHD
from ga.mcts.config import Config
from utils.evaluate import Evaluator
from utils.problem import ProblemPrompts, adapt_prompt


@dataclass
class AHDConfig:
    max_fe: int = 1000  # maximum number of function evaluations
    pop_size: int = 10  # population size for GA
    init_pop_size: int = 4  # initial population size for GA
    timeout: int = 60  # timeout for evaluation of a single heuristic


class AHD:
    def __init__(self, problem_name: str, root_dir, workdir, client) -> None:

        self.root_dir = root_dir

        self.problem_config = ProblemPrompts.load_problem_prompts(
            f"{self.root_dir}/prompts/{problem_name}"
        )

        self.output_file = (
            f"{self.root_dir}/problems/{self.problem_config.problem_name}/gpt.py"
        )

        if self.problem_config.problem_type == "constructive":
            from utils.problem import TSP_CONSTRUCTIVE_PROMPTS

            self.prompts = TSP_CONSTRUCTIVE_PROMPTS
        elif self.problem_config.problem_type == "online":
            from utils.problem import BPP_ONLINE_PROMPTS

            self.prompts = BPP_ONLINE_PROMPTS
        else:
            self.prompts = adapt_prompt(self.problem_config)

        self.evaluator = Evaluator(self.prompts, self.root_dir)

        ahd_config = AHDConfig()

        self.paras = Config(
            init_size=ahd_config.init_pop_size,
            pop_size=ahd_config.pop_size,
            ec_fe_max=ahd_config.max_fe,
            exp_output_path=f"{workdir}/",
        )

        self.llm_client = client

    def evolve(self):
        print("- Evolution Start -")

        method = MCTS_AHD(self.paras, self.prompts, self.evaluator, self.llm_client)

        results = method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------")
        print("---  MCTS-AHD successfully finished!  ---")
        print("-----------------------------------------")

        return results
