from dataclasses import dataclass

from utils.evaluate import Evaluator
from utils.llm_client.base import BaseClient
from ga.eoh.eoh import EOH
from ga.eoh.config import Config
from utils.problem import ProblemPrompts, adapt_prompt


@dataclass
class EoHConfig:
    max_fe: int = 100  # maximum number of function evaluations
    pop_size: int = 10  # population size for GA
    init_pop_size: int = 30  # initial population size for GA
    mutation_rate: float = 0.5  # mutation rate for GA
    timeout: int = 20  # timeout for evaluation of a single heuristic
    diversify_init_pop: bool = True  # whether to diversify the initial population


class EoH:
    def __init__(self, problem_name: str, root_dir: str, client: BaseClient) -> None:

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

        eoh_config = EoHConfig()

        self.paras = Config(
            ec_pop_size=eoh_config.pop_size,
            ec_n_pop=(eoh_config.max_fe - 2 * eoh_config.pop_size)
            // (4 * eoh_config.pop_size)
            + 1,  # total evals = 2 * pop_size + n_pop * 4 * pop_size; for pop_size = 10, n_pop = 5, total evals = 2 * 10 + 4 * 5 * 10 = 220
            exp_output_path="./",
        )

        self.llm_client = client

    def evolve(self):
        print("- Evolution Start -")

        method = EOH(self.paras, self.prompts, self.evaluator, self.llm_client)

        results = method.run()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")

        return results
