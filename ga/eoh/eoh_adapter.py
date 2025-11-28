from utils.llm_client.base import BaseClient
from .original.eoh import EOH
from .original.config import Config
from .problem_adapter import Problem

from utils.utils import init_client


class EoH:
    def __init__(self, cfg, root_dir: str, client: BaseClient) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)

        self.paras = Config(
            ec_pop_size=self.cfg.pop_size,
            ec_n_pop=(self.cfg.max_fe - 2 * self.cfg.pop_size)
            // (4 * self.cfg.pop_size)
            + 1,  # total evals = 2 * pop_size + n_pop * 4 * pop_size; for pop_size = 10, n_pop = 5, total evals = 2 * 10 + 4 * 5 * 10 = 220
            exp_output_path="./",
        )

        self.llm_client = init_client(self.cfg)

    def evolve(self):
        print("- Evolution Start -")

        method = EOH(self.paras, self.problem, self.llm_client)

        results = method.run()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")

        return results
