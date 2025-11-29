from ga.mcts.source.mcts_ahd import MCTS_AHD
from ga.mcts.source.config import Config
from ga.mcts.problem_adapter import Problem


class AHD:
    def __init__(self, cfg, root_dir, workdir, client) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)

        self.paras = Config(
            init_size=self.cfg.init_pop_size,
            pop_size=self.cfg.pop_size,
            ec_fe_max=self.cfg.max_fe,
            exp_output_path=f"{workdir}/",
        )

        self.llm_client = client

    def evolve(self):
        print("- Evolution Start -")

        method = MCTS_AHD(self.paras, self.problem, self.llm_client)

        results = method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------")
        print("---  MCTS-AHD successfully finished!  ---")
        print("-----------------------------------------")

        return results
