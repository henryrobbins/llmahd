from dataclasses import dataclass


@dataclass
class Config:

    # EC settings
    ec_pop_size: int = 5  # number of algorithms in each population
    ec_n_pop: int = 5  # number of populations
    ec_operators: list[str] = ["e1", "e2", "m1", "m2"]  # evolution operators
    ec_m: int = 2  # number of parents for 'e1' and 'e2' operators
    ec_operator_weights: list[int] = [1, 1, 1, 1]  # weights for operators

    # Exp settings
    exp_output_path: str = "./"  # default folder for ael outputs
    exp_use_seed: bool = False
    exp_seed_path: str = "./seeds/seeds.json"
    exp_use_continue: bool = False
    exp_continue_id: int = 0
    exp_continue_path: str = "./results/pops/population_generation_0.json"
