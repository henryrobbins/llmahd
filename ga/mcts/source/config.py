from dataclasses import dataclass


@dataclass
class Config:

    # MCTS configuration
    pop_size = 10  # Size of Elite set E, default = 10
    init_size = 4  # Number of initial nodes N_I, default = 4
    ec_fe_max = 1000  # Number of evaluations, default = 1000
    ec_operators = ["e1", "e2", "m1", "m2", "s1"]
    ec_m = 2
    ec_operator_weights = [0, 1, 2, 2, 1]  # weights for operators default

    # Exp settings
    exp_output_path = "/"  # default folder for ael outputs
