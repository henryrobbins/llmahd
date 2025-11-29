from dataclasses import dataclass, field


@dataclass
class Config:

    # MCTS configuration
    pop_size: int = 10  # Size of Elite set E, default = 10
    init_size: int = 4  # Number of initial nodes N_I, default = 4
    ec_fe_max: int = 1000  # Number of evaluations, default = 1000
    ec_operators: list[str] = field(
        default_factory=lambda: ["e1", "e2", "m1", "m2", "s1"]
    )  # evolution operators
    ec_m: int = 2
    ec_operator_weights: list[int] = field(
        default_factory=lambda: [0, 1, 2, 2, 1]
    )  # weights for operators default

    # Exp settings
    exp_output_path: str = "/"  # default folder for ael outputs
