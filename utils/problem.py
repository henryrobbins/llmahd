import os
import yaml
from dataclasses import dataclass

from utils.utils import file_to_string


@dataclass
class ProblemPrompts:
    problem_name: str
    problem_type: str
    obj_type: str
    problem_size: int
    func_name: str
    problem_desc: str
    seed_func: str
    func_signature: str
    func_desc: str
    external_knowledge: str

    def load_problem_prompts(path: str) -> "ProblemPrompts":

        with open(f"{path}/problem.yaml", "r") as f:
            config = yaml.safe_load(f)
        seed_func = file_to_string(f"{path}/seed_func.txt")
        func_signature = file_to_string(f"{path}/func_signature.txt")
        func_desc = file_to_string(f"{path}/func_desc.txt")
        if os.path.exists(f"{path}/external_knowledge.txt"):
            external_knowledge = file_to_string(f"{path}/external_knowledge.txt")
        else:
            external_knowledge = ""

        return ProblemPrompts(
            problem_name=config["problem_name"],
            problem_type=config["problem_type"],
            obj_type=config["obj_type"],
            problem_size=config["problem_size"],
            func_name=config["func_name"],
            problem_desc=config["description"],
            seed_func=seed_func,
            func_signature=func_signature,
            func_desc=func_desc,
            external_knowledge=external_knowledge,
        )
