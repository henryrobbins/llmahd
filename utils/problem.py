import os
from dataclasses import dataclass

from utils.utils import file_to_string


@dataclass
class ProblemPrompts:
    func_name: str
    problem_desc: str
    seed_func: str
    func_signature: str
    func_desc: str
    external_knowledge: str

    def load_problem_prompts(
        func_name: str, problem_desc: str, path: str
    ) -> "ProblemPrompts":

        seed_func = file_to_string(f"{path}/seed_func.txt")
        func_signature = file_to_string(f"{path}/func_signature.txt")
        func_desc = file_to_string(f"{path}/func_desc.txt")
        if os.path.exists(f"{path}/external_knowledge.txt"):
            external_knowledge = file_to_string(f"{path}/external_knowledge.txt")
        else:
            external_knowledge = ""

        return ProblemPrompts(
            func_name=func_name,
            problem_desc=problem_desc,
            seed_func=seed_func,
            func_signature=func_signature,
            func_desc=func_desc,
            external_knowledge=external_knowledge,
        )
