from importlib.resources import files
import logging
import os
from pathlib import Path
import re
import yaml
from enum import Enum
from dataclasses import dataclass

from llamda.utils import file_to_string

logger = logging.getLogger("llamda")


class ProblemName(Enum):
    BPP_OFFLINE_ACO = "bpp_offline_aco"
    BPP_ONLINE = "bpp_online"
    CVRP_ACO = "cvrp_aco"
    MKP_ACO = "mkp_aco"
    TSP_ACO = "tsp_aco"
    TSP_CONSTRUCTIVE = "tsp_constructive"


@dataclass
class BaseProblem:
    name: str
    type: str
    obj_type: str
    size: int
    description: str
    func_name: str
    path: Path

    @property
    def eval_path(self) -> Path:
        return self.path / "eval.py"

    @property
    def code_path(self) -> Path:
        return self.path / "gpt.py"


@dataclass
class Problem(BaseProblem):
    seed_func: str
    func_signature: str
    func_desc: str
    external_knowledge: str

    @classmethod
    def load_problem(cls, path: Path) -> "Problem":

        prompts_path = path / "prompts"
        with open(prompts_path / "problem.yaml", "r") as f:
            config = yaml.safe_load(f)
        seed_func = file_to_string(prompts_path / "seed_func.txt")
        func_signature = file_to_string(prompts_path / "func_signature.txt")
        func_desc = file_to_string(prompts_path / "func_desc.txt")
        if os.path.exists(prompts_path / "external_knowledge.txt"):
            external_knowledge = file_to_string(prompts_path / "external_knowledge.txt")
            logger.debug("External knowledge file found and loaded")
        else:
            external_knowledge = ""
            logger.debug("No external knowledge file found")

        return cls(
            name=config["problem_name"],
            type=config["problem_type"],
            obj_type=config["obj_type"],
            size=config["problem_size"],
            func_name=config["func_name"],
            description=config["description"],
            path=path,
            seed_func=seed_func,
            func_signature=func_signature,
            func_desc=func_desc,
            external_knowledge=external_knowledge,
        )

    @classmethod
    def load_builtin(cls, problem_name: ProblemName) -> "Problem":

        problem_path = files("llamda.problems") / problem_name.value
        return cls.load_problem(problem_path)


@dataclass
class EohProblem(BaseProblem):
    func_inputs: list[str]
    func_outputs: list[str]
    inout_info: str
    other_info: str


# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/original/prompts/bpp_online.py
# Originally from EoH: https://github.com/FeiLiu36/EoH/blob/main/eoh/src/eoh/problems/optimization/bp_online/prompts.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)
BPP_ONLINE_PROMPTS = EohProblem(
    name="bpp_online",
    type="online",
    obj_type="min",
    size=5000,
    description="I need help designing a novel score function that scoring a set \
of bins to assign an item. In each step, the item will be assigned to the bin with \
the maximum score. If the rest capacity of a bin equals the maximum capacity, it \
will not be used. The final goal is to minimize the number of used bins.",
    path=files("llamda.problems") / ProblemName.BPP_ONLINE.value,
    func_name="score",
    func_inputs=["item", "bins"],
    func_outputs=["scores"],
    inout_info="'item' and 'bins' are the size of current item and the rest \
capacities of feasible bins, which are larger than the item size. The output named \
'scores' is the scores for the bins for assignment. ",
    other_info="Note that 'item' is of type int, while 'bins' and 'scores' \
are both Numpy arrays. The novel function should be sufficiently complex in order \
to achieve better performance. It is important to ensure self-consistency.",
)

# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/original/prompts/tsp_greedy.py
# Originally from EoH: https://github.com/FeiLiu36/EoH/blob/main/eoh/src/eoh/problems/optimization/tsp_greedy/prompts.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)
TSP_CONSTRUCTIVE_PROMPTS = EohProblem(
    name="tsp_constructive",
    type="constructive",
    obj_type="min",
    size=50,
    description="Given a set of nodes with their coordinates, you need to find the \
shortest route that visits each node once and returns to the starting node. The task \
can be solved step-by-step by starting from the current node and iteratively choosing \
the next node. Help me design a novel algorithm that is different from the algorithms \
in literature to select the next node in each step.",
    path=files("llamda.problems") / ProblemName.TSP_CONSTRUCTIVE.value,
    func_name="select_next_node",
    func_inputs=[
        "current_node",
        "destination_node",
        "univisited_nodes",
        "distance_matrix",
    ],
    func_outputs=["next_node"],
    inout_info="'current_node', 'destination_node', 'next_node', and \
'unvisited_nodes' are node IDs. 'distance_matrix' is the distance matrix of nodes.",
    other_info="All are Numpy arrays.",
)


# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/problem_adapter.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)
def adapt_prompt(problem: Problem) -> EohProblem:

    match = re.match(r"^def +(.+?)\((.*)\) *-> *(.*?) *:", problem.func_signature)
    assert match is not None
    func_name = problem.func_name
    func_inputs = [txt.split(":")[0].strip() for txt in match.group(2).split(",")]
    if func_name.startswith("select_next_node"):
        func_outputs = ["next_node"]
    elif func_name.startswith("priority"):
        func_outputs = ["priority"]
    elif func_name.startswith("heuristics"):
        func_outputs = ["heuristics_matrix"]
    elif func_name.startswith("crossover"):
        func_outputs = ["offsprings"]
    elif func_name.startswith("utility"):
        func_outputs = ["utility_value"]
    else:
        func_outputs = ["result"]

    logger.debug(
        "Prompt adapted",
        extra={
            "func_name": func_name,
            "func_inputs": func_inputs,
            "func_outputs": func_outputs,
        },
    )

    return EohProblem(
        name=problem.name,
        type=problem.type,
        obj_type=problem.obj_type,
        size=problem.size,
        description=problem.description,
        path=problem.path,
        func_name=func_name,
        func_inputs=func_inputs,
        func_outputs=func_outputs,
        inout_info=problem.func_desc,
        other_info="",
    )
