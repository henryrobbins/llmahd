import os
import re
from typing import TypeVar
import yaml
from dataclasses import dataclass

from llamda.utils.individual import Individual
from llamda.utils.utils import file_to_string


T = TypeVar("T", bound=Individual)


@dataclass
class BaseProblemPrompts:
    problem_name: str
    problem_type: str
    obj_type: str
    problem_size: int
    problem_desc: str
    func_name: str


@dataclass
class ProblemPrompts(BaseProblemPrompts):
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


@dataclass
class EOHProblemPrompts(BaseProblemPrompts):
    func_inputs: list[str]
    func_outputs: list[str]
    inout_inf: str
    other_inf: str

# I believe these were the exact prompts used in the original EOH paper
# See the reevo EoH baseline implementation for reference:
# https://github.com/ai4co/reevo/blob/main/baselines/eoh/original/prompts/bpp_online.py
BPP_ONLINE_PROMPTS = EOHProblemPrompts(
    problem_name="bpp_online",
    problem_type="online",
    obj_type="min",
    problem_size=5000,
    problem_desc="I need help designing a novel score function that scoring a set \
of bins to assign an item. In each step, the item will be assigned to the bin with \
the maximum score. If the rest capacity of a bin equals the maximum capacity, it \
will not be used. The final goal is to minimize the number of used bins.",
    func_name="score",
    func_inputs=["item", "bins"],
    func_outputs=["scores"],
    inout_inf="'item' and 'bins' are the size of current item and the rest \
capacities of feasible bins, which are larger than the item size. The output named \
'scores' is the scores for the bins for assignment. ",
    other_inf="Note that 'item' is of type int, while 'bins' and 'scores' \
are both Numpy arrays. The novel function should be sufficiently complex in order \
to achieve better performance. It is important to ensure self-consistency.",
)

# I believe these were the exact prompts used in the original EOH paper
# See the reevo EoH baseline implementation for reference:
# https://github.com/ai4co/reevo/blob/main/baselines/eoh/original/prompts/tsp_greedy.py
TSP_CONSTRUCTIVE_PROMPTS = EOHProblemPrompts(
    problem_name="tsp_constructive",
    problem_type="constructive",
    obj_type="min",
    problem_size=50,
    problem_desc="Given a set of nodes with their coordinates, you need to find the \
shortest route that visits each node once and returns to the starting node. The task \
can be solved step-by-step by starting from the current node and iteratively choosing \
the next node. Help me design a novel algorithm that is different from the algorithms \
in literature to select the next node in each step.",
    func_name="select_next_node",
    func_inputs=[
        "current_node",
        "destination_node",
        "univisited_nodes",
        "distance_matrix",
    ],
    func_outputs=["next_node"],
    inout_inf="'current_node', 'destination_node', 'next_node', and \
'unvisited_nodes' are node IDs. 'distance_matrix' is the distance matrix of nodes.",
    other_inf="All are Numpy arrays.",
)


def adapt_prompt(prompts: ProblemPrompts) -> EOHProblemPrompts:

    match = re.match(r"^def +(.+?)\((.*)\) *-> *(.*?) *:", prompts.func_signature)
    assert match is not None
    prompt_func_name = prompts.func_name
    prompt_func_inputs = [
        txt.split(":")[0].strip() for txt in match.group(2).split(",")
    ]
    if prompt_func_name.startswith("select_next_node"):
        prompt_func_outputs = ["next_node"]
    elif prompt_func_name.startswith("priority"):
        prompt_func_outputs = ["priority"]
    elif prompt_func_name.startswith("heuristics"):
        prompt_func_outputs = ["heuristics_matrix"]
    elif prompt_func_name.startswith("crossover"):
        prompt_func_outputs = ["offsprings"]
    elif prompt_func_name.startswith("utility"):
        prompt_func_outputs = ["utility_value"]
    else:
        prompt_func_outputs = ["result"]

    return EOHProblemPrompts(
        problem_name=prompts.problem_name,
        problem_type=prompts.problem_type,
        obj_type=prompts.obj_type,
        problem_size=prompts.problem_size,
        problem_desc=prompts.problem_desc,
        func_name=prompt_func_name,
        func_inputs=prompt_func_inputs,
        func_outputs=prompt_func_outputs,
        inout_inf=prompts.func_desc,
        other_inf="",
    )


def hydrate_individual(
    individual: T, response_id, output_dir: str, iteration: int = 0, file_name=None
) -> T:

    # Write response to file
    file_name = (
        f"problem_iter{iteration}_response{response_id}.txt"
        if file_name is None
        else file_name + ".txt"
    )
    file_name = f"{output_dir}/{file_name}"
    with open(file_name, "w", encoding="utf-8") as file:
        file.writelines(individual.code + "\n")

    # Extract code and description from response
    std_out_filepath = (
        f"problem_iter{iteration}_stdout{response_id}.txt"
        if file_name is None
        else file_name.rstrip(".txt") + "_stdout.txt"
    )

    individual.stdout_filepath = std_out_filepath
    individual.code_path = f"problem_iter{iteration}_code{response_id}.py"
    individual.response_id = response_id

    return individual
