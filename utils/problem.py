import os
import re
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


@dataclass
class EOHProblemPrompts:
    prompt_task: str
    prompt_func_name: str
    prompt_func_inputs: list[str]
    prompt_func_outputs: list[str]
    prompt_inout_inf: str
    prompt_other_inf: str


BPP_ONLINE_PROMPTS = EOHProblemPrompts(
    prompt_task="I need help designing a novel score function that scoring a set \
of bins to assign an item. In each step, the item will be assigned to the bin with \
the maximum score. If the rest capacity of a bin equals the maximum capacity, it \
will not be used. The final goal is to minimize the number of used bins.",
    prompt_func_name="score",
    prompt_func_inputs=["item", "bins"],
    prompt_func_outputs=["scores"],
    prompt_inout_inf="'item' and 'bins' are the size of current item and the rest \
capacities of feasible bins, which are larger than the item size. The output named \
'scores' is the scores for the bins for assignment. ",
    prompt_other_inf="Note that 'item' is of type int, while 'bins' and 'scores' \
are both Numpy arrays. The novel function should be sufficiently complex in order \
to achieve better performance. It is important to ensure self-consistency.",
)

TSP_CONSTRUCTIVE_PROMPTS = EOHProblemPrompts(
    prompt_task="Given a set of nodes with their coordinates, you need to find the \
shortest route that visits each node once and returns to the starting node. The task \
can be solved step-by-step by starting from the current node and iteratively choosing \
the next node. Help me design a novel algorithm that is different from the algorithms \
in literature to select the next node in each step.",
    prompt_func_name="select_next_node",
    prompt_func_inputs=[
        "current_node",
        "destination_node",
        "univisited_nodes",
        "distance_matrix",
    ],
    prompt_func_outputs=["next_node"],
    prompt_inout_inf="'current_node', 'destination_node', 'next_node', and \
'unvisited_nodes' are node IDs. 'distance_matrix' is the distance matrix of nodes.",
    prompt_other_inf="All are Numpy arrays.",
)


def adapt_prompt(prompts: ProblemPrompts) -> EOHProblemPrompts:

    func_signature = prompts.func_signature
    func_desc = prompts.func_desc

    match = re.match(r"^def +(.+?)\((.*)\) *-> *(.*?) *:", func_signature)
    assert match is not None
    prompt_func_name = match.group(1)
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
        prompt_task=prompts.problem_desc,
        prompt_func_name=prompt_func_name,
        prompt_func_inputs=prompt_func_inputs,
        prompt_func_outputs=prompt_func_outputs,
        prompt_inout_inf=func_desc,
        prompt_other_inf="",
    )
