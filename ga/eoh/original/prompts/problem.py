from dataclasses import dataclass


@dataclass
class ProblemPrompts:
    prompt_task: str
    prompt_func_name: str
    prompt_func_inputs: list[str]
    prompt_func_outputs: list[str]
    prompt_inout_inf: str
    prompt_other_inf: str


BPP_ONLINE_PROMPTS = ProblemPrompts(
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

TSP_CONSTRUCTIVE_PROMPTS = ProblemPrompts(
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
