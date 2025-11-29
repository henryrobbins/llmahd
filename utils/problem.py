import logging
import os
import re
import subprocess
from typing import TypeVar
import yaml
from dataclasses import dataclass

from utils.individual import Individual
from utils.utils import block_until_running, file_to_string, filter_traceback


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


class Problem:
    def __init__(self, problem_name: str, root_dir: str):
        self.root_dir = root_dir

        self.problem_config = ProblemPrompts.load_problem_prompts(
            f"{self.root_dir}/prompts/{problem_name}"
        )

        self.output_file = (
            f"{self.root_dir}/problems/{self.problem_config.problem_name}/gpt.py"
        )

        if self.problem_config.problem_type == "constructive":
            from utils.problem import TSP_CONSTRUCTIVE_PROMPTS

            self.prompts = TSP_CONSTRUCTIVE_PROMPTS
        elif self.problem_config.problem_type == "online":
            from utils.problem import BPP_ONLINE_PROMPTS

            self.prompts = BPP_ONLINE_PROMPTS
        else:
            self.prompts = adapt_prompt(self.problem_config)

        self.function_evals = 0

    def mark_invalid_individual(self, individual: T, traceback_msg: str) -> T:
        """
        Mark an individual as invalid.
        """
        individual.exec_success = False
        individual.obj = float("inf")
        individual.traceback_msg = traceback_msg
        return individual

    def batch_evaluate(self, population: list[T], iteration: int = 0) -> list[T]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        self.iteration = iteration

        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id].code is None:
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], "Invalid response!"
                )
                inner_runs.append(None)
                continue

            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")

            try:
                process = self._run_code(population[response_id], response_id)
                inner_runs.append(process)
            except Exception as e:  # If code execution fails
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )
                inner_runs.append(None)

        # Update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None:  # If code execution fails, skip
                continue
            try:
                inner_run.communicate(
                    timeout=self.config.timeout
                )  # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual.stdout_filepath
            with open(stdout_filepath, "r") as f:  # read the stdout file
                stdout_str = f.read()
            traceback_msg = filter_traceback(stdout_str)

            # Store objective value for each individual
            if traceback_msg == "":  # If execution has no error
                try:
                    individual.obj = float(stdout_str.split("\n")[-2])
                    assert individual.obj > 0, "Objective value <= 0 is not supported."
                    individual.obj = (
                        -individual.obj
                        if self.problem_config.obj_type == "max"
                        else individual.obj
                    )
                    individual.exec_success = True
                except:
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id], "Invalid std out / objective value!"
                    )
            else:  # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], traceback_msg
                )

            logging.info(
                f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}"
            )
        return population

    def _run_code(self, individual: T, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")

        with open(self.output_file, "w") as file:
            file.writelines(individual.code + "\n")

        # Execute the python file with flags
        with open(individual.stdout_filepath, "w") as f:
            eval_file_path = (
                f"{self.root_dir}/problems/{self.prompts.problem_name}/eval.py"
                if self.prompts.problem_type != "black_box"
                else f"{self.root_dir}/problems/{self.prompts.problem_name}/eval_black_box.py"
            )
            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    eval_file_path,
                    f"{self.prompts.problem_size}",
                    self.root_dir,
                    "train",
                ],
                stdout=f,
                stderr=f,
            )

        block_until_running(
            individual.stdout_filepath,
            log_status=True,
            iter_num=self.iteration,
            response_id=response_id,
        )
        return process


def hydrate_individual(
    individual: T, response_id, iteration: int = 0, file_name=None
) -> T:

    # Write response to file
    file_name = (
        f"problem_iter{iteration}_response{response_id}.txt"
        if file_name is None
        else file_name + ".txt"
    )
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
