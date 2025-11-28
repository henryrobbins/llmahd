import logging
import os
import subprocess
import re
from typing import Any

from ga.mcts.source.prompts.problem import ProblemPrompts
from utils.utils import block_until_running, file_to_string, filter_traceback


def adapt_prompt(problem_cfg: dict, root_dir: str):

    cfg = problem_cfg
    problem = problem_cfg.problem_name
    root_dir = root_dir
    problem_type = problem_cfg.problem_type
    prompt_dir = f"{root_dir}/prompts"

    prompt_path_suffix = "_black_box" if problem_type == "black_box" else ""
    problem_prompt_path = f"{prompt_dir}/{problem}{prompt_path_suffix}"
    func_signature = (
        file_to_string(f"{problem_prompt_path}/func_signature.txt")
        .format(version=2)
        .strip()
    )
    func_desc = file_to_string(f"{problem_prompt_path}/func_desc.txt")

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

    return ProblemPrompts(
        prompt_task=cfg.description,
        prompt_func_name=prompt_func_name,
        prompt_func_inputs=prompt_func_inputs,
        prompt_func_outputs=prompt_func_outputs,
        prompt_inout_inf=func_desc,
        prompt_other_inf="",
    )


class Problem:
    def __init__(self, cfg, root_dir):
        self.config = cfg
        self.root_dir = root_dir

        self.problem = self.config.problem.problem_name
        self.problem_description = self.config.problem.description
        self.problem_size = self.config.problem.problem_size
        self.obj_type = self.config.problem.obj_type
        self.problem_type = self.config.problem.problem_type
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"

        if self.problem_type == "tsp_constructive":
            from ga.mcts.source.prompts.problem import TSP_CONSTRUCTIVE_PROMPTS

            self.prompts = TSP_CONSTRUCTIVE_PROMPTS
        elif self.problem_type == "bpp_online":
            from ga.mcts.source.prompts.problem import BPP_ONLINE_PROMPTS

            self.prompts = BPP_ONLINE_PROMPTS
        else:
            self.prompts = adapt_prompt(self.config.problem, root_dir)

    def response_to_individual(self, code, response_id, file_name=None) -> dict:
        """
        Convert response to individual
        """
        outdir = "./evaluations/"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        runid = hash(code)
        # Write response to file
        file_name = (
            outdir + f"problem_eval{runid}.txt"
            if file_name is None
            else file_name + ".txt"
        )
        with open(file_name, "w", encoding="utf-8") as file:
            file.writelines(code + "\n")

        # Extract code and description from response
        std_out_filepath = (
            outdir + f"problem_eval{runid}_stdout.txt"
            if file_name is None
            else file_name.rstrip(".txt") + "_stdout.txt"
        )

        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": outdir + f"problem_eval{runid}_code.py",
            "code": code,
            "response_id": response_id,
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    def batch_evaluate(self, codes: list[str], iteration: int) -> str | list[float]:
        """
        Evaluate population by running code in parallel and computing objective values and fitness.
        """
        self.iteration = iteration
        population = [
            self.response_to_individual(resp, index) for index, resp in enumerate(codes)
        ]
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            runid = hash(population[response_id]["code"])
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], "Invalid response!"
                )
                inner_runs.append(None)
                continue

            logging.info(f"Iteration {self.iteration}: Running Code {runid}")
            individual = population[response_id]

            try:
                logging.debug(
                    f"Iteration {self.iteration}: Processing Code Run {runid}"
                )

                with open(self.output_file, "w", encoding="utf-8") as file:
                    file.writelines(individual["code"] + "\n")

                # Execute the python file with flags
                with open(individual["stdout_filepath"], "w") as f:
                    file_path = (
                        f"{self.root_dir}/problems/{self.problem}/eval.py"
                        if self.problem_type != "black_box"
                        else f"{self.root_dir}/problems/{self.problem}/eval_black_box.py"
                    )
                    inner_run = process = subprocess.Popen(
                        [
                            "python",
                            "-u",
                            file_path,
                            f"{self.problem_size}",
                            self.root_dir,
                            "train",
                        ],
                        stdout=f,
                        stderr=f,
                    )

                block_until_running(individual["stdout_filepath"], log_status=True)
                inner_runs.append(process)
            except Exception as e:  # If code execution fails
                print(e)
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )
                inner_runs.append(None)

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
                return "timeout"

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, "r") as f:  # read the stdout file
                stdout_str = f.read()
            traceback_msg = filter_traceback(stdout_str)

            # Store objective value and fitness for each individual
            if traceback_msg == "":  # If execution has no error
                try:
                    individual["obj"] = float(stdout_str.split("\n")[-2])
                    assert (
                        individual["obj"] > 0
                    ), "Objective value <= 0 is not supported."
                    individual["obj"] = (
                        -individual["obj"]
                        if self.obj_type == "max"
                        else individual["obj"]
                    )
                    # individual["fitness"] = 1 / individual["obj"] if self.obj_type == "min" else individual["obj"]
                    individual["exec_success"] = True
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
        return [indiv["obj"] for indiv in population]
