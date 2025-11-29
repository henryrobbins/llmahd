import logging
import os
import subprocess

from utils.problem import ProblemPrompts, adapt_prompt
from utils.utils import block_until_running, filter_traceback


class Problem:
    def __init__(self, problem_name, root_dir):
        self.problem_name = problem_name
        self.root_dir = root_dir

        self.problem_config = ProblemPrompts.load_problem_prompts(
            f"{self.root_dir}/prompts/{problem_name}"
        )

        self.problem = self.problem_config.problem_name
        self.problem_description = self.problem_config.problem_desc
        self.problem_size = self.problem_config.problem_size
        self.obj_type = self.problem_config.obj_type
        self.problem_type = self.problem_config.problem_type
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"

        if self.problem_type == "tsp_constructive":
            from utils.problem import TSP_CONSTRUCTIVE_PROMPTS

            self.prompts = TSP_CONSTRUCTIVE_PROMPTS
        elif self.problem_type == "bpp_online":
            from utils.problem import BPP_ONLINE_PROMPTS

            self.prompts = BPP_ONLINE_PROMPTS
        else:
            self.prompts = adapt_prompt(self.problem_config)

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
        with open(file_name, "w") as file:
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

    def batch_evaluate(self, codes: list[str], iteration: int) -> list[dict]:
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

                with open(self.output_file, "w") as file:
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
                continue

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
