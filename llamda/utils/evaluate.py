from importlib.resources import files
import logging
import subprocess
from typing import TypeVar

from llamda.utils.individual import Individual
from llamda.utils.problem import BaseProblemPrompts
from llamda.utils.utils import block_until_running, filter_traceback

T = TypeVar("T", bound=Individual)


class Evaluator:
    def __init__(self, problem_prompts: BaseProblemPrompts):
        self.problem_prompts = problem_prompts

        problems_dir = files("llamda.problems")

        self.output_file = problems_dir / f"{self.problem_prompts.problem_name}/gpt.py"
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
                        if self.problem_prompts.obj_type == "max"
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
                f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual.obj}"
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

            problems_dir = files("llamda.problems")

            eval_file_path = (
                problems_dir / f"{self.problem_prompts.problem_name}/eval.py"
                if self.problem_prompts.problem_type != "black_box"
                else problems_dir
                / f"{self.problem_prompts.problem_name}/eval_black_box.py"
            )
            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    eval_file_path,
                    f"{self.problem_prompts.problem_size}",
                    str(problems_dir),
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
