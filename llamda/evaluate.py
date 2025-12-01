# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/problem_adapter.py
# and https://github.com/ai4co/reevo/blob/main/reevo.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging
import subprocess
from typing import TypeVar

from llamda.individual import Individual
from llamda.problem import BaseProblem
from llamda.utils import block_until_running, filter_traceback

logger = logging.getLogger("llamda")

T = TypeVar("T", bound=Individual)


class Evaluator:
    def __init__(self, problem: BaseProblem, timeout: int = 30) -> None:
        self.problem = problem
        self.timeout = timeout
        self.function_evals = 0

    def mark_invalid_individual(self, individual: T, traceback_msg: str) -> T:
        """
        Mark an individual as invalid.
        """
        logger.warning(
            "Marking individual as invalid",
            extra={
                "response_id": individual.response_id,
                "traceback_msg": traceback_msg[:200],  # Truncate long messages
            },
        )
        individual.exec_success = False
        individual.obj = float("inf")
        individual.traceback_msg = traceback_msg
        return individual

    def batch_evaluate(self, population: list[T], iteration: int = 0) -> list[T]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        self.iteration = iteration
        logger.info(
            "Starting batch evaluation",
            extra={
                "iteration": iteration,
                "population_size": len(population),
                "function_evals": self.function_evals,
            },
        )

        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id].code is None:
                logger.debug(
                    "Skipping invalid response",
                    extra={"response_id": response_id, "iteration": iteration},
                )
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], "Invalid response!"
                )
                inner_runs.append(None)
                continue

            logger.info(f"Iteration {self.iteration}: Running Code {response_id}")

            try:
                process = self._run_code(population[response_id], response_id)
                inner_runs.append(process)
            except Exception as e:  # If code execution fails
                logger.exception(
                    "Code execution error",
                    extra={"response_id": response_id, "iteration": iteration},
                )
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
                    timeout=self.timeout
                )  # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logger.error(
                    "Timeout expired during code execution",
                    extra={
                        "response_id": response_id,
                        "iteration": self.iteration,
                        "timeout": self.timeout,
                    },
                )
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
                        if self.problem.obj_type == "max"
                        else individual.obj
                    )
                    individual.exec_success = True
                except Exception as e:
                    logger.error(
                        "Failed to parse objective value",
                        extra={
                            "response_id": response_id,
                            "iteration": self.iteration,
                            "error": str(e),
                        },
                    )
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id], "Invalid std out / objective value!"
                    )
            else:  # Otherwise, also provide execution traceback error feedback
                logger.debug(
                    "Individual execution failed with traceback",
                    extra={
                        "response_id": response_id,
                        "iteration": self.iteration,
                        "traceback": traceback_msg,
                    },
                )
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], traceback_msg
                )

            logger.debug(
                "Individual evaluated successfully",
                extra={
                    "response_id": response_id,
                    "iteration": self.iteration,
                    "objective": individual.obj,
                },
            )

        logger.info(
            "Batch evaluation completed",
            extra={
                "iteration": iteration,
                "function_evals": self.function_evals,
                "valid_individuals": sum(1 for ind in population if ind.exec_success),
            },
        )
        return population

    def _run_code(self, individual: T, response_id: int) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logger.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")

        with open(self.problem.code_path, "w") as file:
            file.writelines(individual.code + "\n")

        # Execute the python file with flags
        with open(individual.stdout_filepath, "w") as f:

            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    str(self.problem.eval_path),
                    f"{self.problem.size}",
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
