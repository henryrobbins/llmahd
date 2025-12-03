# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/problem_adapter.py
# and https://github.com/ai4co/reevo/blob/main/reevo.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import os
import logging
from pathlib import Path
import subprocess
from typing import TypeVar

from llamda.individual import Individual
from llamda.problem import BaseProblem

logger = logging.getLogger("llamda")

T = TypeVar("T", bound=Individual)


class Evaluator:
    def __init__(self, problem: BaseProblem, timeout: int = 30) -> None:
        self.problem = problem
        self.timeout = timeout
        self.function_evals = 0

    def _logging_context(self) -> dict:
        return {
            "problem_name": self.problem.name,
            "function_evals": self.function_evals,
        }

    def mark_invalid_individual(self, individual: T, traceback_msg: str) -> T:
        """
        Mark an individual as invalid.
        """
        logger.debug(
            "Marking individual as invalid", extra={"individual_name": individual.name}
        )
        individual.exec_success = False
        individual.obj = float("inf")
        individual.traceback_msg = traceback_msg
        return individual

    def batch_evaluate(self, population: list[T], output_dir: Path) -> list[T]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        logger.info(
            "Starting batch evaluation",
            extra={
                **self._logging_context(),
                "population_size": len(population),
                "population_names": [ind.name for ind in population],
            },
        )

        inner_runs = []
        # Run code to evaluate
        for i, individual in enumerate(population):

            logger.info(
                f"Evaluating individual [{i}/{len(population)-1}]",
                extra={**self._logging_context(), "individual_name": individual.name},
            )

            stdout_filepath = (
                output_dir / "individuals" / individual.name / "stdout.txt"
            )
            os.makedirs(stdout_filepath.parent, exist_ok=True)
            self.function_evals += 1
            # Skip if response is invalid
            if individual.code is None:
                logger.debug(
                    "There is no code to run for this individual.",
                    extra={
                        **self._logging_context(),
                        "individual_name": individual.name,
                    },
                )
                individual = self.mark_invalid_individual(
                    individual, "Invalid response!"
                )
                inner_runs.append(None)
                continue

            try:
                process = self._run_code(individual)
                inner_runs.append(process)
            except Exception as e:  # If code execution fails
                logger.exception(
                    "Failed to run code for individual.",
                    extra={
                        **self._logging_context(),
                        "individual_name": individual.name,
                    },
                )
                individual = self.mark_invalid_individual(individual, str(e))
                inner_runs.append(None)

        # Update population with objective values
        for individual, inner_run in zip(population, inner_runs):
            if inner_run is None:  # If code execution fails, skip
                continue
            try:
                stdout_str, _ = inner_run.communicate(timeout=self.timeout)
                stdout_filepath = (
                    output_dir / "individuals" / individual.name / "stdout.txt"
                )
                os.makedirs(stdout_filepath.parent, exist_ok=True)
                with open(stdout_filepath, "w") as f:
                    f.write(stdout_str)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Timeout expired during code execution",
                    extra={
                        **self._logging_context(),
                        "timeout": self.timeout,
                        "individual_name": individual.name,
                    },
                )
                individual = self.mark_invalid_individual(
                    individual, "Timeout expired during code execution"
                )
                inner_run.kill()
                continue

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
                except Exception:
                    logger.exception(
                        "Failed to parse objective value",
                        extra={
                            **self._logging_context(),
                            "individual_name": individual.name,
                        },
                    )
                    individual = self.mark_invalid_individual(
                        individual, "Invalid std out / objective value!"
                    )
            else:  # Otherwise, also provide execution traceback error feedback
                logger.warning(
                    "Code evaluation of individual failed with traceback",
                    extra={
                        **self._logging_context(),
                        "individual_name": individual.name,
                        "traceback": traceback_msg,
                    },
                )
                individual = self.mark_invalid_individual(individual, traceback_msg)

            logger.debug(
                "Individual evaluated successfully",
                extra={
                    **self._logging_context(),
                    "individual_name": individual.name,
                    "objective": individual.obj,
                },
            )

        logger.info(
            "Batch evaluation completed",
            extra={
                **self._logging_context(),
                "successful_individuals": sum(
                    1 for ind in population if ind.exec_success
                ),
                "failed_individuals": sum(
                    1 for ind in population if not ind.exec_success
                ),
            },
        )
        return population

    def _run_code(self, individual: T) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """

        with open(self.problem.code_path, "w") as file:
            file.writelines(individual.code + "\n")

        process = subprocess.Popen(
            [
                "python",
                "-u",
                str(self.problem.eval_path),
                f"{self.problem.size}",
                "train",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        return process


def filter_traceback(s: str) -> str:
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
    return ""  # Return an empty string if no Traceback is found
