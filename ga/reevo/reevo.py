from typing import Optional
import logging
import subprocess
import numpy as np
import os
from omegaconf import DictConfig

from ga.reevo.evolution import Evolution, ReEvoLLMClients
from utils.llm_client.base import BaseClient
from utils.problem import ProblemPrompts
from utils.utils import (
    extract_code_from_generator,
    filter_traceback,
    block_until_running,
    print_hyperlink,
)


class ReEvo:
    def __init__(
        self,
        cfg: DictConfig,
        root_dir: str,
        generator_llm: BaseClient,
        reflector_llm: Optional[BaseClient] = None,
        # Support setting different LLMs for each of the four operators:
        # Short-term Reflection, Long-term Reflection, Crossover, Mutation
        short_reflector_llm: Optional[BaseClient] = None,
        long_reflector_llm: Optional[BaseClient] = None,
        crossover_llm: Optional[BaseClient] = None,
        mutation_llm: Optional[BaseClient] = None,
    ) -> None:
        self.cfg = cfg

        self.root_dir = root_dir
        self.prompt_dir = f"{self.root_dir}/prompts"

        self.prompts = ProblemPrompts.load_problem_prompts(
            path=f"{self.prompt_dir}/{self.cfg.problem}",
        )

        self.output_file = (
            f"{self.root_dir}/problems/{self.prompts.problem_name}/gpt.py"
        )

        self.evol = Evolution(
            init_pop_size=self.cfg.init_pop_size,
            pop_size=self.cfg.pop_size,
            mutation_rate=self.cfg.mutation_rate,
            root_dir=self.root_dir,
            llm_clients=ReEvoLLMClients(
                generator_llm=generator_llm,
                reflector_llm=reflector_llm,
                short_reflector_llm=short_reflector_llm,
                long_reflector_llm=long_reflector_llm,
                crossover_llm=crossover_llm,
                mutation_llm=mutation_llm,
            ),
            prompts=self.prompts,
        )

        self.root_dir = root_dir

        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None

        self.init_population()

    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.prompts.seed_func).replace("v1", "v2")
        logging.info("Seed function code: \n" + code)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(
                f"Seed function is invalid. Please check the stdout file in {os.getcwd()}."
            )

        self.update_iter()

        # Generate responses
        responses = self.evol.seed_population(self.long_term_reflection_str)

        population = [
            self.response_to_individual(response, response_id)
            for response_id, response in enumerate(responses)
        ]

        # Run code and evaluate population
        population = self.evaluate_population(population)

        # Update iteration
        self.population = population
        self.update_iter()

    def response_to_individual(
        self, response: str, response_id: int, file_name: str = None
    ) -> dict:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = (
            f"problem_iter{self.iteration}_response{response_id}.txt"
            if file_name is None
            else file_name + ".txt"
        )
        with open(file_name, "w") as file:
            file.writelines(response + "\n")

        code = extract_code_from_generator(response)

        # Extract code and description from response
        std_out_filepath = (
            f"problem_iter{self.iteration}_stdout{response_id}.txt"
            if file_name is None
            else file_name + "_stdout.txt"
        )

        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
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

    def evaluate_population(self, population: list[dict]) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:
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
                    timeout=self.cfg.timeout
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

            individual = population[response_id]
            # Store objective value for each individual
            if traceback_msg == "":  # If execution has no error
                try:
                    individual["obj"] = (
                        float(stdout_str.split("\n")[-2])
                        if self.prompts.obj_type == "min"
                        else -float(stdout_str.split("\n")[-2])
                    )
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
        return population

    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")

        with open(self.output_file, "w") as file:
            file.writelines(individual["code"] + "\n")

        # Execute the python file with flags
        with open(individual["stdout_filepath"], "w") as f:
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
            individual["stdout_filepath"],
            log_status=True,
            iter_num=self.iteration,
            response_id=response_id,
        )
        return process

    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual["obj"] for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))

        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]

        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")

        best_path = self.best_code_path_overall.replace(".py", ".txt").replace(
            "code", "response"
        )
        logging.info(
            f"Best obj: {self.best_obj_overall}, Best Code Path: {print_hyperlink(best_path, self.best_code_path_overall)}"
        )
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1

    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank-based selection, select individuals with probability proportional to their rank.
        """
        if self.prompts.problem_type == "black_box":
            population = [
                individual
                for individual in population
                if individual["exec_success"]
                and individual["obj"] < self.seed_ind["obj"]
            ]
        else:
            population = [
                individual for individual in population if individual["exec_success"]
            ]
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x["obj"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.prompts.problem_type == "black_box":
            population = [
                individual
                for individual in population
                if individual["exec_success"]
                and individual["obj"] < self.seed_ind["obj"]
            ]
        else:
            population = [
                individual for individual in population if individual["exec_success"]
            ]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
        self.long_term_reflection_str = self.evol.long_term_reflection(
            short_term_reflections, self.long_term_reflection_str
        )

        # Write reflections to file
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, "w") as file:
            file.writelines("\n".join(short_term_reflections) + "\n")

        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, "w") as file:
            file.writelines(self.long_term_reflection_str + "\n")

    def crossover(
        self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]
    ) -> list[dict]:
        response_lst = self.evol.crossover(short_term_reflection_tuple)
        crossed_population = [
            self.response_to_individual(response, response_id)
            for response_id, response in enumerate(response_lst)
        ]

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population

    def mutate(self) -> list[dict]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        responses = self.evol.mutate(self.long_term_reflection_str, self.elitist)
        population = [
            self.response_to_individual(response, response_id)
            for response_id, response in enumerate(responses)
        ]
        return population

    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            # If all individuals are invalid, stop
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(
                    f"All individuals are invalid. Please check the stdout files in {os.getcwd()}."
                )
            # Select
            population_to_select = (
                self.population
                if (self.elitist is None or self.elitist in self.population)
                else [self.elitist] + self.population
            )  # add elitist to population for selection
            selected_population = self.random_select(population_to_select)
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
            # Short-term reflection
            short_term_reflection_tuple = self.evol.short_term_reflection(
                selected_population
            )  # (response_lst, worse_code_lst, better_code_lst)
            # Crossover
            crossed_population = self.crossover(short_term_reflection_tuple)
            # Evaluate
            self.population = self.evaluate_population(crossed_population)
            # Update
            self.update_iter()
            # Long-term reflection
            self.long_term_reflection(
                [response for response in short_term_reflection_tuple[0]]
            )
            # Mutate
            mutated_population = self.mutate()
            # Evaluate
            self.population.extend(self.evaluate_population(mutated_population))
            # Update
            self.update_iter()

        return self.best_code_overall, self.best_code_path_overall
