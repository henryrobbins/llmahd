# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/reevo.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

from typing import Optional
import logging
import numpy as np
import os
from dataclasses import dataclass

from llamda.ga.base import GeneticAlgorithm
from llamda.ga.reevo.evolution import Evolution
from llamda.evaluate import Evaluator
from llamda.individual import Individual
from llamda.llm_client.base import BaseClient
from llamda.problem import Problem
from llamda.utils import extract_code_from_generator, print_hyperlink

logger = logging.getLogger("llamda")


class ReEvoLLMClients:

    def __init__(
        self,
        generator_llm: BaseClient,
        reflector_llm: BaseClient | None = None,
        short_reflector_llm: BaseClient | None = None,
        long_reflector_llm: BaseClient | None = None,
        crossover_llm: BaseClient | None = None,
        mutation_llm: BaseClient | None = None,
    ) -> None:
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm
        self.short_reflector_llm = short_reflector_llm or self.reflector_llm
        self.long_reflector_llm = long_reflector_llm or self.reflector_llm
        self.crossover_llm = crossover_llm or generator_llm
        self.mutation_llm = mutation_llm or generator_llm


@dataclass
class ReEvoConfig:

    max_fe: int = 100  # maximum number of function evaluations
    pop_size: int = 10  # population size for GA
    init_pop_size: int = 30  # initial population size for GA
    mutation_rate: float = 0.5  # mutation rate for GA
    diversify_init_pop: bool = True  # whether to diversify the initial population


class ReEvo(GeneticAlgorithm[ReEvoConfig, Problem]):
    def __init__(
        self,
        config: ReEvoConfig,
        problem: Problem,
        evaluator: Evaluator,
        output_dir: str,
        llm_client: BaseClient,
        reflector_llm: Optional[BaseClient] = None,
        # Support setting different LLMs for each of the four operators:
        # Short-term Reflection, Long-term Reflection, Crossover, Mutation
        short_reflector_llm: Optional[BaseClient] = None,
        long_reflector_llm: Optional[BaseClient] = None,
        crossover_llm: Optional[BaseClient] = None,
        mutation_llm: Optional[BaseClient] = None,
    ) -> None:

        super().__init__(
            config=config,
            problem=problem,
            evaluator=evaluator,
            llm_client=llm_client,
            output_dir=output_dir,
        )

        self.llm_clients = ReEvoLLMClients(
            generator_llm=llm_client,
            reflector_llm=reflector_llm,
            short_reflector_llm=short_reflector_llm,
            long_reflector_llm=long_reflector_llm,
            crossover_llm=crossover_llm,
            mutation_llm=mutation_llm,
        )
        self.evol = Evolution(self.problem)

        self.evaluator = evaluator

        self.mutation_rate = self.config.mutation_rate
        self.iteration = 0
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None

        self.init_population()

    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        code = extract_code_from_generator(self.problem.seed_func).replace("v1", "v2")
        seed_ind = Individual(
            stdout_filepath=f"{self.output_dir}/problem_iter{self.iteration}_stdout0.txt",
            code_path=f"{self.output_dir}/problem_iter{self.iteration}_code0.py",
            code=code,
            response_id=0,
        )
        self.population = self.evaluator.batch_evaluate([seed_ind])
        self.seed_ind = self.population[0]

        # If seed function is invalid, stop
        if not self.seed_ind.exec_success:
            raise RuntimeError(
                "Seed function is invalid. "
                f"Please check the stdout file in {os.getcwd()}."
            )

        self.update_iter()

        # Generate responses
        messages = self.evol.get_seed_population_messages(self.long_term_reflection_str)
        responses = self.llm_clients.generator_llm.multi_chat_completion(
            [messages],
            self.config.init_pop_size,
            temperature=self.llm_clients.generator_llm.temperature + 0.3,
        )

        population = [
            self.response_to_individual(resp, index)
            for index, resp in enumerate(responses)
        ]

        # Run code and evaluate population
        population = self.evaluator.batch_evaluate(population)
        # Update iteration
        self.population = population
        self.update_iter()

    def response_to_individual(
        self, response: str, response_id: int, file_name: str = None
    ) -> Individual:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = (
            f"problem_iter{self.iteration}_response{response_id}.txt"
            if file_name is None
            else file_name + ".txt"
        )
        file_name = f"{self.output_dir}/{file_name}"
        with open(file_name, "w", encoding="utf-8") as file:
            file.writelines(response + "\n")

        code = extract_code_from_generator(response)

        # Extract code and description from response
        std_out_filepath = (
            f"problem_iter{self.iteration}_stdout{response_id}.txt"
            if file_name is None
            else file_name.rstrip(".txt") + "_stdout.txt"
        )

        individual = Individual(
            stdout_filepath=std_out_filepath,
            code_path=f"problem_iter{self.iteration}_code{response_id}.py",
            code=code,
            response_id=response_id,
        )

        return individual

    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual.obj for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))

        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx].code
            self.best_code_path_overall = population[best_sample_idx].code_path

        # update elitist
        if self.elitist is None or best_obj < self.elitist.obj:
            self.elitist = population[best_sample_idx]
            logger.info(f"Iteration {self.iteration}: Elitist: {self.elitist.obj}")

        best_path = self.best_code_path_overall.replace(".py", ".txt").replace(
            "code", "response"
        )
        logger.info(
            f"Best obj: {self.best_obj_overall}, Best Code Path: {print_hyperlink(best_path, self.best_code_path_overall)}"
        )
        logger.info(f"Iteration {self.iteration} finished...")
        logger.info(f"Function Evals: {self.evaluator.function_evals}")

        self.iteration += 1

    def rank_select(self, population: list[Individual]) -> list[Individual] | None:
        """
        Rank-based selection, select individuals with probability proportional to rank.
        """
        if self.problem.type == "black_box":
            population = [
                individual
                for individual in population
                if individual.exec_success and individual.obj < self.seed_ind.obj
            ]
        else:
            population = [
                individual for individual in population if individual.exec_success
            ]
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x.obj)
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.config.pop_size:
            trial += 1
            parents: list[Individual] = np.random.choice(
                np.array(population), size=2, replace=False, p=probs
            ).tolist()
            if parents[0].obj != parents[1].obj:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def random_select(self, population: list[Individual]) -> list[Individual] | None:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population: list[Individual] = []
        # Eliminate invalid individuals
        if self.problem.type == "black_box":
            population = [
                individual
                for individual in population
                if individual.exec_success and individual.obj < self.seed_ind.obj
            ]
        else:
            population = [
                individual for individual in population if individual.exec_success
            ]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.config.pop_size:
            trial += 1
            parents: list[Individual] = np.random.choice(
                np.array(population), size=2, replace=False
            ).tolist()
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0].obj != parents[1].obj:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
        messages = self.evol.get_long_term_reflection_messages(
            short_term_reflections, self.long_term_reflection_str
        )
        self.long_term_reflection_str = (
            self.llm_clients.long_reflector_llm.multi_chat_completion([messages])[0]
        )

        # Write reflections to file
        file_name = (
            f"{self.output_dir}/problem_iter{self.iteration}_short_term_reflections.txt"
        )
        with open(file_name, "w") as file:
            file.writelines("\n".join(short_term_reflections) + "\n")

        file_name = (
            f"{self.output_dir}/problem_iter{self.iteration}_long_term_reflection.txt"
        )
        with open(file_name, "w") as file:
            file.writelines(self.long_term_reflection_str + "\n")

    def evolve(self) -> tuple[str, str]:
        logger.info("Starting ReEvo evolution")
        while self.evaluator.function_evals < self.config.max_fe:
            logger.debug(
                "Evolution iteration",
                extra={
                    "iteration": self.iteration,
                    "function_evals": self.evaluator.function_evals,
                    "max_fe": self.config.max_fe,
                },
            )
            # If all individuals are invalid, stop
            if all([not individual.exec_success for individual in self.population]):
                logger.error("All individuals are invalid, stopping evolution")
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
                logger.error("Selection failed")
                raise RuntimeError("Selection failed. Please check the population.")

            logger.debug(f"Selected {len(selected_population)} individuals")

            # Short-term reflection
            messages_lst, worse_code_lst, better_code_lst = (
                self.evol.get_short_term_reflection_messages(selected_population)
            )
            short_term_reflections = (
                self.llm_clients.short_reflector_llm.multi_chat_completion(messages_lst)
            )

            logger.debug("Short-term reflection complete")

            # Crossover
            crossover_messages = self.evol.get_crossover_messages(
                (short_term_reflections, worse_code_lst, better_code_lst)
            )
            crossed_response_lst = self.llm_clients.crossover_llm.multi_chat_completion(
                crossover_messages
            )
            crossed_population = [
                self.response_to_individual(response, response_id)
                for response_id, response in enumerate(crossed_response_lst)
            ]

            logger.debug(f"Crossover generated {len(crossed_population)} individuals")

            # Evaluate
            self.population = self.evaluator.batch_evaluate(crossed_population)
            # Update
            self.update_iter()
            # Long-term reflection
            self.long_term_reflection(short_term_reflections)

            logger.debug("Long-term reflection complete")

            # Mutate
            mutation_messages = self.evol.get_mutation_messages(
                self.long_term_reflection_str, self.elitist
            )
            mutated_response_lst = self.llm_clients.mutation_llm.multi_chat_completion(
                [mutation_messages],
                int(self.config.pop_size * self.config.mutation_rate),
            )
            mutated_population = [
                self.response_to_individual(response, response_id)
                for response_id, response in enumerate(mutated_response_lst)
            ]

            logger.debug(f"Mutation generated {len(mutated_population)} individuals")

            # Evaluate
            self.population.extend(self.evaluator.batch_evaluate(mutated_population))
            # Update
            self.update_iter()

        logger.info(
            "ReEvo evolution completed",
            extra={
                "best_objective": self.best_obj_overall,
                "function_evals": self.evaluator.function_evals,
            },
        )

        return self.best_code_overall, self.best_code_path_overall
