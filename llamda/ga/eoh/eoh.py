# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/original/eoh.py
# Originally from EoH: https://github.com/FeiLiu36/EoH/blob/main/eoh/src/eoh/methods/eoh/eoh.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging
from dataclasses import dataclass, field
import numpy as np
import json
import heapq
import time

from llamda.ga.base import GeneticAlgorithm
from llamda.ga.eoh.eoh_prompts import EOHOperator
from llamda.evaluate import Evaluator
from llamda.ga.eoh.eoh_interface_EC import EOHIndividual, InterfaceEC
from llamda.ga.utils import population_checkpoint
from llamda.llm_client.base import BaseClient
from llamda.problem import EohProblem

logger = logging.getLogger("llamda")

# See original EOH adapter implementation in the reevo repository:
# https://github.com/ai4co/reevo/blob/main/baselines/eoh/eoh_adapter.py
#
# For the default configuration: https://github.com/ai4co/reevo/blob/main/cfg/config.yaml
#
# max_fe: int = 100
# pop_size: int = 10
# init_pop_size: int = 30
#
# total evals = 2 * pop_size + n_pop * 4 * pop_size
# 100 = 2 * 10 + n_pop * 4 * 10
# n_pop = (100 - 20) / 40 + 1 = 3


@dataclass
class EoHConfig:

    # EC settings
    pop_size: int = 10  # number of algorithms in each population
    n_pop: int = 3  # number of populations
    operators: list[str] = field(
        default_factory=lambda: ["e1", "e2", "m1", "m2"]
    )  # evolution operators
    m: int = 2  # number of parents for 'e1' and 'e2' operators
    operator_weights: list[int] = field(
        default_factory=lambda: [1, 1, 1, 1]
    )  # weights for operators

    # Exp settings
    exp_use_seed: bool = False
    exp_seed_path: str = "./seeds/seeds.json"
    exp_use_continue: bool = False
    exp_continue_id: int = 0
    exp_continue_path: str = "./results/pops/population_generation_0.json"


class EOH(GeneticAlgorithm[EoHConfig, EohProblem]):

    def __init__(
        self,
        config: EoHConfig,
        problem: EohProblem,
        evaluator: Evaluator,
        llm_client: BaseClient,
        output_dir: str,
    ) -> None:

        super().__init__(
            config=config,
            problem=problem,
            evaluator=evaluator,
            llm_client=llm_client,
            output_dir=output_dir,
        )

        # Validation
        assert config.m <= config.pop_size or config.m > 1

    def _logging_context(self) -> dict:
        return {
            "method": "EoH",
            "problem_name": self.problem.name,
            "pop_size": self.config.pop_size,
            "n_pop": self.config.n_pop,
        }

    # add new individual to population
    def add2pop(
        self, population: list[EOHIndividual], offspring: list[EOHIndividual]
    ) -> None:
        for off in offspring:
            for ind in population:
                if ind.obj == off.obj:
                    # TODO: No retry logic actually happened in original code
                    pass
            population.append(off)

    def _load_seed_population(
        self, interface_ec: InterfaceEC
    ) -> tuple[list[EOHIndividual], int]:
        with open(self.config.exp_seed_path) as file:
            data = json.load(file)
        population = interface_ec.population_generation_seed(data)
        filename = f"{self.output_dir}/population_generation_0.json"
        with open(filename, "w") as f:
            json.dump([individual.to_dict() for individual in population], f, indent=5)
        n_start = 0
        return population, n_start

    def _load_population(self) -> tuple[list[EOHIndividual], int]:
        population = []
        with open(self.config.exp_continue_path) as file:
            data = json.load(file)
        for individual in data:
            population.append(individual)
        n_start = self.config.exp_continue_id
        return population, n_start

    def _create_new_population(
        self, interface_ec: InterfaceEC
    ) -> tuple[list[EOHIndividual], int]:
        population = interface_ec.population_generation()
        population = manage_population(population, self.config.pop_size)

        # Save population to a file
        filename = f"{self.output_dir}/population_generation_0.json"
        with open(filename, "w") as f:
            json.dump([individual.to_dict() for individual in population], f, indent=5)
        n_start = 0

        return population, n_start

    def _initialize_population(
        self, interface_ec: InterfaceEC
    ) -> tuple[list[EOHIndividual], int]:
        if self.config.exp_use_seed:
            return self._load_seed_population(interface_ec)
        if self.config.exp_use_continue:
            return self._load_population()
        return self._create_new_population(interface_ec)

    def run(self) -> tuple[str, str]:

        logger.info("Starting EoH evolution", extra=self._logging_context())

        time_start = time.time()

        interface_ec = InterfaceEC(
            pop_size=self.config.pop_size,
            m=self.config.m,
            llm_client=self.llm_client,
            problem=self.problem,
            evaluator=self.evaluator,
            output_dir=self.output_dir,
        )

        population, n_start = self._initialize_population(interface_ec)
        logger.info(
            "Initial population created",
            extra={
                "population_size": len(population),
                "n_start": n_start,
                **self._logging_context(),
            },
        )

        for pop in range(n_start, self.config.n_pop):
            logger.info(
                f"Starting population [{pop + 1}/{self.config.n_pop}]",
                extra={**self._logging_context()},
            )
            for i, op in enumerate(self.config.operators):
                logger.info(
                    f"Applying operator [{i + 1} / {len(self.config.operators)}]",
                    extra={
                        "operator": op,
                        "population": pop + 1,
                        **self._logging_context(),
                    },
                )
                # TODO: These operator weights aren't being used as expected
                op_w = self.config.operator_weights[i]
                if np.random.rand() < op_w:
                    _, offsprings = interface_ec.get_algorithm(
                        population, EOHOperator(op), f"population_{pop}_operator_{op}"
                    )
                # Check duplication, and add the new offspring
                # TODO: No retry logic actually happened in original code
                self.add2pop(population, offsprings)

                # Population management
                size_act = min(len(population), self.config.pop_size)
                population = manage_population(population, size_act)

            # Save checkpoint
            filename = population_checkpoint(
                population=population,
                name=f"population_{pop}",
                output_dir=self.output_dir,
            )

            # Logging
            elapsed_time = (time.time() - time_start) / 60
            logger.info(
                f"Population {pop + 1} completed",
                extra={
                    "population_index": pop + 1,
                    "elapsed_time_minutes": elapsed_time,
                    "best_objective": population[0].obj if population else None,
                    "pop_objectives": [indiv.obj for indiv in population],
                    **self._logging_context(),
                },
            )

        code = population[0].code
        assert code is not None

        logger.info(
            "EOH evolution completed",
            extra={
                "best_objective": population[0].obj,
                "total_time_minutes": (time.time() - time_start) / 60,
                **self._logging_context(),
            },
        )

        return code, filename


def manage_population(pop: list[EOHIndividual], size: int) -> list[EOHIndividual]:
    pop = [individual for individual in pop if individual.obj is not None]
    if size > len(pop):
        size = len(pop)
    unique_pop: list[EOHIndividual] = []
    unique_objectives = []
    for individual in pop:
        if individual.obj not in unique_objectives:
            unique_pop.append(individual)
            unique_objectives.append(individual.obj)
    # Delete the worst individual
    pop_new = heapq.nsmallest(size, unique_pop, key=lambda x: x.obj)
    return pop_new
