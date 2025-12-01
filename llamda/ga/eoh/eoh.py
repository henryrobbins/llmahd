from dataclasses import dataclass, field
import numpy as np
import json
import random
import heapq
import time

from llamda.ga.base import GeneticAlgorithm
from llamda.ga.eoh.eoh_evolution import EOHOperator
from llamda.evaluate import Evaluator
from llamda.ga.eoh.eoh_interface_EC import EOHIndividual, InterfaceEC
from llamda.llm_client.base import BaseClient
from llamda.problem import EohProblem

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
    ec_pop_size: int = 10  # number of algorithms in each population
    ec_n_pop: int = 3  # number of populations
    ec_operators: list[str] = field(
        default_factory=lambda: ["e1", "e2", "m1", "m2"]
    )  # evolution operators
    ec_m: int = 2  # number of parents for 'e1' and 'e2' operators
    ec_operator_weights: list[int] = field(
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

        # EOH Configuration
        self.pop_size = config.ec_pop_size
        self.n_pop = config.ec_n_pop
        self.operators = config.ec_operators
        self.operator_weights = config.ec_operator_weights
        assert config.ec_m <= self.pop_size or config.ec_m > 1
        self.m = config.ec_m

        # Experiment Configuration
        self.use_seed = config.exp_use_seed
        self.seed_path = config.exp_seed_path
        self.load_pop = config.exp_use_continue
        self.load_pop_path = config.exp_continue_path
        self.load_pop_id = config.exp_continue_id

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(
        self, population: list[EOHIndividual], offspring: list[EOHIndividual]
    ) -> None:
        for off in offspring:
            for ind in population:
                if ind.obj == off.obj:
                    # TODO: No retry logic is actually happening here
                    print("duplicated result, retrying ... ")
            population.append(off)

    def _load_seed_population(
        self, interface_ec: InterfaceEC
    ) -> tuple[list[EOHIndividual], int]:
        with open(self.seed_path) as file:
            data = json.load(file)
        population = interface_ec.population_generation_seed(data)
        filename = f"{self.output_dir}/population_generation_0.json"
        with open(filename, "w") as f:
            json.dump([individual.to_dict() for individual in population], f, indent=5)
        n_start = 0
        return population, n_start

    def _load_population(self) -> tuple[list[EOHIndividual], int]:
        population = []
        with open(self.load_pop_path) as file:
            data = json.load(file)
        for individual in data:
            population.append(individual)
        print("initial population has been loaded!")
        n_start = self.load_pop_id
        return population, n_start

    def _create_new_population(
        self, interface_ec: InterfaceEC
    ) -> tuple[list[EOHIndividual], int]:
        population = interface_ec.population_generation()
        population = manage_population(population, self.pop_size)

        print("Pop initial: ")
        for off in population:
            print(" Obj: ", off.obj, end="|")
        print()
        print("initial population has been created!")
        # Save population to a file
        filename = f"{self.output_dir}/population_generation_0.json"
        with open(filename, "w") as f:
            json.dump([individual.to_dict() for individual in population], f, indent=5)
        n_start = 0

        return population, n_start

    def _initialize_population(
        self, interface_ec: InterfaceEC
    ) -> tuple[list[EOHIndividual], int]:
        if self.use_seed:
            return self._load_seed_population(interface_ec)
        if self.load_pop:
            return self._load_population()
        return self._create_new_population(interface_ec)

    def _population_checkpoint(self, n: int, population: list[EOHIndividual]) -> str:

        # Save population to a file
        filename = f"{self.output_dir}/population_generation_{str(n + 1)}.json"
        with open(filename, "w") as f:
            json.dump([individual.to_dict() for individual in population], f, indent=5)

        # Save the best one to a file
        filename = f"{self.output_dir}/best_population_generation_{str(n + 1)}.json"
        with open(filename, "w") as f:
            json.dump(population[0].to_dict(), f, indent=5)

        return filename

    def run(self) -> tuple[str, str]:

        print("- Evolution Start -")

        time_start = time.time()

        interface_ec = InterfaceEC(
            pop_size=self.pop_size,
            m=self.m,
            llm_client=self.llm_client,
            problem=self.problem,
            evaluator=self.evaluator,
            output_dir=self.output_dir,
        )

        population, n_start = self._initialize_population(interface_ec)

        for pop in range(n_start, self.n_pop):
            # print(f" [{na + 1} / {self.pop_size}] ", end="|")
            for i, op in enumerate(self.operators):
                print(f" OP: {op}, [{i + 1} / {len(self.operators)}] ", end="|")
                # TODO: These operator weights aren't being used as expected
                op_w = self.operator_weights[i]
                if np.random.rand() < op_w:
                    _, offsprings = interface_ec.get_algorithm(
                        population, EOHOperator(op)
                    )
                # Check duplication, and add the new offspring
                self.add2pop(population, offsprings)
                for off in offsprings:
                    print(" Obj: ", off.obj, end="|")
                # Population management
                size_act = min(len(population), self.pop_size)
                population = manage_population(population, size_act)

            # Save checkpoint
            filename = self._population_checkpoint(n=pop, population=population)

            # Logging
            print(
                f"--- {pop + 1} of {self.n_pop} populations finished. "
                f"Time Cost:  {((time.time()-time_start)/60):.1f} m"
            )
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i].obj) + " ", end="")
            print()

        code = population[0].code
        assert code is not None

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
