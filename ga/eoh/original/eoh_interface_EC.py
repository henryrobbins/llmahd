from dataclasses import dataclass
import random
from typing import List

import numpy as np

from ga.eoh.original.eoh_evolution import EOHOperator, Evolution
from ga.eoh.problem_adapter import Problem
from utils.llm_client.base import BaseClient


@dataclass
class Heuristic:
    algorithm: str
    code: str
    objective: float
    other_inf: dict


class InterfaceEC:
    def __init__(
        self, pop_size: int, m: int, interface_prob: Problem, llm_client: BaseClient
    ):
        self.pop_size = pop_size
        self.m = m
        self.interface_eval = interface_prob
        self.evol = Evolution(llm_client=llm_client, prompts=interface_prob.prompts)

    def check_duplicate(self, population: list[Heuristic], code: str) -> bool:
        for ind in population:
            if code == ind.code:
                return True
        return False

    def population_generation(self) -> list[Heuristic]:
        n_create = 2
        population = []
        for _ in range(n_create):
            _, pop = self.get_algorithm([], EOHOperator.I1)
            for p in pop:
                population.append(p)
        return population

    def population_generation_seed(self, seeds: list[Heuristic]) -> list[Heuristic]:

        population: list[Heuristic] = []
        fitness = self.interface_eval.batch_evaluate([seed.code for seed in seeds])
        for i in range(len(seeds)):
            obj = np.array(fitness[i])
            seed_alg = Heuristic(
                algorithm=seeds[i].algorithm,
                code=seeds[i].code,
                objective=np.round(obj, 5),
                other_inf={},
            )
            population.append(seed_alg)

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")

        return population

    def _get_alg(
        self, pop: list[Heuristic], operator: EOHOperator
    ) -> tuple[list[Heuristic], Heuristic]:

        match operator:
            case EOHOperator.I1:
                parents = None
                code, algorithm = self.evol.i1()
            case EOHOperator.E1:
                parents = select_parents(pop, self.m)
                code, algorithm = self.evol.e1(parents)
            case EOHOperator.E2:
                parents = select_parents(pop, self.m)
                code, algorithm = self.evol.e2(parents)
            case EOHOperator.M1:
                parents = select_parents(pop, 1)
                code, algorithm = self.evol.m1(parents[0])
            case EOHOperator.M2:
                parents = select_parents(pop, 1)
                code, algorithm = self.evol.m2(parents[0])
            case _:
                raise ValueError(
                    f"Evolution operator [{operator}] has not been implemented!"
                )

        offspring = Heuristic(
            algorithm=algorithm,
            code=code,
            objective=None,
            other_inf={},
        )

        return parents, offspring

    def get_offspring(
        self, pop: list[Heuristic], operator: EOHOperator
    ) -> tuple[list[Heuristic], Heuristic]:

        try:
            p, offspring = self._get_alg(pop, operator)
            n_retry = 1
            while self.check_duplicate(pop, offspring.code):
                n_retry += 1
                p, offspring = self._get_alg(pop, operator)
                if n_retry > 1:
                    break

        except Exception as e:
            print(e)

        return p, offspring

    def get_algorithm(
        self, pop: list[Heuristic], operator: EOHOperator
    ) -> tuple[list[list[Heuristic]], list[Heuristic]]:
        offspring_list: list[tuple[list[Heuristic], Heuristic]] = []
        for _ in range(self.pop_size):
            p, offspring = self.get_offspring(pop, operator)
            offspring_list.append((p, offspring))

        objs = self.interface_eval.batch_evaluate(
            [offspring.code for _, offspring in offspring_list], 0
        )
        for i, (_, offspring) in enumerate(offspring_list):
            offspring.objective = np.round(objs[i], 5)

        results = offspring_list

        out_p: list[list[Heuristic]] = []
        out_off: list[Heuristic] = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
        return out_p, out_off


def select_parents(pop: List, m: int) -> List:
    ranks = [i for i in range(len(pop))]
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    parents = random.choices(pop, weights=probs, k=m)
    return parents
