import copy
from dataclasses import dataclass
import random
from typing import List

import numpy as np

from utils.individual import Individual
from utils.problem import Problem, hydrate_individual
from ga.mcts.source.evolution import Evolution, MCTSOperator
from utils.llm_client.base import BaseClient


@dataclass
class MCTSIndividual(Individual):
    algorithm: str | None = None
    thought: str | None = None


class InterfaceEC:

    def __init__(self, m: int, interface_prob: Problem, llm_client: BaseClient):
        self.m = m
        self.interface_eval = interface_prob
        self.evol = Evolution(llm_client, interface_prob.prompts)

    def check_duplicate_obj(self, population: list[MCTSIndividual], obj: float) -> bool:
        for ind in population:
            if obj == ind.obj:
                return True
        return False

    def check_duplicate(self, population: list[MCTSIndividual], code: str) -> bool:
        for ind in population:
            if code == ind.code:
                return True
        return False

    def _get_alg(
        self,
        pop: list[MCTSIndividual],
        operator: MCTSOperator,
        father: MCTSIndividual | None = None,
    ) -> tuple[list[MCTSIndividual], MCTSIndividual]:

        match operator:
            case MCTSOperator.I1:
                parents = None
                code, thought = self.evol.i1()
            case MCTSOperator.E1:
                real_m = random.randint(2, self.m)
                real_m = min(real_m, len(pop))
                parents = select_parents_e1(pop, real_m)
                code, thought = self.evol.e1(parents)
            case MCTSOperator.E2:
                other = copy.deepcopy(pop)
                if father in pop:
                    other.remove(father)
                real_m = 1
                # real_m = random.randint(2, self.m) - 1
                # real_m = min(real_m, len(other))
                parents = select_parents(other, real_m)
                parents.append(father)
                code, thought = self.evol.e2(parents)
            case MCTSOperator.M1:
                parents = [father]
                code, thought = self.evol.m1(parents[0])
            case MCTSOperator.M2:
                parents = [father]
                code, thought = self.evol.m2(parents[0])
            case MCTSOperator.S1:
                parents = pop
                code, thought = self.evol.s1(pop)
            case _:
                print(f"Evolution operator [{operator}] has not been implemented ! \n")

        algorithm = self.evol.post_thought(code, thought)

        offspring = MCTSIndividual(
            algorithm=algorithm,
            thought=thought,
            code=code,
            obj=None,
        )

        return parents, offspring

    def get_offspring(
        self,
        pop: list[MCTSIndividual],
        operator: MCTSOperator,
        father: MCTSIndividual | None = None,
    ) -> tuple[list[MCTSIndividual], MCTSIndividual]:
        while True:
            try:
                p, offspring = self._get_alg(pop, operator, father=father)
                n_retry = 1
                while self.check_duplicate(pop, offspring.code):
                    n_retry += 1
                    print("duplicated code, wait 1 second and retrying ... ")
                    p, offspring = self._get_alg(pop, operator, father=father)
                    if n_retry > 1:
                        break
                break
            except Exception as e:
                print(e)
        return p, offspring

    def get_algorithm(
        self, pop: list[MCTSIndividual], operator: MCTSOperator
    ) -> tuple[int, list[MCTSIndividual], MCTSIndividual]:
        n_evals = 0
        while True:
            n_evals += 1
            _, offspring = self.get_offspring(pop, operator)
            offspring = hydrate_individual(offspring, 0, 0)
            obj = self.interface_eval.batch_evaluate([offspring], 0)[0].obj
            if (
                obj == "timeout"
                or obj == float("inf")
                or self.check_duplicate_obj(pop, np.round(obj, 5))
            ):
                continue

            offspring.obj = float(np.round(obj, 5))
            return n_evals, pop, offspring

    def evolve_algorithm(
        self, eval_times, pop, node, brother_node, operator: MCTSOperator
    ):
        for i in range(3):
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator, father=node)
            offspring = hydrate_individual(offspring, 0, 0)
            population = self.interface_eval.batch_evaluate([offspring], 0)
            objs = [indiv.obj for indiv in population]
            if objs == "timeout":
                return eval_times, None
            if objs[0] == float("inf") or self.check_duplicate(pop, offspring.code):
                continue
            offspring.obj = np.round(objs[0], 5)

            return eval_times, offspring
        return eval_times, None


def select_parents(pop: List, m: int) -> List:
    ranks = [i for i in range(len(pop))]
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    parents = random.choices(pop, weights=probs, k=m)
    return parents


def select_parents_e1(pop: List, m: int) -> List:
    probs = [1 for i in range(len(pop))]
    parents = random.choices(pop, weights=probs, k=m)
    return parents
