import copy
import random
from typing import List

import numpy as np

from ga.mcts.problem_adapter import Problem
from ga.mcts.source.evolution import Evolution, MCTSOperator
from utils.llm_client.base import BaseClient


class InterfaceEC:

    def __init__(self, m: int, interface_prob: Problem, llm_client: BaseClient):
        self.m = m
        self.interface_eval = interface_prob
        self.evol = Evolution(llm_client, interface_prob.prompts)

    def check_duplicate_obj(self, population, obj):
        for ind in population:
            if obj == ind["objective"]:
                return True
        return False

    def check_duplicate(self, population, code):
        for ind in population:
            if code == ind["code"]:
                return True
        return False

    def _get_alg(self, pop: list[dict], operator: MCTSOperator, father=None):
        offspring = {
            "algorithm": None,
            "thought": None,
            "code": None,
            "objective": None,
            "other_inf": None,
        }

        match operator:
            case MCTSOperator.I1:
                parents = None
                [offspring["code"], offspring["algorithm"]] = self.evol.i1()
            case MCTSOperator.E1:
                real_m = random.randint(2, self.m)
                real_m = min(real_m, len(pop))
                parents = select_parents_e1(pop, real_m)
                [offspring["code"], offspring["thought"]] = self.evol.e1(parents)
            case MCTSOperator.E2:
                other = copy.deepcopy(pop)
                if father in pop:
                    other.remove(father)
                real_m = 1
                # real_m = random.randint(2, self.m) - 1
                # real_m = min(real_m, len(other))
                parents = select_parents(other, real_m)
                parents.append(father)
                [offspring["code"], offspring["thought"]] = self.evol.e2(parents)
            case MCTSOperator.M1:
                parents = [father]
                [offspring["code"], offspring["thought"]] = self.evol.m1(parents[0])
            case MCTSOperator.M2:
                parents = [father]
                [offspring["code"], offspring["thought"]] = self.evol.m2(parents[0])
            case MCTSOperator.S1:
                parents = pop
                [offspring["code"], offspring["thought"]] = self.evol.s1(pop)
            case _:
                print(f"Evolution operator [{operator}] has not been implemented ! \n")

        offspring["algorithm"] = self.evol.post_thought(
            offspring["code"], offspring["thought"]
        )
        return parents, offspring

    def get_offspring(self, pop, operator: MCTSOperator, father=None):
        while True:
            try:
                p, offspring = self._get_alg(pop, operator, father=father)
                code = offspring["code"]
                n_retry = 1
                while self.check_duplicate(pop, offspring["code"]):
                    n_retry += 1
                    print("duplicated code, wait 1 second and retrying ... ")
                    p, offspring = self._get_alg(pop, operator, father=father)
                    code = offspring["code"]
                    if n_retry > 1:
                        break
                break
            except Exception as e:
                print(e)
        return p, offspring

    def get_algorithm(self, eval_times, pop, operator: MCTSOperator):
        while True:
            eval_times += 1
            parents, offspring = self.get_offspring(pop, operator)
            objs = self.interface_eval.batch_evaluate([offspring["code"]], 0)
            if (
                objs == "timeout"
                or objs[0] == float("inf")
                or self.check_duplicate_obj(pop, np.round(objs[0], 5))
            ):
                continue
            offspring["objective"] = np.round(objs[0], 5)

            return eval_times, pop, offspring
        return eval_times, None, None

    def evolve_algorithm(
        self, eval_times, pop, node, brother_node, operator: MCTSOperator
    ):
        for i in range(3):
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator, father=node)
            objs = self.interface_eval.batch_evaluate([offspring["code"]], 0)
            if objs == "timeout":
                return eval_times, None
            if objs[0] == float("inf") or self.check_duplicate(pop, offspring["code"]):
                continue
            offspring["objective"] = np.round(objs[0], 5)

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
