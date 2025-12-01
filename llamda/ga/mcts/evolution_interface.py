# Adapted from MCTS-AHD: https://github.com/zz1358m/MCTS-AHD-master/blob/main/source/evolution_interface.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging
import copy

import numpy as np

from llamda.evaluate import Evaluator
from llamda.problem import EohProblem, hydrate_individual
from llamda.ga.mcts.evolution import Evolution, MCTSIndividual, MCTSOperator
from llamda.llm_client.base import BaseClient

logger = logging.getLogger("llamda")


class InterfaceEC:

    def __init__(
        self,
        m: int,
        problem: EohProblem,
        evaluator: Evaluator,
        llm_client: BaseClient,
        output_dir: str,
    ):
        self.m = m
        self.interface_eval = evaluator
        self.evol = Evolution(llm_client, problem)
        self.output_dir = output_dir

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
                parents = []
                code, thought = self.evol.i1()
            case MCTSOperator.E1:
                real_m = np.random.randint(2, self.m)
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
                logger.warning(
                    f"Evolution operator [{operator}] has not been implemented!"
                )

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
                code = offspring.code
                n_retry = 1
                while code is None or self.check_duplicate(pop, code):
                    n_retry += 1
                    logger.warning("duplicated code, wait 1 second and retrying ... ")
                    p, offspring = self._get_alg(pop, operator, father=father)
                    code = offspring.code
                    if n_retry > 1:
                        break
                break
            except Exception as e:
                logger.error(e)
        return p, offspring

    def get_algorithm(
        self, pop: list[MCTSIndividual], operator: MCTSOperator
    ) -> tuple[int, list[MCTSIndividual], MCTSIndividual]:
        n_evals = 0
        while True:
            n_evals += 1
            _, offspring = self.get_offspring(pop, operator)
            offspring = hydrate_individual(offspring, 0, self.output_dir, 0)
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
        self,
        eval_times: int,
        pop: list[MCTSIndividual],
        node: MCTSIndividual,
        operator: MCTSOperator,
    ) -> tuple[int, MCTSIndividual | None]:
        for i in range(3):
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator, father=node)
            offspring = hydrate_individual(offspring, 0, self.output_dir, 0)
            population = self.interface_eval.batch_evaluate([offspring], 0)
            objs = [indiv.obj for indiv in population]
            if objs == "timeout":
                return eval_times, None
            if objs[0] == float("inf") or self.check_duplicate(pop, offspring.code):
                continue
            offspring.obj = np.round(objs[0], 5)

            return eval_times, offspring
        return eval_times, None


def select_parents(pop: list[MCTSIndividual], m: int) -> list[MCTSIndividual]:
    ranks = [i for i in range(len(pop))]
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    parents = np.random.choice(
        np.array(pop), p=probs / np.sum(probs), size=m, replace=True
    ).tolist()
    return parents


def select_parents_e1(pop: list[MCTSIndividual], m: int) -> list[MCTSIndividual]:
    probs = [1 for i in range(len(pop))]
    parents = np.random.choice(
        np.array(pop), p=probs / np.sum(probs), size=m, replace=True
    ).tolist()
    return parents
