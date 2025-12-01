# Adapted from MCTS-AHD: https://github.com/zz1358m/MCTS-AHD-master/blob/main/source/mcts_ahd.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import copy
import heapq
import json
import logging
from dataclasses import dataclass, field

import numpy as np

from llamda.ga.base import GeneticAlgorithm
from llamda.ga.mcts.evolution import MCTSOperator
from llamda.ga.mcts.mcts import MCTS, MCTSNode
from llamda.evaluate import Evaluator
from llamda.ga.mcts.evolution_interface import MCTSIndividual, InterfaceEC
from llamda.llm_client.base import BaseClient
from llamda.problem import EohProblem

logger = logging.getLogger("llamda")


@dataclass
class AHDConfig:

    # MCTS configuration
    pop_size: int = 10  # Size of Elite set E, default = 10
    init_size: int = 4  # Number of initial nodes N_I, default = 4
    fe_max: int = 1000  # Number of evaluations, default = 1000
    operators: list[str] = field(
        default_factory=lambda: ["e1", "e2", "m1", "m2", "s1"]
    )  # evolution operators
    m: int = 5  # Note: m=5 was a manual override in the original implementation
    operator_weights: list[int] = field(
        default_factory=lambda: [0, 1, 2, 2, 1]
    )  # weights for operators default


class MCTS_AHD(GeneticAlgorithm[AHDConfig, EohProblem]):

    def __init__(
        self,
        config: AHDConfig,
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

        self.eval_times = 0  # number of populations

    # add new individual to population
    def add2pop(
        self, population: list[MCTSIndividual], offspring: MCTSIndividual
    ) -> None:
        for ind in population:
            if ind.algorithm == offspring.algorithm:
                # TODO: no actual retry logic implemented
                logger.warning("duplicated result, retrying ... ")
        population.append(offspring)

    def expand(
        self,
        mcts: MCTS,
        cur_node: MCTSNode,
        nodes_set: list[MCTSIndividual],
        option: str,
    ) -> list[MCTSIndividual]:
        if option == "s1":
            path_set: list[MCTSIndividual] = []
            now = copy.deepcopy(cur_node)
            while now.code != "Root":
                path_set.append(now.raw_info)
                now = copy.deepcopy(now.parent)
            path_set = manage_population_s1(path_set, len(path_set))
            if len(path_set) == 1:
                return nodes_set
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(
                eval_times=self.eval_times,
                pop=path_set,
                node=cur_node.raw_info,
                operator=MCTSOperator.S1,
            )
        elif option == "e1":
            e1_set = [
                copy.deepcopy(
                    children.subtree[np.random.randint(len(children.subtree))].raw_info
                )
                for children in mcts.root.children
            ]
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(
                eval_times=self.eval_times,
                pop=e1_set,
                node=cur_node.raw_info,
                operator=MCTSOperator.E1,
            )
        else:
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(
                eval_times=self.eval_times,
                pop=nodes_set,
                node=cur_node.raw_info,
                operator=MCTSOperator(option),
            )
        if offsprings == None:
            logger.warning(f"Timeout emerge, no expanding with action {option}.")
            return nodes_set

        if option != "e1":
            logger.info(
                "Action",
                extra={
                    "action": option,
                    "father_obj": cur_node.raw_info.obj,
                    "now_obj": offsprings.obj,
                    "depth": cur_node.depth + 1,
                },
            )
        else:
            if self.interface_ec.check_duplicate_obj(
                mcts.root.children_info, offsprings.obj
            ):
                logger.info(
                    "Duplicated e1, no action, Father is Root",
                    extra={"abandon_obj": offsprings.obj},
                )
                return nodes_set
            else:
                logger.info(
                    "Action, Father is Root",
                    extra={"now_obj": offsprings.obj},
                )
        if offsprings.obj != float("inf"):
            self.add2pop(
                nodes_set, offsprings
            )  # Check duplication, and add the new offspring
            size_act = min(len(nodes_set), self.config.pop_size)
            nodes_set = manage_population(nodes_set, size_act)
            nownode = MCTSNode(
                offsprings.algorithm,
                offsprings.code,
                offsprings.obj,
                parent=cur_node,
                depth=cur_node.depth + 1,
                visit=1,
                Q=-1 * offsprings.obj,
                raw_info=offsprings,
            )
            if option == "e1":
                nownode.subtree.append(nownode)
            cur_node.add_child(nownode)
            cur_node.children_info.append(offsprings)
            mcts.backpropagate(nownode)
        return nodes_set

    # run eoh
    def run(self) -> tuple[str, str]:
        logger.info("Starting MCTS-AHD evolution")

        self.interface_ec = InterfaceEC(
            m=self.config.m,
            problem=self.problem,
            evaluator=self.evaluator,
            llm_client=self.llm_client,
            output_dir=self.output_dir,
        )

        brothers: list[MCTSIndividual] = []
        mcts = MCTS("Root")
        # main loop
        n_op = len(self.config.operators)
        n_evals, brothers, offspring = self.interface_ec.get_algorithm(
            brothers, MCTSOperator.I1
        )
        self.eval_times += n_evals
        brothers.append(offspring)
        nownode = MCTSNode(
            offspring.algorithm,
            offspring.code,
            offspring.obj,
            parent=mcts.root,
            depth=1,
            visit=1,
            Q=-1 * offspring.obj,
            raw_info=offspring,
        )
        mcts.root.add_child(nownode)
        mcts.root.children_info.append(offspring)
        mcts.backpropagate(nownode)
        nownode.subtree.append(nownode)

        logger.info(
            "Initial node created",
            extra={
                "objective": offspring.obj,
                "eval_times": self.eval_times,
            },
        )

        for i in range(1, self.config.init_size):
            n_evals, brothers, offspring = self.interface_ec.get_algorithm(
                brothers, MCTSOperator.E1
            )
            self.eval_times += n_evals
            brothers.append(offspring)
            nownode = MCTSNode(
                offspring.algorithm,
                offspring.code,
                offspring.obj,
                parent=mcts.root,
                depth=1,
                visit=1,
                Q=-1 * offspring.obj,
                raw_info=offspring,
            )
            mcts.root.add_child(nownode)
            mcts.root.children_info.append(offspring)
            mcts.backpropagate(nownode)
            nownode.subtree.append(nownode)
        nodes_set = brothers
        size_act = min(len(nodes_set), self.config.pop_size)
        nodes_set = manage_population(nodes_set, size_act)

        logger.info(
            "Initialization completed",
            extra={
                "population_size": len(nodes_set),
                "eval_times": self.eval_times,
            },
        )
        while self.eval_times < self.config.fe_max:
            logger.info("MCTS-AHD iteration", extra={"rank_list": mcts.rank_list})
            cur_node = mcts.root
            while len(cur_node.children) > 0 and cur_node.depth < mcts.max_depth:
                uct_scores = [
                    mcts.uct(node, max(1 - self.eval_times / self.config.fe_max, 0))
                    for node in cur_node.children
                ]
                selected_pair_idx = uct_scores.index(max(uct_scores))
                if int((cur_node.visits) ** mcts.alpha) > len(cur_node.children):
                    if cur_node == mcts.root:
                        op = "e1"
                        nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                    else:
                        # i = random.randint(1, n_op - 1)
                        i = 1
                        op = self.config.operators[i]
                        nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                cur_node = cur_node.children[selected_pair_idx]
            for i in range(n_op):
                op = self.config.operators[i]
                logger.info(f"Iter: {self.eval_times}/{self.config.fe_max} OP: {op}")
                op_w = self.config.operator_weights[i]
                for j in range(op_w):
                    nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                assert len(cur_node.children) == len(cur_node.children_info)
            # Save population to a file
            filename = f"{self.output_dir}/population_generation_{self.eval_times}.json"
            with open(filename, "w") as f:
                json.dump(
                    [individual.to_dict() for individual in nodes_set], f, indent=5
                )

            # Save the best one to a file
            filename = (
                f"{self.output_dir}/best_population_generation_{self.eval_times}.json"
            )
            with open(filename, "w") as f:
                json.dump(nodes_set[0].to_dict(), f, indent=5)

        return nodes_set[0].code, filename


def manage_population(
    pop_input: list[MCTSIndividual], size: int
) -> list[MCTSIndividual]:
    pop = [individual for individual in pop_input if individual.obj is not None]
    if size > len(pop):
        size = len(pop)
    unique_pop: list[MCTSIndividual] = []
    unique_objectives = []
    for individual in pop:
        if individual.obj not in unique_objectives:
            unique_pop.append(individual)
            unique_objectives.append(individual.obj)
    # Delete the worst individual
    pop_new = heapq.nsmallest(size, unique_pop, key=lambda x: x.obj)
    return pop_new


def manage_population_s1(
    pop_input: list[MCTSIndividual], size: int
) -> list[MCTSIndividual]:
    pop = [individual for individual in pop_input if individual.obj is not None]
    if size > len(pop):
        size = len(pop)
    unique_pop: list[MCTSIndividual] = []
    unique_algorithms = []
    for individual in pop:
        if individual.algorithm not in unique_algorithms:
            unique_pop.append(individual)
            unique_algorithms.append(individual.algorithm)
    # Delete the worst individual
    pop_new = heapq.nlargest(size, unique_pop, key=lambda x: x.obj)
    return pop_new
