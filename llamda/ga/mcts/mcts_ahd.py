import os
import copy
import heapq
import json
import random
from dataclasses import dataclass, field

from llamda.ga.mcts.evolution import MCTSOperator
from llamda.ga.mcts.mcts import MCTS, MCTSNode
from llamda.utils.evaluate import Evaluator
from llamda.ga.mcts.evolution_interface import MCTSIndividual, InterfaceEC
from llamda.utils.llm_client.base import BaseClient
from llamda.utils.problem import EOHProblemPrompts


@dataclass
class AHDConfig:

    # MCTS configuration
    pop_size: int = 10  # Size of Elite set E, default = 10
    init_size: int = 4  # Number of initial nodes N_I, default = 4
    ec_fe_max: int = 1000  # Number of evaluations, default = 1000
    ec_operators: list[str] = field(
        default_factory=lambda: ["e1", "e2", "m1", "m2", "s1"]
    )  # evolution operators
    ec_m: int = 2
    ec_operator_weights: list[int] = field(
        default_factory=lambda: [0, 1, 2, 2, 1]
    )  # weights for operators default

    # Exp settings
    exp_output_path: str = "/"  # default folder for ael outputs


class MCTS_AHD:

    def __init__(
        self,
        paras: AHDConfig,
        prompts: EOHProblemPrompts,
        evaluator: Evaluator,
        llm_client: BaseClient,
        output_dir: str,
    ) -> None:

        self.prompts = prompts
        self.evaluator = evaluator
        self.llm_client = llm_client
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # MCTS Configuration
        self.init_size = paras.init_size
        self.pop_size = paras.pop_size
        self.fe_max = paras.ec_fe_max  # function evaluation times
        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        paras.ec_m = 5
        self.m = paras.ec_m

        self.output_path = paras.exp_output_path

        self.eval_times = 0  # number of populations

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population: list[dict], offspring: dict) -> None:
        for ind in population:
            if ind["algorithm"] == offspring["algorithm"]:
                # TODO: no actual retry logic implemented
                print("duplicated result, retrying ... ")
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
                    children.subtree[
                        random.choices(range(len(children.subtree)), k=1)[0]
                    ].raw_info
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
                pop=e1_set,
                node=cur_node.raw_info,
                operator=MCTSOperator(option),
            )
        if offsprings == None:
            print(f"Timeout emerge, no expanding with action {option}.")
            return nodes_set

        if option != "e1":
            print(
                f"Action: {option}, Father Obj: {cur_node.raw_info.obj}, Now Obj: {offsprings.obj}, Depth: {cur_node.depth + 1}"
            )
        else:
            if self.interface_ec.check_duplicate_obj(
                mcts.root.children_info, offsprings.obj
            ):
                print(
                    f"Duplicated e1, no action, Father is Root, Abandon Obj: {offsprings.obj}"
                )
                return nodes_set
            else:
                print(f"Action: {option}, Father is Root, Now Obj: {offsprings.obj}")
        if offsprings.obj != float("inf"):
            self.add2pop(
                nodes_set, offsprings
            )  # Check duplication, and add the new offspring
            size_act = min(len(nodes_set), self.pop_size)
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
        print("- Initialization Start -")

        self.interface_ec = InterfaceEC(
            m=self.m,
            prompts=self.prompts,
            evaluator=self.evaluator,
            llm_client=self.llm_client,
            output_dir=self.output_dir,
        )

        brothers: list[MCTSIndividual] = []
        mcts = MCTS("Root")
        # main loop
        n_op = len(self.operators)
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
        for i in range(1, self.init_size):
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
        size_act = min(len(nodes_set), self.pop_size)
        nodes_set = manage_population(nodes_set, size_act)
        print("- Initialization Finished - Evolution Start -")
        while self.eval_times < self.fe_max:
            print(f"Current performances of MCTS nodes: {mcts.rank_list}")
            # print([len(node.subtree) for node in mcts.root.children])
            cur_node = mcts.root
            while len(cur_node.children) > 0 and cur_node.depth < mcts.max_depth:
                uct_scores = [
                    mcts.uct(node, max(1 - self.eval_times / self.fe_max, 0))
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
                        op = self.operators[i]
                        nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                cur_node = cur_node.children[selected_pair_idx]
            for i in range(n_op):
                op = self.operators[i]
                print(f"Iter: {self.eval_times}/{self.fe_max} OP: {op}", end="|")
                op_w = self.operator_weights[i]
                for j in range(op_w):
                    nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                assert len(cur_node.children) == len(cur_node.children_info)
            # Save population to a file
            filename = (
                self.output_path
                + "population_generation_"
                + str(self.eval_times)
                + ".json"
            )
            with open(filename, "w") as f:
                json.dump(nodes_set, f, indent=5)

            # Save the best one to a file
            filename = (
                self.output_path
                + "best_population_generation_"
                + str(self.eval_times)
                + ".json"
            )
            with open(filename, "w") as f:
                json.dump(nodes_set[0], f, indent=5)

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
