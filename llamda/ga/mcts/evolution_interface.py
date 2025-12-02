# Adapted from MCTS-AHD: https://github.com/zz1358m/MCTS-AHD-master/blob/main/source/evolution_interface.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging
import copy
import os
from pathlib import Path

import numpy as np

from llamda.evaluate import Evaluator
from llamda.problem import EohProblem
from llamda.ga.mcts.mcts_prompts import MCTSPrompts, MCTSIndividual, MCTSOperator
from llamda.llm_client.base import BaseClient
from llamda.ga.utils import generate_thought_and_code

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
        self.evol = MCTSPrompts(problem)
        self.problem = problem
        self.llm_client = llm_client
        self.output_dir = output_dir

    def _logging_context(self) -> dict:
        return {"method": "MCTS-AHD", "problem_name": self.problem.name}

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

    def _get_thought(self, prompt_content: str) -> str:
        """Call LLM with refine prompt to get algorithm description."""
        response = self.llm_client.chat_completion(
            n=1,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=0,
        )
        return response[0].message.content

    def _post_thought(self, code: str, thought: str) -> str:
        """Refine thought into algorithm description."""
        prompt_content = self.evol.refine(code, thought)
        return self._get_thought(prompt_content)

    def get_offspring(
        self,
        pop: list[MCTSIndividual],
        operator: MCTSOperator,
        name: str,
        father: MCTSIndividual | None = None,
    ) -> tuple[list[MCTSIndividual], MCTSIndividual]:

        match operator:
            case MCTSOperator.I1:
                parents = []
                prompt_content = self.evol.i1()
            case MCTSOperator.E1:
                real_m = np.random.randint(2, self.m)
                real_m = min(real_m, len(pop))
                parents = select_parents_e1(pop, real_m)
                prompt_content = self.evol.e1(parents)
            case MCTSOperator.E2:
                other = copy.deepcopy(pop)
                if father in pop:
                    other.remove(father)
                real_m = 1
                # real_m = random.randint(2, self.m) - 1
                # real_m = min(real_m, len(other))
                parents = select_parents(other, real_m)
                parents.append(father)
                prompt_content = self.evol.e2(parents)
            case MCTSOperator.M1:
                parents = [father]
                prompt_content = self.evol.m1(parents[0])
            case MCTSOperator.M2:
                parents = [father]
                prompt_content = self.evol.m2(parents[0])
            case MCTSOperator.S1:
                parents = pop
                prompt_content = self.evol.s1(pop)
            case _:
                logger.warning(
                    f"Evolution operator [{operator}] has not been implemented!"
                )

        logger.debug(
            f"Executing MCTS operator {operator.value}",
            extra={
                "individual_name": name,
                "parent_names": [p.name for p in parents],
                **self._logging_context(),
            },
        )
        for _ in range(3):
            response, thought, code = generate_thought_and_code(
                prompt_content=prompt_content,
                func_outputs=self.problem.func_outputs,
                llm_client=self.llm_client,
            )
            algorithm = self._post_thought(code, thought)
            if not self.check_duplicate(pop, code):
                offspring = MCTSIndividual(
                    name=name, algorithm=algorithm, thought=thought, code=code
                )
                individual_dir = f"{self.output_dir}/individuals/{offspring.name}"
                os.makedirs(individual_dir, exist_ok=True)
                offspring.write_code_to_file(f"{individual_dir}/code.py")
                response_filepath = f"{individual_dir}/response.txt"
                with open(response_filepath, "w") as f:
                    f.write(response)
                prompt_filepath = f"{individual_dir}/prompt.txt"
                with open(prompt_filepath, "w") as f:
                    f.write(prompt_content)
                return parents, offspring
            else:
                logger.warning(
                    "Duplicate code detected, regenerating offspring.",
                    extra={"individual_name": name, **self._logging_context()},
                )

        raise ValueError("Unable to generate unique offspring after multiple attempts.")

    def get_algorithm(
        self, pop: list[MCTSIndividual], operator: MCTSOperator, name: str
    ) -> tuple[int, list[MCTSIndividual], MCTSIndividual]:
        n_evals = 0
        for _ in range(10):
            n_evals += 1
            _, offspring = self.get_offspring(pop, operator, name=name)
            obj = self.interface_eval.batch_evaluate(
                [offspring], Path(self.output_dir)
            )[0].obj
            if (
                obj == "timeout"
                or obj == float("inf")
                or self.check_duplicate_obj(pop, np.round(obj, 5))
            ):
                continue

            offspring.obj = float(np.round(obj, 5))
            return n_evals, pop, offspring

        raise ValueError("Unable to generate offspring with unique objective value.")

    def evolve_algorithm(
        self,
        eval_times: int,
        pop: list[MCTSIndividual],
        node: MCTSIndividual,
        operator: MCTSOperator,
        name: str,
    ) -> tuple[int, MCTSIndividual | None]:
        for i in range(3):
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator, name=name, father=node)
            population = self.interface_eval.batch_evaluate(
                [offspring], Path(self.output_dir)
            )
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
