# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/original/eoh_interface_EC.py
# Originally from EoH: https://github.com/FeiLiu36/EoH/blob/main/eoh/src/eoh/methods/eoh/eoh_interface_EC.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging
import os
from pathlib import Path
import numpy as np

from llamda.ga.eoh.eoh_prompts import EOHIndividual, EOHOperator, EOHPrompts
from llamda.evaluate import Evaluator
from llamda.problem import EohProblem
from llamda.llm_client.base import BaseClient
from llamda.ga.utils import generate_thought_and_code

logger = logging.getLogger("llamda")


class InterfaceEC:
    def __init__(
        self,
        pop_size: int,
        m: int,
        problem: EohProblem,
        evaluator: Evaluator,
        llm_client: BaseClient,
        output_dir: str,
    ):
        self.pop_size = pop_size
        self.m = m
        self.interface_eval = evaluator
        self.evol = EOHPrompts(problem=problem)
        self.llm_client = llm_client
        self.problem = problem
        self.output_dir = output_dir

    def _logging_context(self) -> dict:
        return {"method": "EoH", "problem_name": self.problem.name}

    def check_duplicate(self, population: list[EOHIndividual], code: str) -> bool:
        for ind in population:
            if code == ind.code:
                return True
        return False

    def population_generation(self) -> list[EOHIndividual]:
        n_create = 2
        population = []
        for _ in range(n_create):
            _, pop = self.get_algorithm([], EOHOperator.I1, "init_population")
            for p in pop:
                population.append(p)
        return population

    def population_generation_seed(
        self, seeds: list[EOHIndividual]
    ) -> list[EOHIndividual]:
        population = self.interface_eval.batch_evaluate(seeds, Path(self.output_dir))
        return population

    def get_offspring(
        self, pop: list[EOHIndividual], operator: EOHOperator, name: str
    ) -> tuple[list[EOHIndividual], EOHIndividual]:

        match operator:
            case EOHOperator.I1:
                parents = []
                prompt_content = self.evol.i1()
            case EOHOperator.E1:
                parents = select_parents(pop, self.m)
                prompt_content = self.evol.e1(parents)
            case EOHOperator.E2:
                parents = select_parents(pop, self.m)
                prompt_content = self.evol.e2(parents)
            case EOHOperator.M1:
                parents = select_parents(pop, 1)
                prompt_content = self.evol.m1(parents[0])
            case EOHOperator.M2:
                parents = select_parents(pop, 1)
                prompt_content = self.evol.m2(parents[0])
            case _:
                raise ValueError(
                    f"Evolution operator [{operator}] has not been implemented!"
                )

        logger.debug(
            f"Executing EoH operator {operator.value}",
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
            if not self.check_duplicate(pop, code):
                offspring = EOHIndividual(name=name, algorithm=thought, code=code)
                individual_dir = f"{self.output_dir}/individuals/{offspring.name}"
                os.makedirs(individual_dir, exist_ok=True)
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
        self, pop: list[EOHIndividual], operator: EOHOperator, batch_name: str
    ) -> tuple[list[list[EOHIndividual]], list[EOHIndividual]]:
        offspring_list: list[tuple[list[EOHIndividual], EOHIndividual]] = []

        logger.info(
            f"[EoH] Generating offspring using operator {operator}",
            extra={"n_offspring": self.pop_size, **self._logging_context()},
        )

        for i in range(self.pop_size):

            logger.info(
                f"Generating offspring [{i + 1}/{self.pop_size}]",
                extra={**self._logging_context()},
            )

            p, offspring = self.get_offspring(
                pop, operator, f"{batch_name}_offspring_{i}"
            )
            offspring_list.append((p, offspring))

        pop = [offspring for _, offspring in offspring_list]
        pop = self.interface_eval.batch_evaluate(pop, Path(self.output_dir))
        objs = [indiv.obj for indiv in pop]
        for i, (_, offspring) in enumerate(offspring_list):
            offspring.obj = np.round(objs[i], 5)

        logger.info(
            "[EoH] Offspring generation completed",
            extra={
                "operator": str(operator),
                "n_offspring": len(offspring_list),
                "objectives": [float(obj) for obj in objs if obj is not None],
                **self._logging_context(),
            },
        )

        results = offspring_list

        out_p: list[list[EOHIndividual]] = []
        out_off: list[EOHIndividual] = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
        return out_p, out_off


def select_parents(pop: list[EOHIndividual], m: int) -> list[EOHIndividual]:
    ranks = [i for i in range(len(pop))]
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    parents = np.random.choice(
        np.array(pop), p=probs / np.sum(probs), size=m, replace=True
    ).tolist()
    return parents
