# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/baselines/eoh/original/eoh_interface_EC.py
# Originally from EoH: https://github.com/FeiLiu36/EoH/blob/main/eoh/src/eoh/methods/eoh/eoh_interface_EC.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging
import numpy as np

from llamda.ga.eoh.eoh_evolution import EOHIndividual, EOHOperator, Evolution
from llamda.evaluate import Evaluator
from llamda.problem import EohProblem, hydrate_individual
from llamda.llm_client.base import BaseClient
from llamda.utils import parse_response

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
        self.evol = Evolution(problem=problem)
        self.llm_client = llm_client
        self.output_dir = output_dir

    def check_duplicate(self, population: list[EOHIndividual], code: str) -> bool:
        for ind in population:
            if code == ind.code:
                return True
        return False

    def _chat_completion(self, prompt_content: str) -> str:
        response = self.llm_client.chat_completion(
            1, [{"role": "user", "content": prompt_content}]
        )
        response_content = response[0].message.content
        return response_content

    def _call_llm_and_parse(self, prompt_content: str) -> tuple[str, str]:
        """Call LLM with prompt and parse response into algorithm and code."""
        response_content = self._chat_completion(prompt_content)
        algorithms, code = parse_response(response_content)

        n_retry = 1
        while len(algorithms) == 0 or len(code) == 0:
            logger.warning(f"Algorithm or code not identified, retrying ({n_retry}/3)")
            response_content = self._chat_completion(prompt_content)
            algorithms, code = parse_response(response_content)

            if n_retry > 3:
                logger.warning("Max retries reached, algorithm generation failed")
                break
            n_retry += 1

        algorithm = algorithms[0]
        code_all = code[0] + " " + ", ".join(s for s in self.evol.problem.func_outputs)

        logger.debug(
            "Algorithm generated successfully",
            extra={"algorithm": algorithm, "code": len(code)},
        )

        return code_all, algorithm

    def population_generation(self) -> list[EOHIndividual]:
        n_create = 2
        population = []
        for _ in range(n_create):
            _, pop = self.get_algorithm([], EOHOperator.I1)
            for p in pop:
                population.append(p)
        return population

    def population_generation_seed(
        self, seeds: list[EOHIndividual]
    ) -> list[EOHIndividual]:

        population = [
            hydrate_individual(indiv, i, self.output_dir)
            for i, indiv in enumerate(seeds)
        ]
        population = self.interface_eval.batch_evaluate(population)
        logger.info(
            "Initialization finished! Get " + str(len(seeds)) + " seed algorithms"
        )

        return population

    def _get_alg(
        self, pop: list[EOHIndividual], operator: EOHOperator
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

        logger.debug(f"Executing operator {operator.value}")
        code, algorithm = self._call_llm_and_parse(prompt_content)

        offspring = EOHIndividual(
            algorithm=algorithm,
            code=code,
            obj=None,
        )

        return parents, offspring

    def get_offspring(
        self, pop: list[EOHIndividual], operator: EOHOperator
    ) -> tuple[list[EOHIndividual], EOHIndividual]:

        try:
            p, offspring = self._get_alg(pop, operator)
            code = offspring.code
            n_retry = 1
            while code is None or self.check_duplicate(pop, code):
                n_retry += 1
                p, offspring = self._get_alg(pop, operator)
                code = offspring.code
                if n_retry > 1:
                    break

        except Exception as e:
            logger.error(e)

        return p, offspring

    def get_algorithm(
        self, pop: list[EOHIndividual], operator: EOHOperator
    ) -> tuple[list[list[EOHIndividual]], list[EOHIndividual]]:
        offspring_list: list[tuple[list[EOHIndividual], EOHIndividual]] = []
        for _ in range(self.pop_size):
            p, offspring = self.get_offspring(pop, operator)
            offspring_list.append((p, offspring))

        pop = [offspring for _, offspring in offspring_list]
        pop = [
            hydrate_individual(offspring, i, self.output_dir)
            for i, offspring in enumerate(pop)
        ]
        pop = self.interface_eval.batch_evaluate(pop, 0)
        objs = [indiv.obj for indiv in pop]
        for i, (_, offspring) in enumerate(offspring_list):
            offspring.obj = np.round(objs[i], 5)

        logger.debug(
            "Algorithm generation complete",
            extra={
                "operator": str(operator),
                "n_offspring": len(offspring_list),
                "objectives": [float(obj) for obj in objs if obj is not None],
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
