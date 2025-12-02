# Adapted from MCTS-AHD: https://github.com/zz1358m/MCTS-AHD-master/blob/main/source/evolution.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging
from enum import StrEnum
from dataclasses import dataclass

from jinja2 import Environment, PackageLoader, StrictUndefined

from llamda.individual import Individual
from llamda.problem import EohProblem
from llamda.llm_client.base import BaseClient
from llamda.utils import parse_response

logger = logging.getLogger("llamda")


@dataclass
class MCTSIndividual(Individual):
    algorithm: str | None = None
    thought: str | None = None


class MCTSOperator(StrEnum):
    I1 = "i1"
    E1 = "e1"
    E2 = "e2"
    M1 = "m1"
    M2 = "m2"
    S1 = "s1"


def quote_and_join(items: list[str]) -> str:
    """Quote each item and join with commas"""
    if len(items) > 1:
        return ", ".join(f"'{s}'" for s in items)
    else:
        return f"'{items[0]}'"


class Evolution:

    def __init__(self, llm_client: BaseClient, problem: EohProblem):

        self.env = Environment(
            loader=PackageLoader("llamda.prompts.ga", "mcts"), undefined=StrictUndefined
        )
        self.env.filters["quote_and_join"] = quote_and_join

        self.problem = problem
        self.llm_client = llm_client

    def _get_problem_context(self) -> dict:
        """Get the common problem context for all templates."""
        return {
            "description": self.problem.description,
            "func_name": self.problem.func_name,
            "func_inputs": self.problem.func_inputs,
            "func_outputs": self.problem.func_outputs,
            "inout_info": self.problem.inout_info,
            "other_info": self.problem.other_info,
        }

    def get_prompt_post(self, code: str) -> str:
        template = self.env.get_template("post.j2")
        return template.render(
            description=self.problem.description,
            func_name=self.problem.func_name,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            code=code,
        )

    def get_prompt_refine(self, code: str, algorithm: str) -> str:
        template = self.env.get_template("refine.j2")
        return template.render(
            description=self.problem.description,
            func_name=self.problem.func_name,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            algorithm=algorithm,
            code=code,
        )

    def get_prompt_i1(self) -> str:
        template = self.env.get_template(f"{MCTSOperator.I1.value}.j2")
        return template.render(**self._get_problem_context())

    def get_prompt_e1(self, indivs: list[MCTSIndividual]) -> str:
        template = self.env.get_template(f"{MCTSOperator.E1.value}.j2")
        return template.render(**self._get_problem_context(), indivs=indivs)

    def get_prompt_e2(self, indivs: list[MCTSIndividual]) -> str:
        template = self.env.get_template(f"{MCTSOperator.E2.value}.j2")
        return template.render(**self._get_problem_context(), indivs=indivs)

    def get_prompt_m1(self, indiv1: MCTSIndividual) -> str:
        template = self.env.get_template(f"{MCTSOperator.M1.value}.j2")
        return template.render(
            **self._get_problem_context(),
            algorithm=indiv1.algorithm,
            code=indiv1.code,
        )

    def get_prompt_m2(self, indiv1: MCTSIndividual) -> str:
        template = self.env.get_template(f"{MCTSOperator.M2.value}.j2")
        return template.render(
            **self._get_problem_context(),
            algorithm=indiv1.algorithm,
            code=indiv1.code,
        )

    def get_prompt_s1(self, indivs: list[MCTSIndividual]) -> str:
        template = self.env.get_template(f"{MCTSOperator.S1.value}.j2")
        return template.render(**self._get_problem_context(), indivs=indivs)

    def _get_thought(self, prompt_content: str) -> str:

        response = chat_completion(
            client=self.llm_client, prompt_content=prompt_content, temperature=0
        )

        # algorithm = response.split(':')[-1]
        return response

    def _get_alg(self, prompt_content: str) -> tuple[str, str]:

        response = chat_completion(
            client=self.llm_client, prompt_content=prompt_content
        )
        algorithms, code = parse_response(response)

        n_retry = 1
        while len(algorithms) == 0 or len(code) == 0:
            logger.warning(f"Algorithm or code not identified, retrying ({n_retry}/3)")

            response = chat_completion(
                client=self.llm_client, prompt_content=prompt_content
            )

            algorithms, code = parse_response(response)

            if n_retry > 3:
                logger.warning("Max retries reached, algorithm generation failed")
                break
            n_retry += 1

        # TODO: I believe this resolves a bug in original implementation
        algorithm = algorithms[0]
        code_all = code[0] + " " + ", ".join(s for s in self.problem.func_outputs)

        logger.debug(
            "Algorithm generated successfully",
            extra={"algorithm": algorithm, "code": code},
        )

        return code_all, algorithm

    def post_thought(self, code: str, algorithm: str) -> str:
        prompt_content = self.get_prompt_refine(code, algorithm)
        return self._get_thought(prompt_content)

    def i1(self) -> tuple[str, str]:
        prompt_content = self.get_prompt_i1()
        return self._get_alg(prompt_content)

    def e1(self, parents: list[MCTSIndividual]) -> tuple[str, str]:
        prompt_content = self.get_prompt_e1(parents)
        return self._get_alg(prompt_content)

    def e2(self, parents: list[MCTSIndividual]) -> tuple[str, str]:
        prompt_content = self.get_prompt_e2(parents)
        return self._get_alg(prompt_content)

    def m1(self, parents: MCTSIndividual) -> tuple[str, str]:
        prompt_content = self.get_prompt_m1(parents)
        return self._get_alg(prompt_content)

    def m2(self, parents: MCTSIndividual) -> tuple[str, str]:
        prompt_content = self.get_prompt_m2(parents)
        return self._get_alg(prompt_content)

    def s1(self, parents: list[MCTSIndividual]) -> tuple[str, str]:
        prompt_content = self.get_prompt_s1(parents)
        return self._get_alg(prompt_content)


def chat_completion(
    client: BaseClient, prompt_content: str, temperature: int | None = None
) -> str:
    response = client.chat_completion(
        n=1,
        messages=[{"role": "user", "content": prompt_content}],
        temperature=temperature,
    )
    return response[0].message.content
