# Adapted from MCTS-AHD: https://github.com/zz1358m/MCTS-AHD-master/blob/main/source/evolution.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import logging
from enum import StrEnum
from dataclasses import dataclass
from importlib.resources import files

from llamda.individual import Individual
from llamda.problem import EohProblem
from llamda.llm_client.base import BaseClient
from llamda.utils import file_to_string, parse_response

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


class Evolution:

    def __init__(self, llm_client: BaseClient, problem: EohProblem):

        self.prompts_dir = files("llamda.prompts.ga.mcts")
        self.problem = problem
        if len(self.problem.func_inputs) > 1:
            self.joined_inputs = ", ".join(
                "'" + s + "'" for s in self.problem.func_inputs
            )
        else:
            self.joined_inputs = "'" + self.problem.func_inputs[0] + "'"

        if len(self.problem.func_outputs) > 1:
            self.joined_outputs = ", ".join(
                "'" + s + "'" for s in self.problem.func_outputs
            )
        else:
            self.joined_outputs = "'" + self.problem.func_outputs[0] + "'"

        self.llm_client = llm_client

    def get_prompt_post(self, code: str) -> str:
        post = file_to_string(self.prompts_dir / "post.txt")
        return post.format(
            description=self.problem.description,
            func_name=self.problem.func_name,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            code=code,
        )

    def get_prompt_refine(self, code: str, algorithm: str) -> str:
        refine = file_to_string(self.prompts_dir / "refine.txt")
        return refine.format(
            description=self.problem.description,
            func_name=self.problem.func_name,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            algorithm=algorithm,
            code=code,
        )

    def get_prompt_i1(self) -> str:
        i1 = file_to_string(self.prompts_dir / "i1.txt")
        return i1.format(
            description=self.problem.description,
            func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            func_outputs=self.joined_outputs,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
        )

    def get_prompt_e1(self, indivs: list[MCTSIndividual]) -> str:
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm's description, its corresponding code and "
                + " its objective value are: \n"
                + indivs[i].algorithm
                + "\n"
                + indivs[i].code
                + "\n"
                + f"Objective value: {indivs[i].obj}"
                + "\n\n"
            )

        e1 = file_to_string(self.prompts_dir / "e1.txt")
        return e1.format(
            description=self.problem.description,
            func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            func_outputs=self.joined_outputs,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            n_indivs=len(indivs),
            indivs=prompt_indiv,
        )

    def get_prompt_e2(self, indivs: list[MCTSIndividual]) -> str:
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm's description, its corresponding code "
                + "and its objective value are: \n"
                + indivs[i].algorithm
                + "\n"
                + indivs[i].code
                + "\n"
                + f"Objective value: {indivs[i].obj}"
                + "\n\n"
            )

        e2 = file_to_string(self.prompts_dir / "e2.txt")
        return e2.format(
            description=self.problem.description,
            func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            func_outputs=self.joined_outputs,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            n_indivs=len(indivs),
            indivs=prompt_indiv,
        )

    def get_prompt_m1(self, indiv1: MCTSIndividual) -> str:
        m1 = file_to_string(self.prompts_dir / "m1.txt")
        return m1.format(
            description=self.problem.description,
            func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            func_outputs=self.joined_outputs,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            algorithm=indiv1.algorithm,
            code=indiv1.code,
        )

    def get_prompt_m2(self, indiv1: MCTSIndividual) -> str:
        m2 = file_to_string(self.prompts_dir / "m2.txt")
        return m2.format(
            description=self.problem.description,
            func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            func_outputs=self.joined_outputs,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            algorithm=indiv1.algorithm,
            code=indiv1.code,
        )

    def get_prompt_s1(self, indivs: list[MCTSIndividual]) -> str:
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm's description, its corresponding code "
                + "and its objective value are: \n"
                + indivs[i].algorithm
                + "\n"
                + indivs[i].code
                + "\n"
                + f"Objective value: {indivs[i].obj}"
                + "\n\n"
            )

        s1 = file_to_string(self.prompts_dir / "s1.txt")
        return s1.format(
            description=self.problem.description,
            func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            func_outputs=self.joined_outputs,
            inout_info=self.problem.inout_info,
            other_info=self.problem.other_info,
            n_indivs=len(indivs),
            indivs=prompt_indiv,
        )

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
