from enum import StrEnum
from dataclasses import dataclass
from importlib.resources import files

from llamda.individual import Individual
from llamda.problem import EohProblem
from llamda.llm_client.base import BaseClient
from llamda.utils import file_to_string, parse_response


@dataclass
class EOHIndividual(Individual):
    algorithm: str | None = None


class EOHOperator(StrEnum):
    I1 = "i1"
    E1 = "e1"
    E2 = "e2"
    M1 = "m1"
    M2 = "m2"


class Evolution:

    def __init__(self, llm_client: BaseClient, problem: EohProblem) -> None:

        self.prompts_dir = files("llamda.prompts.ga.eoh")
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

    def get_prompt_i1(self) -> str:
        i1 = file_to_string(self.prompts_dir / f"{EOHOperator.I1.value}.txt")
        return i1.format(
            prompt_task=self.problem.description,
            prompt_func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            prompt_func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            prompt_func_outputs=self.joined_outputs,
            prompt_inout_inf=self.problem.inout_info,
            prompt_other_inf=self.problem.other_info,
        )

    def get_prompt_e1(self, indivs: list[EOHIndividual]) -> str:
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm and the corresponding code are: \n"
                + indivs[i].algorithm
                + "\n"
                + indivs[i].code
                + "\n"
            )

        e1 = file_to_string(self.prompts_dir / f"{EOHOperator.E1.value}.txt")
        return e1.format(
            m=len(indivs),
            prompt_task=self.problem.description,
            prompt_func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            prompt_func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            prompt_func_outputs=self.joined_outputs,
            prompt_inout_inf=self.problem.inout_info,
            prompt_other_inf=self.problem.other_info,
            n_indivs=len(indivs),
            prompt_indiv=prompt_indiv,
        )

    def get_prompt_e2(self, indivs: list[EOHIndividual]) -> str:
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm and the corresponding code are: \n"
                + indivs[i].algorithm
                + "\n"
                + indivs[i].code
                + "\n"
            )

        e2 = file_to_string(self.prompts_dir / f"{EOHOperator.E2.value}.txt")
        return e2.format(
            m=len(indivs),
            prompt_task=self.problem.description,
            prompt_func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            prompt_func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            prompt_func_outputs=self.joined_outputs,
            prompt_inout_inf=self.problem.inout_info,
            prompt_other_inf=self.problem.other_info,
            n_indivs=len(indivs),
            prompt_indiv=prompt_indiv,
        )

    def get_prompt_m1(self, indiv1: EOHIndividual) -> str:
        m1 = file_to_string(self.prompts_dir / f"{EOHOperator.M1.value}.txt")
        return m1.format(
            prompt_task=self.problem.description,
            prompt_func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            prompt_func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            prompt_func_outputs=self.joined_outputs,
            prompt_inout_inf=self.problem.inout_info,
            prompt_other_inf=self.problem.other_info,
            algorithm=indiv1.algorithm,
            code=indiv1.code,
        )

    def get_prompt_m2(self, indiv1: EOHIndividual) -> str:
        m2 = file_to_string(self.prompts_dir / f"{EOHOperator.M2.value}.txt")
        return m2.format(
            prompt_task=self.problem.description,
            prompt_func_name=self.problem.func_name,
            n_inputs=len(self.problem.func_inputs),
            prompt_func_inputs=self.joined_inputs,
            n_outputs=len(self.problem.func_outputs),
            prompt_func_outputs=self.joined_outputs,
            prompt_inout_inf=self.problem.inout_info,
            prompt_other_inf=self.problem.other_info,
            algorithm=indiv1.algorithm,
            code=indiv1.code,
        )

    def _get_alg(self, prompt_content: str) -> tuple[str, str]:

        response = chat_completion(
            client=self.llm_client, prompt_content=prompt_content
        )
        algorithms, code = parse_response(response)

        n_retry = 1
        while len(algorithms) == 0 or len(code) == 0:

            response = chat_completion(
                client=self.llm_client, prompt_content=prompt_content
            )

            algorithms, code = parse_response(response)

            if n_retry > 3:
                break
            n_retry += 1

        algorithm = algorithms[0]
        code_all = code[0] + " " + ", ".join(s for s in self.problem.func_outputs)

        return code_all, algorithm

    def i1(self) -> tuple[str, str]:
        prompt_content = self.get_prompt_i1()
        return self._get_alg(prompt_content)

    def e1(self, parents: list[EOHIndividual]) -> tuple[str, str]:
        prompt_content = self.get_prompt_e1(parents)
        return self._get_alg(prompt_content)

    def e2(self, parents: list[EOHIndividual]) -> tuple[str, str]:
        prompt_content = self.get_prompt_e2(parents)
        return self._get_alg(prompt_content)

    def m1(self, parents: EOHIndividual) -> tuple[str, str]:
        prompt_content = self.get_prompt_m1(parents)
        return self._get_alg(prompt_content)

    def m2(self, parents: EOHIndividual) -> tuple[str, str]:
        prompt_content = self.get_prompt_m2(parents)
        return self._get_alg(prompt_content)


def chat_completion(client: BaseClient, prompt_content: str) -> str:
    response = client.chat_completion(1, [{"role": "user", "content": prompt_content}])
    return response[0].message.content
