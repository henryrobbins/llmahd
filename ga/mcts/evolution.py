from enum import StrEnum
from pathlib import Path

from utils.problem import EOHProblemPrompts
from utils.llm_client.base import BaseClient
from utils.utils import file_to_string, parse_response


class MCTSOperator(StrEnum):
    I1 = "i1"
    E1 = "e1"
    E2 = "e2"
    M1 = "m1"
    M2 = "m2"
    S1 = "s1"


class Evolution:

    def __init__(self, llm_client: BaseClient, prompts: EOHProblemPrompts):

        self.prompts_dir = Path(__file__).parent / "prompts"
        self.prompts = prompts
        if len(self.prompts.func_inputs) > 1:
            self.joined_inputs = ", ".join(
                "'" + s + "'" for s in self.prompts.func_inputs
            )
        else:
            self.joined_inputs = "'" + self.prompts.func_inputs[0] + "'"

        if len(self.prompts.func_outputs) > 1:
            self.joined_outputs = ", ".join(
                "'" + s + "'" for s in self.prompts.func_outputs
            )
        else:
            self.joined_outputs = "'" + self.prompts.func_outputs[0] + "'"

        self.llm_client = llm_client

    def get_prompt_post(self, code: str) -> str:
        post = file_to_string(self.prompts_dir / "post.txt")
        return post.format(
            prompt_task=self.prompts.problem_desc,
            func_name=self.prompts.func_name,
            inout_inf=self.prompts.inout_inf,
            other_inf=self.prompts.other_inf,
            code=code,
        )

    def get_prompt_refine(self, code: str, algorithm: str) -> str:
        refine = file_to_string(self.prompts_dir / "refine.txt")
        return refine.format(
            prompt_task=self.prompts.problem_desc,
            func_name=self.prompts.func_name,
            inout_inf=self.prompts.inout_inf,
            other_inf=self.prompts.other_inf,
            algorithm=algorithm,
            code=code,
        )

    def get_prompt_i1(self) -> str:
        i1 = file_to_string(self.prompts_dir / "i1.txt")
        return i1.format(
            prompt_task=self.prompts.problem_desc,
            func_name=self.prompts.func_name,
            n_inputs=len(self.prompts.func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.inout_inf,
            other_inf=self.prompts.other_inf,
        )

    def get_prompt_e1(self, indivs: list[dict]) -> str:
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm's description, its corresponding code and its objective value are: \n"
                + indivs[i]["algorithm"]
                + "\n"
                + indivs[i]["code"]
                + "\n"
                + f"Objective value: {indivs[i]['objective']}"
                + "\n\n"
            )

        e1 = file_to_string(self.prompts_dir / "e1.txt")
        return e1.format(
            prompt_task=self.prompts.problem_desc,
            func_name=self.prompts.func_name,
            n_inputs=len(self.prompts.func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.inout_inf,
            other_inf=self.prompts.other_inf,
            n_indivs=len(indivs),
            prompt_indiv=prompt_indiv,
        )

    def get_prompt_e2(self, indivs: list[dict]) -> str:
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm's description, its corresponding code and its objective value are: \n"
                + indivs[i]["algorithm"]
                + "\n"
                + indivs[i]["code"]
                + "\n"
                + f"Objective value: {indivs[i]['objective']}"
                + "\n\n"
            )

        e2 = file_to_string(self.prompts_dir / "e2.txt")
        return e2.format(
            prompt_task=self.prompts.problem_desc,
            func_name=self.prompts.func_name,
            n_inputs=len(self.prompts.func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.inout_inf,
            other_inf=self.prompts.other_inf,
            n_indivs=len(indivs),
            prompt_indiv=prompt_indiv,
        )

    def get_prompt_m1(self, indiv1: dict) -> str:
        m1 = file_to_string(self.prompts_dir / "m1.txt")
        return m1.format(
            prompt_task=self.prompts.problem_desc,
            func_name=self.prompts.func_name,
            n_inputs=len(self.prompts.func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.inout_inf,
            other_inf=self.prompts.other_inf,
            indiv_algorithm=indiv1["algorithm"],
            indiv_code=indiv1["code"],
        )

    def get_prompt_m2(self, indiv1: dict) -> str:
        m2 = file_to_string(self.prompts_dir / "m2.txt")
        return m2.format(
            prompt_task=self.prompts.problem_desc,
            func_name=self.prompts.func_name,
            n_inputs=len(self.prompts.func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.inout_inf,
            other_inf=self.prompts.other_inf,
            indiv_algorithm=indiv1["algorithm"],
            indiv_code=indiv1["code"],
        )

    def get_prompt_s1(self, indivs: list[dict]) -> str:
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = (
                prompt_indiv
                + "No."
                + str(i + 1)
                + " algorithm's description, its corresponding code and its objective value are: \n"
                + indivs[i]["algorithm"]
                + "\n"
                + indivs[i]["code"]
                + "\n"
                + f"Objective value: {indivs[i]['objective']}"
                + "\n\n"
            )

        s1 = file_to_string(self.prompts_dir / "s1.txt")
        return s1.format(
            prompt_task=self.prompts.problem_desc,
            func_name=self.prompts.func_name,
            n_inputs=len(self.prompts.func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.inout_inf,
            other_inf=self.prompts.other_inf,
            n_indivs=len(indivs),
            prompt_indiv=prompt_indiv,
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
            print(
                "Error: algorithm or code not identified, wait 1 seconds and retrying ... "
            )

            response = chat_completion(
                client=self.llm_client, prompt_content=prompt_content
            )

            algorithms, code = parse_response(response)

            if n_retry > 3:
                break
            n_retry += 1

        # TODO: I believe this resolves a bug in original implementation
        algorithm = algorithms[0]
        code = code[0]
        code_all = code + " " + ", ".join(s for s in self.prompts.func_outputs)

        return code_all, algorithm

    def post_thought(self, code: str, algorithm: str) -> str:
        prompt_content = self.get_prompt_refine(code, algorithm)
        return self._get_thought(prompt_content)

    def i1(self) -> tuple[str, str]:
        prompt_content = self.get_prompt_i1()
        return self._get_alg(prompt_content)

    def e1(self, parents: list[dict]) -> tuple[str, str]:
        prompt_content = self.get_prompt_e1(parents)
        return self._get_alg(prompt_content)

    def e2(self, parents: list[dict]) -> tuple[str, str]:
        prompt_content = self.get_prompt_e2(parents)
        return self._get_alg(prompt_content)

    def m1(self, parents: dict) -> tuple[str, str]:
        prompt_content = self.get_prompt_m1(parents)
        return self._get_alg(prompt_content)

    def m2(self, parents: dict) -> tuple[str, str]:
        prompt_content = self.get_prompt_m2(parents)
        return self._get_alg(prompt_content)

    def s1(self, parents: list[dict]) -> tuple[str, str]:
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
