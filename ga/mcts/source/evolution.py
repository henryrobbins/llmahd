import re
from pathlib import Path
from typing import Dict, List

from ga.mcts.source.prompts.problem import ProblemPrompts
from utils.llm_client.base import BaseClient
from utils.utils import file_to_string

input = lambda: ...


class Evolution:

    def __init__(
        self, api_endpoint, api_key, model_LLM, debug_mode, prompts: ProblemPrompts
    ):

        self.prompts_dir = Path(__file__).parent / "prompts"
        self.prompts = prompts
        if len(self.prompts.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join(
                "'" + s + "'" for s in self.prompts.prompt_func_inputs
            )
        else:
            self.joined_inputs = "'" + self.prompts.prompt_func_inputs[0] + "'"

        if len(self.prompts.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join(
                "'" + s + "'" for s in self.prompts.prompt_func_outputs
            )
        else:
            self.joined_outputs = "'" + self.prompts.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

    def get_prompt_post(self, code, algorithm):
        post = file_to_string(self.prompts_dir / "post.txt")
        return post.format(
            prompt_task=self.prompts.prompt_task,
            func_name=self.prompts.prompt_func_name,
            inout_inf=self.prompts.prompt_inout_inf,
            other_inf=self.prompts.prompt_other_inf,
            code=code,
        )

    def get_prompt_refine(self, code, algorithm):
        refine = file_to_string(self.prompts_dir / "refine.txt")
        return refine.format(
            prompt_task=self.prompts.prompt_task,
            func_name=self.prompts.prompt_func_name,
            inout_inf=self.prompts.prompt_inout_inf,
            other_inf=self.prompts.prompt_other_inf,
            algorithm=algorithm,
            code=code,
        )

    def get_prompt_i1(self):
        i1 = file_to_string(self.prompts_dir / "i1.txt")
        return i1.format(
            prompt_task=self.prompts.prompt_task,
            func_name=self.prompts.prompt_func_name,
            n_inputs=len(self.prompts.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.prompt_inout_inf,
            other_inf=self.prompts.prompt_other_inf,
        )

    def get_prompt_e1(self, indivs):
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
            prompt_task=self.prompts.prompt_task,
            func_name=self.prompts.prompt_func_name,
            n_inputs=len(self.prompts.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.prompt_inout_inf,
            other_inf=self.prompts.prompt_other_inf,
            n_indivs=len(indivs),
            prompt_indiv=prompt_indiv,
        )

    def get_prompt_e2(self, indivs):
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
            prompt_task=self.prompts.prompt_task,
            func_name=self.prompts.prompt_func_name,
            n_inputs=len(self.prompts.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.prompt_inout_inf,
            other_inf=self.prompts.prompt_other_inf,
            n_indivs=len(indivs),
            prompt_indiv=prompt_indiv,
        )

    def get_prompt_m1(self, indiv1):
        m1 = file_to_string(self.prompts_dir / "m1.txt")
        return m1.format(
            prompt_task=self.prompts.prompt_task,
            func_name=self.prompts.prompt_func_name,
            n_inputs=len(self.prompts.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.prompt_inout_inf,
            other_inf=self.prompts.prompt_other_inf,
            indiv_algorithm=indiv1["algorithm"],
            indiv_code=indiv1["code"],
        )

    def get_prompt_m2(self, indiv1):
        m2 = file_to_string(self.prompts_dir / "m2.txt")
        return m2.format(
            prompt_task=self.prompts.prompt_task,
            func_name=self.prompts.prompt_func_name,
            n_inputs=len(self.prompts.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.prompt_inout_inf,
            other_inf=self.prompts.prompt_other_inf,
            indiv_algorithm=indiv1["algorithm"],
            indiv_code=indiv1["code"],
        )

    def get_prompt_s1(self, indivs):
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
            prompt_task=self.prompts.prompt_task,
            func_name=self.prompts.prompt_func_name,
            n_inputs=len(self.prompts.prompt_func_inputs),
            joined_inputs=self.joined_inputs,
            n_outputs=len(self.prompts.prompt_func_outputs),
            joined_outputs=self.joined_outputs,
            inout_inf=self.prompts.prompt_inout_inf,
            other_inf=self.prompts.prompt_other_inf,
            n_indivs=len(indivs),
            prompt_indiv=prompt_indiv,
        )

    def _get_thought(self, prompt_content):

        response = chat_completion(
            client=self.model_LLM, prompt_content=prompt_content, temperature=0
        )

        # algorithm = response.split(':')[-1]
        return response

    def _get_alg(self, prompt_content):

        response = chat_completion(client=self.model_LLM, prompt_content=prompt_content)

        algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1)
        if len(algorithm) == 0:
            if "python" in response:
                algorithm = re.findall(r"^.*?(?=python)", response, re.DOTALL)
            elif "import" in response:
                algorithm = re.findall(r"^.*?(?=import)", response, re.DOTALL)
            else:
                algorithm = re.findall(r"^.*?(?=def)", response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        while len(algorithm) == 0 or len(code) == 0:
            if self.debug_mode:
                print(
                    "Error: algorithm or code not identified, wait 1 seconds and retrying ... "
                )

            response = chat_completion(
                client=self.model_LLM, prompt_content=prompt_content
            )

            algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1)
            if len(algorithm) == 0:
                if "python" in response:
                    algorithm = re.findall(r"^.*?(?=python)", response, re.DOTALL)
                elif "import" in response:
                    algorithm = re.findall(r"^.*?(?=import)", response, re.DOTALL)
                else:
                    algorithm = re.findall(r"^.*?(?=def)", response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        code = code[0]
        code_all = code + " " + ", ".join(s for s in self.prompts.prompt_func_outputs)

        return [code_all, algorithm]

    def post_thought(self, code, algorithm):

        prompt_content = self.get_prompt_refine(code, algorithm)

        post_thought = self._get_thought(prompt_content)

        return post_thought

    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ i1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e1(self, parents):

        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ e1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e2(self, parents):

        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ e2 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m1(self, parents):

        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ m1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m2(self, parents):

        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ m2 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def s1(self, parents):

        prompt_content = self.get_prompt_s1(parents)

        if self.debug_mode:
            print(
                "\n >>> check prompt for creating algorithm using [ s1 ] : \n",
                prompt_content,
            )
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]


def chat_completion(
    client: BaseClient, prompt_content: str, temperature: int | None = None
) -> List[Dict]:
    response = client.chat_completion(
        n=1,
        messages=[{"role": "user", "content": prompt_content}],
        temperature=temperature,
    )
    return response[0].message.content
