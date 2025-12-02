# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/utils/utils.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import re
import logging
from typing import TypeVar

from llamda.individual import Individual
from llamda.llm_client.base import BaseClient

logger = logging.getLogger("llamda")


class LLMParsingError(Exception):
    """Raised when unable to parse expected content from LLM response."""

    def __init__(self, message: str, response: str):
        super().__init__(message)
        self.response = response


# reevo + hsevo
def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r"```python(.*?)```"
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split("\n")
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith("def"):
                start = i
            if "return" in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = "\n".join(lines[start : end + 1])

    if code_string is None:
        return None
    # Add import statements if not present
    # TODO: Using HSEvo convention for this (other methods used commented out code)
    # if "np" in code_string:
    #     code_string = "import numpy as np\n" + code_string
    # if "torch" in code_string:
    #     code_string = "import torch\n" + code_string
    if "import" not in code_string:
        code_string = (
            "import numpy as np\nimport random\nimport math\nimport scipy\nimport torch\n"
            + code_string
        )
    return code_string


# EoH + MCTS-AHD
def _extract_thoughts_and_code_fragments(response: str) -> tuple[list[str], list[str]]:
    thoughts = re.findall(r"\{(.*)\}", response, re.DOTALL)
    if len(thoughts) == 0:
        if "python" in response:
            thoughts = re.findall(r"^.*?(?=python)", response, re.DOTALL)
        elif "import" in response:
            thoughts = re.findall(r"^.*?(?=import)", response, re.DOTALL)
        else:
            thoughts = re.findall(r"^.*?(?=def)", response, re.DOTALL)

    code_fragments = re.findall(r"import.*return", response, re.DOTALL)
    if len(code_fragments) == 0:
        code_fragments = re.findall(r"def.*return", response, re.DOTALL)
    return thoughts, code_fragments


def _chat_completion(llm_client: BaseClient, prompt_content: str) -> str:
    response = llm_client.chat_completion(
        1, [{"role": "user", "content": prompt_content}]
    )
    response_content = response[0].message.content
    return response_content


def generate_thought_and_code(
    prompt_content: str, func_outputs: list[str], llm_client: BaseClient
) -> tuple[str, str, str]:
    """Call LLM with prompt and parse response into thought and code."""

    response = ""
    thoughts: list[str] = []
    code_fragments: list[str] = []

    for i in range(3):
        response = _chat_completion(llm_client, prompt_content)
        thoughts, code_fragments = _extract_thoughts_and_code_fragments(response)
        if len(code_fragments) > 0 and len(thoughts) > 0:
            break
        else:
            logger.warning(
                "Failed to extract thought and code from response",
                extra={
                    "attempt": i + 1,
                    "response": response,
                    "thoughts": thoughts,
                    "code_fragments": code_fragments,
                },
            )

    if len(code_fragments) == 0 or len(thoughts) == 0:
        raise LLMParsingError(
            "Unable to extract thought and code from LLM response", response
        )

    thought = thoughts[0]  # Take the first thought
    code = code_fragments[0] + " " + ", ".join(s for s in func_outputs)

    logger.debug(
        "Thought and code successfully extracted from LLM response",
        extra={"thought": thought, "code": code},
    )

    return response, thought, code


def filter_code(code_string: str) -> str:
    """Remove lines containing signature and import statements."""
    lines = code_string.split("\n")
    filtered_lines = []
    for line in lines:
        if line.startswith("def"):
            continue
        elif line.startswith("import"):
            continue
        elif line.startswith("from"):
            continue
        elif line.startswith("return"):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = "\n".join(filtered_lines)
    return code_string


T = TypeVar("T", bound=Individual)


def hydrate_individual(
    individual: T,
    response_id: int,
    output_dir: str,
    iteration: int = 0,
    file_name: str | None = None,
) -> T:

    # Write response to file
    file_name = (
        f"problem_iter{iteration}_response{response_id}.txt"
        if file_name is None
        else file_name + ".txt"
    )
    file_name = f"{output_dir}/{file_name}"
    with open(file_name, "w", encoding="utf-8") as file:
        file.writelines(individual.code + "\n")

    # Extract code and description from response
    std_out_filepath = (
        f"problem_iter{iteration}_stdout{response_id}.txt"
        if file_name is None
        else file_name.rstrip(".txt") + "_stdout.txt"
    )

    individual.stdout_filepath = std_out_filepath
    individual.code_path = f"problem_iter{iteration}_code{response_id}.py"
    individual.response_id = response_id

    return individual
