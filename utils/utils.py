from datetime import datetime
from litellm import completion
import os
import logging
import concurrent.futures
import time
import re
import inspect


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def print_hyperlink(path, text=None):
    """Print hyperlink to file or folder for convenient navigation"""
    # Format: \033]8;;file:///path/to/file\033\\text\033]8;;\033\\
    text = text or path
    full_path = f"file://{os.path.abspath(path)}"
    return f"\033]8;;{full_path}\033\\{text}\033]8;;\033\\"


def filter_traceback(s):
    lines = s.split("\n")
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith("Traceback"):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return "\n".join(filtered_lines)
    return ""  # Return an empty string if no Traceback is found


def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if len(log) > 0:
            if log_status and "Traceback" in log:
                logging.warning(
                    f"Iteration {iter_num}: Code Run {response_id} execution error! (see {print_hyperlink(stdout_filepath, 'stdout')}))"
                )
            else:
                logging.info(
                    f"Iteration {iter_num}: Code Run {response_id} successful! (see {print_hyperlink(stdout_filepath, 'stdout')})"
                )
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r"<start>(.*?)```python", r"<start>(.*?)<end>"]
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


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


# eoh + mcts
def parse_response(response: str) -> tuple[list[str], list[str]]:
    # TODO: Why did MCTS make this change?
    # algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1)
    algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
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
    return algorithm, code


def filter_code(code_string):
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


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name


def multi_chat_completion(messages_list: list[list[dict]], n, model, temperature):
    """
    An example of messages_list:

    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {"role": "system", "content": "You are a knowledgeable guide."},
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "system", "content": "You are a witty comedian."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    ]
    param: n: number of responses to generate for each message in messages_list
    """
    # If messages_list is not a list of list (i.e., only one conversation), convert it to a list of list
    assert isinstance(messages_list, list), "messages_list should be a list."
    try:
        if not isinstance(messages_list[0], list):
            messages_list = [messages_list]
    except:
        print(messages_list)
        raise IndexError("Something is wrong.")

    if len(messages_list) > 1:
        assert n == 1, "Currently, only n=1 is supported for multi-chat completion."

    num_workers = os.cpu_count()
    if "gpt" not in model:
        # Transform messages if n > 1
        messages_list *= n
        n = 1
        num_workers = 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        args = [(n, messages, model, temperature) for messages in messages_list]
        choices = executor.map(lambda p: chat_completion(*p), args)

    contents: list[str] = []
    for choice in choices:
        for c in choice:
            contents.append(c.message.content)
    return contents


def chat_completion(
    n: int, messages: list[dict], model: str, temperature: float
) -> list[dict]:
    """
    Generate n responses using OpenAI Chat Completions API
    """

    for attempt in range(30):
        try:
            response_cur = completion(
                model=model, messages=messages, temperature=temperature, n=n
            )
            break
        except Exception as e:
            logging.info(f"Attempt {attempt + 1} failed with error: {e}")
            time.sleep(3)
    if response_cur is None:
        logging.info("Code terminated due to too many failed attempts!")
        exit()

    return response_cur.choices


def extract_to_hs(input_string: str):
    code_blocks = input_string.split("```python\n")[1:]

    try:
        parameter_ranges_block = (
            "import numpy as np\n" + code_blocks[1].split("```")[0].strip()
        )
        if any(
            keyword in parameter_ranges_block for keyword in ["inf", "np.inf", "None"]
        ):
            return None, None
        exec_globals = {}
        exec(parameter_ranges_block, exec_globals)
        parameter_ranges = exec_globals["parameter_ranges"]
    except:
        return None, None

    function_block = code_blocks[0].split("```")[0].strip()

    paren_count = 0
    in_signature = False
    signature_start_index = None
    signature_end_index = None

    # Loop through the function block to find the start and end of the function signature
    for i, char in enumerate(function_block):
        if char == "d" and function_block[i : i + 3] == "def":
            in_signature = True
            signature_start_index = i
        if in_signature:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            if char == ":" and paren_count == 0:
                signature_end_index = i
                break

    if signature_start_index is not None and signature_end_index is not None:
        function_signature = function_block[
            signature_start_index : signature_end_index + 1
        ]
        for param in parameter_ranges:
            pattern = rf"(\b{param}\b[^=]*=)[^,)]+"
            replacement = r"\1 {" + param + "}"
            function_signature = re.sub(
                pattern, replacement, function_signature, flags=re.DOTALL
            )
        function_block = (
            function_block[:signature_start_index]
            + function_signature
            + function_block[signature_end_index + 1 :]
        )

    return parameter_ranges, function_block


def format_messages(pre_messages):
    messages = [
        {"role": "system", "content": pre_messages["system"]},
        {"role": "user", "content": pre_messages["user"]},
    ]
    return messages


def get_output_dir(job_name: str, root_dir: str) -> str:
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{root_dir}/outputs/{job_name}/{formatted_time}"
