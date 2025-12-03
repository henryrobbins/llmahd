# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/utils/utils.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import os
import inspect
import importlib.util
from typing import Callable


def file_to_string(filename):
    with open(filename, "r") as file:
        return file.read()


def print_hyperlink(path, text=None):
    """Print hyperlink to file or folder for convenient navigation"""
    # Format: \033]8;;file:///path/to/file\033\\text\033]8;;\033\\
    text = text or path
    full_path = f"file://{os.path.abspath(path)}"
    return f"\033]8;;{full_path}\033\\{text}\033]8;;\033\\"


def _get_heuristic_name(module: object, possible_func_names: list[str]) -> str:
    """Get the name of the heuristic function from the module."""
    for func_name in possible_func_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name
    raise ValueError("No valid heuristic function found in the module.")


def load_heuristic_from_code(
    code_path: str, possible_func_names: list[str]
) -> Callable:
    """Dynamically load heuristic function from given code path."""
    spec = importlib.util.spec_from_file_location("gpt", code_path)
    gpt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpt)
    return getattr(gpt, _get_heuristic_name(gpt, possible_func_names))
