from dataclasses import dataclass


@dataclass
class Individual:
    stdout_filepath: str | None = None
    code_path: str | None = None
    code: str | None = None
    response_id: int | None = None
    exec_success: bool | None = None
    obj: float | None = None
    traceback_msg: str | None = None
