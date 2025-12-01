import sys
import json
import logging
from copy import copy
from typing import Literal
import datetime as dt

import click

TRACE_LOG_LEVEL = 5
LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}
LOG_RECORD_IGNORE_ARGS = {
    "color_message",  # uvicorn colored_message
}


# Modified version of uvicorn.logging.ColourizedFormatter
class ColorizedFormatter(logging.Formatter):
    """
    A custom log formatter class that:

    * Outputs the LOG_LEVEL with an appropriate color.
    * If a log call includes an `extras={"color_message": ...}` it will be used
      for formatting the output, instead of the plain text message.
    """

    level_name_colors = {
        TRACE_LOG_LEVEL: lambda level_name: click.style(
            str(level_name), bold=True, fg="blue"
        ),
        logging.DEBUG: lambda level_name: click.style(
            str(level_name), bold=True, fg="cyan"
        ),
        logging.INFO: lambda level_name: click.style(
            str(level_name), bold=True, fg="green"
        ),
        logging.WARNING: lambda level_name: click.style(
            str(level_name), bold=True, fg="yellow"
        ),
        logging.ERROR: lambda level_name: click.style(
            str(level_name), bold=True, fg="red"
        ),
        logging.CRITICAL: lambda level_name: click.style(
            str(level_name), bold=True, fg="bright_red"
        ),
    }

    @staticmethod
    def arg_color(arg: str, color: str) -> str:
        return click.style(str(arg), fg=color)

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
        use_colors: bool | None = None,
    ):
        if use_colors in (True, False):
            self.use_colors = use_colors
        else:
            self.use_colors = sys.stdout.isatty()
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def color_level_name(self, level_name: str, level_no: int) -> str:
        def default(level_name: str) -> str:
            return str(level_name)  # pragma: no cover

        func = self.level_name_colors.get(level_no, default)
        return func(level_name)

    def should_use_colors(self) -> bool:
        return True  # pragma: no cover

    def formatMessage(self, record: logging.LogRecord) -> str:
        recordcopy = copy(record)
        levelname = recordcopy.levelname
        if self.use_colors:
            levelname = self.color_level_name(levelname, recordcopy.levelno)
            if "color_message" in recordcopy.__dict__:
                recordcopy.msg = recordcopy.__dict__["color_message"]
                recordcopy.__dict__["message"] = recordcopy.getMessage()
        recordcopy.__dict__["levelname"] = levelname

        recordcopy.name = ColorizedFormatter.arg_color(str(recordcopy.name), "blue")
        recordcopy.asctime = ColorizedFormatter.arg_color(
            str(recordcopy.asctime), "green"
        )
        recordcopy.msg = ColorizedFormatter.arg_color(recordcopy.getMessage(), "white")

        return super().formatMessage(recordcopy)

    def format(self, record: logging.LogRecord) -> str:
        extras = {}
        for key, val in record.__dict__.items():
            if (
                key not in LOG_RECORD_BUILTIN_ATTRS
                and key not in LOG_RECORD_IGNORE_ARGS
            ):
                extras[key] = val
        if extras:
            record.msg = (
                f"{record.msg} {ColorizedFormatter.arg_color(str(extras), 'magenta')}"
            )
        return super().format(record)


# https://github.com/mCodingLLC/VideosSampleCode/blob/master/videos/135_modern_logging/mylogger.py
class MyJSONFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict:
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created)
            .astimezone()
            .isoformat(),
            "epoch_ms": int(record.created * 1000),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: (
                msg_val
                if (msg_val := always_fields.pop(val, None)) is not None
                else getattr(record, val)
            )
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        for key, val in record.__dict__.items():
            if (
                key not in LOG_RECORD_BUILTIN_ATTRS
                and key not in LOG_RECORD_IGNORE_ARGS
            ):
                message[key] = val

        return message
