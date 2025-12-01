import logging
from dev.logging import ColorizedFormatter


def pytest_configure(config: dict) -> None:
    # Get the root logger or specific logger
    logger = logging.getLogger()

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create a handler with your ColorizedFormatter
    handler = logging.StreamHandler()
    formatter = ColorizedFormatter(
        fmt="[%(levelname)s] %(name)s | %(asctime)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        use_colors=True,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
