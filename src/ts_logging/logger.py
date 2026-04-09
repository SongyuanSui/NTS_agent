# src/logging/logger.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_logger(
    name: str,
    logs_dir: str | Path,
    log_filename: str = "run.log",
    level: int = logging.INFO,
    enable_console: bool = True,
    enable_file: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """
    Build and return a configured logger.

    Parameters
    ----------
    name:
        Logger name.
    logs_dir:
        Directory where log files will be written.
    log_filename:
        Log file name under logs_dir.
    level:
        Logging level, e.g. logging.INFO.
    enable_console:
        Whether to print logs to stdout/stderr stream.
    enable_file:
        Whether to write logs to file.
    propagate:
        Whether this logger should propagate to parent loggers.

    Notes
    -----
    - This function clears existing handlers for the same logger name to avoid
      duplicate logs during repeated script execution.
    - Path creation is allowed here only for the already-resolved logs_dir.
      It does NOT decide the experiment path layout.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Logger name must be a non-empty string.")

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Clear old handlers to prevent duplicate logging in notebooks / repeated runs.
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    if enable_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if enable_file:
        file_handler = logging.FileHandler(logs_dir / log_filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_child_logger(parent_logger: logging.Logger, child_name: str) -> logging.Logger:
    """
    Return a child logger under an existing logger hierarchy.

    Example
    -------
    parent: ts_agent.run
    child_name: retrieval
    -> ts_agent.run.retrieval
    """
    if not isinstance(parent_logger, logging.Logger):
        raise TypeError("parent_logger must be a logging.Logger.")
    if not isinstance(child_name, str) or not child_name.strip():
        raise ValueError("child_name must be a non-empty string.")

    return logging.getLogger(f"{parent_logger.name}.{child_name}")


def log_config(logger: logging.Logger, config: dict, title: str = "Config") -> None:
    """
    Log a config dict in a readable way.
    """
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger.")
    if not isinstance(config, dict):
        raise TypeError("config must be a dict.")

    logger.info("%s:", title)
    for key in sorted(config.keys()):
        logger.info("  %s = %r", key, config[key])


def log_section(logger: logging.Logger, title: str, char: str = "=") -> None:
    """
    Log a visual section divider.
    """
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string.")
    if not isinstance(char, str) or len(char) != 1:
        raise ValueError("char must be a single-character string.")

    line = char * max(10, len(title) + 8)
    logger.info(line)
    logger.info(title)
    logger.info(line)