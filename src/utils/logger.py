"""
logger.py
=========
Centralised, colour-aware logging used by every module in the platform.

Design goals
------------
* One call -> a fully configured logger (console + rotating file).
* Idempotent: importing/configuring twice never duplicates handlers.
* Safe on Windows terminals (no hard dependency on ANSI support).

Usage
-----
>>> from src.utils.logger import get_logger
>>> log = get_logger(__name__)
>>> log.info("pipeline started")
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_DEFAULT_FMT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Track configured loggers so handlers are attached only once.
_CONFIGURED: set[str] = set()


class _ColourFormatter(logging.Formatter):
    """Adds ANSI colour to console output; falls back to plain text."""

    COLOURS = {
        "DEBUG": "\033[37m",      # grey
        "INFO": "\033[36m",       # cyan
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31m",      # red
        "CRITICAL": "\033[41m",   # red background
    }
    RESET = "\033[0m"

    def __init__(self, use_colour: bool, fmt: str = _DEFAULT_FMT) -> None:
        super().__init__(fmt, datefmt=_DATE_FMT)
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        msg = super().format(record)
        if self.use_colour:
            colour = self.COLOURS.get(record.levelname, "")
            return f"{colour}{msg}{self.RESET}"
        return msg


def get_logger(
    name: str = "traffic",
    level: int = logging.INFO,
    log_dir: Optional[str | Path] = "logs",
    log_file: str = "pipeline.log",
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : module name (usually ``__name__``).
    level : logging level for the console.
    log_dir : directory for the rotating file handler (None disables file logs).
    log_file : file name inside ``log_dir``.
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # --- console handler -------------------------------------------------
    use_colour = sys.stdout.isatty()
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ColourFormatter(use_colour))
    logger.addHandler(console)

    # --- rotating file handler ------------------------------------------
    if log_dir is not None:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            fileh = RotatingFileHandler(
                log_path / log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            fileh.setLevel(logging.DEBUG)
            fileh.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DATE_FMT))
            logger.addHandler(fileh)
        except OSError:
            # Read-only environments (e.g. some cloud sandboxes) -> console only.
            logger.warning("Could not create file handler; logging to console only.")

    _CONFIGURED.add(name)
    return logger
