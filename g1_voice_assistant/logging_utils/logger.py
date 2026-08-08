"""
logging_utils/logger.py
-----------------------
Centralised logger factory for the G1 Voice Assistant.

Every module should call ``get_logger(__name__)`` to get a properly
configured logger.  The root handler is configured once at import time.

Python 3.8 compatible.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


_ROOT_LOGGER_NAME = "g1_voice"
_handler_installed = False


def _ensure_root_handler(level: str = "INFO", log_dir: Optional[Path] = None) -> None:
    """Install stream + optional file handler on the root logger (once)."""
    global _handler_installed
    if _handler_installed:
        return
    _handler_installed = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(numeric_level)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Optional rotating file handler
    if log_dir is not None:
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / "g1_voice.log", encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError as exc:
            root.warning("Could not create log file in %s: %s", log_dir, exc)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a child logger of ``g1_voice``.

    Parameters
    ----------
    name:
        Module ``__name__`` – used as the logger suffix.
    level:
        Override log level for this specific logger (optional).
    """
    # Ensure the root handler is set up with env-driven defaults
    _ensure_root_handler(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        log_dir=Path(os.environ.get("LOG_DIR", "./logs")),
    )

    logger = logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
    if level is not None:
        numeric = getattr(logging, level.upper(), None)
        if numeric is not None:
            logger.setLevel(numeric)
    return logger
