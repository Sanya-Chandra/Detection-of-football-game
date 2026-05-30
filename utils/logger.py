"""
Sports Analytics CV — Logging Utility
Centralized logger configuration for the entire application.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = "logs/app.log", level: str = "INFO") -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: Logger name (usually __name__ of calling module)
        log_file: Path to log file
        level: Logging level string (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logging.Logger instance
    """
    parent = Path(log_file).parent
    try:
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # Prevent permission/sharing errors on Windows from crashing the app

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)-30s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass  # Fail silently if log file cannot be written

    return logger
