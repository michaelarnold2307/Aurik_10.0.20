"""
Logging-Konfiguration für das Pitch-Korrekturmodul.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "pitch_correction") -> logging.Logger:
    """
    Configure structured logging for pitch correction operations.

    Args:
        name: Logger name (defaults to "pitch_correction")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Format: [timestamp] [level] [module] message
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Optional: File handler for audit trail
    log_dir = Path("logs/pitch_correction")
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "pitch_correction.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
