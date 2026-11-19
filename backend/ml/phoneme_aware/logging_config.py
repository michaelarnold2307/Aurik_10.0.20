"""
Logging-Konfiguration für das phonem-bewusste Verarbeitungsmodul.

Provides structured logging for:
- Phoneme detection
- Classification
- Performance metrics
- Errors and warnings
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """
    Setzt up a logger for phoneme-aware processing.

    Args:
        name: Logger name (typically module name)
        level: Logging level (default: INFO)
        log_file: Optional file path for logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format: [TIMESTAMP] LEVEL MODULE - MESSAGE
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger for module
logger = setup_logger("phoneme_aware")
