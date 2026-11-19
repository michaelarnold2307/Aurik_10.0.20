"""
Logging-Konfiguration für das Vokal-Separationsmodul.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Gibt zurück: configured logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
