"""Shim — forwards to canonical backend.logging_config."""

from backend.logging_config import get_logger

__all__ = ["get_logger"]
