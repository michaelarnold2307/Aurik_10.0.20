"""Shared ml_memory_budget guard for DSP ONNX/Torch model loads (§2.37 RELEASE_MUST)."""

import logging

logger = logging.getLogger(__name__)

_logger = logging.getLogger(__name__)


def check_budget(name: str, size_gb: float = 0.1) -> bool:
    """Gibt True if allocation succeeds or budget module unavailable zurück."""
    try:
        from backend.core.ml_memory_budget import try_allocate

        if not try_allocate(name, size_gb):
            _logger.warning("Memory budget exceeded for %s (%.2f GB) — DSP fallback", name, size_gb)
            return False
        return True
    except Exception:
        logger.warning("_memory_budget_guard.py::check_budget fallback", exc_info=True)
        return True


def release_budget(name: str) -> None:
    """Release previously allocated budget slot."""
    try:
        from backend.core.ml_memory_budget import release

        release(name)
    except Exception:
        _logger.debug("ml_memory_budget.release(%s) failed", name, exc_info=True)
