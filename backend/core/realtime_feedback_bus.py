"""
backend/core/realtime_feedback_bus.py — Real-time event bus with latency monitoring
====================================================================================

Lightweight synchronous event bus for real-time UI feedback callbacks.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

_logger = logging.getLogger(__name__)

_LATENCY_THRESHOLD_MS: float = 10.0


class RealtimeFeedbackBus:
    """Event bus for real-time parameter/feedback updates.

    Listeners receive (event, data) synchronous callbacks.
    Latency warnings are printed when a listener exceeds 10 ms.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[str, Any], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, listener: Callable[[str, Any], None]) -> None:
        """Registriert *listener* for all events."""
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[str, Any], None]) -> None:
        """Entfernt *listener*."""
        with self._lock, contextlib.suppress(ValueError):
            self._listeners.remove(listener)

    def notify(self, event: str, data: Any) -> None:
        """Benachrichtigt all listeners with *event* and *data*.

        Prints a German warning when a listener call exceeds 10 ms.
        """
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            t0 = time.perf_counter()
            with contextlib.suppress(Exception):
                listener(event, data)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if elapsed_ms > _LATENCY_THRESHOLD_MS:
                _logger.warning(
                    "[RealtimeFeedbackBus] Latenz für Listener %r überschreitet 10ms (%.1f ms).",
                    listener,
                    elapsed_ms,
                )

    def clear(self) -> None:
        """Entfernt all listeners."""
        with self._lock:
            self._listeners.clear()


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
_realtime_feedback_bus_instance: RealtimeFeedbackBus | None = None
_realtime_feedback_bus_lock = threading.Lock()


def get_realtime_feedback_bus() -> RealtimeFeedbackBus:
    """Gibt the process-wide singleton ``RealtimeFeedbackBus`` instance zurück."""
    global _realtime_feedback_bus_instance
    if _realtime_feedback_bus_instance is None:
        with _realtime_feedback_bus_lock:
            if _realtime_feedback_bus_instance is None:
                _realtime_feedback_bus_instance = RealtimeFeedbackBus()
    return _realtime_feedback_bus_instance
