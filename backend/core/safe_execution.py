"""
safe_execution.py — §v10 Structured Error Reporting
=====================================================

Ersetzt stille `except Exception: pass`-Handler durch strukturiertes
Error-Reporting mit Kontext (Datei, Zeile, Funktion, Fehlertyp).

Verhindert stille Degradation: Fehler in ML-Modellen, DSP-Berechnungen
oder Datei-Operationen werden protokolliert statt verschluckt.

Usage:
    from backend.core.safe_execution import safe_call, SafeExecutionContext

    # Als Context Manager:
    with SafeExecutionContext("phase_03_denoise") as ctx:
        result = potentially_failing_operation()
    if ctx.failed:
        logger.warning(f"Operation fehlgeschlagen: {ctx.last_error}")

    # Als Decorator:
    @safe_call("compute_stft")
    def compute_stft(audio, sr):
        ...
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Globaler Fehler-Zähler für Monitoring
_error_counts: dict[str, int] = defaultdict(int)
_error_counts_lock = threading.Lock()
_last_error_time: dict[str, float] = {}
_error_rate_limit_s: float = 60.0  # Max 1 Log-Eintrag pro Key pro Minute


def _rate_limited_log(key: str, message: str, level: int = logging.WARNING) -> None:
    """Verhindert Log-Flooding durch Rate-Limiting pro Error-Key."""
    now = time.monotonic()
    last = _last_error_time.get(key, 0.0)
    if now - last >= _error_rate_limit_s:
        _last_error_time[key] = now
        logger.log(level, message)


class SafeExecutionContext:
    """Context Manager für strukturiertes Error-Reporting.

    Erfasst Exceptions und protokolliert sie mit vollständigem Kontext,
    ohne die Exception zu verschlucken (re-raise optional).
    """

    def __init__(
        self,
        context: str = "unknown",
        re_raise: bool = False,
        log_level: int = logging.WARNING,
        rate_limit: bool = True,
    ):
        self._context = context
        self._re_raise = re_raise
        self._log_level = log_level
        self._rate_limit = rate_limit
        self.failed = False
        self.last_error: Exception | None = None
        self.last_error_type: str = ""

    def __enter__(self) -> SafeExecutionContext:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_value is not None:
            self.failed = True
            self.last_error = exc_value
            self.last_error_type = type(exc_value).__name__

            # Hole Aufrufer-Kontext
            caller_frame = inspect.currentframe()
            if caller_frame is not None:
                # Gehe 2 Frames zurück: __exit__ → with-Block → Aufrufer
                outer = caller_frame.f_back
                if outer is not None:
                    outer = outer.f_back
                if outer is not None:
                    caller_file = outer.f_code.co_filename
                    caller_line = outer.f_lineno
                    caller_func = outer.f_code.co_name
                else:
                    caller_file = "unknown"
                    caller_line = 0
                    caller_func = "unknown"
            else:
                caller_file = "unknown"
                caller_line = 0
                caller_func = "unknown"

            key = f"{caller_file}:{caller_line}:{self._context}"
            with _error_counts_lock:
                _error_counts[key] += 1
            count = _error_counts[key]

            message = (
                f"§ERR {self._context}: {self.last_error_type} in "
                f"{caller_func} ({caller_file}:{caller_line}) "
                f"[occurrence #{count}]: {exc_value!r}"
            )

            if self._rate_limit:
                _rate_limited_log(key, message, self._log_level)
            else:
                logger.log(self._log_level, message)

            if self._re_raise:
                return False  # Exception weitergeben
            return True  # Exception unterdrückt, aber protokolliert

        return False  # Keine Exception


def safe_call(
    context: str = "unknown",
    re_raise: bool = False,
    log_level: int = logging.WARNING,
) -> Callable:
    """Decorator: Fängt Exceptions und protokolliert sie mit Kontext.

    Args:
        context: Kontext-Name für Logging (z.B. Funktionsname)
        re_raise: True = Exception weiterwerfen, False = unterdrücken
        log_level: Log-Level für Fehler (default: WARNING)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SafeExecutionContext(
                context=context or func.__name__,
                re_raise=re_raise,
                log_level=log_level,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_error_statistics() -> dict:
    """Gibt Fehler-Statistiken für Monitoring zurück."""
    with _error_counts_lock:
        return {
            "total_errors": sum(_error_counts.values()),
            "unique_error_sites": len(_error_counts),
            "top_errors": sorted(_error_counts.items(), key=lambda x: -x[1])[:10],
        }


def reset_error_statistics() -> None:
    """Setzt Fehler-Statistiken zurück (für Test-Cleanup)."""
    with _error_counts_lock:
        _error_counts.clear()
        _last_error_time.clear()
