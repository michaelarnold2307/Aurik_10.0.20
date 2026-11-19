"""§v10.17 PhaseErrorRegistry — strukturierte Fehler-Taxonomie."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhaseError:
    phase_id: str
    error_type: str  # "import", "runtime", "regression", "pss", "timeout", "oom", "shape"
    message: str = ""
    retry_count: int = 0
    severity: str = "warning"  # "warning", "error", "fatal"


class PhaseErrorRegistry:
    def __init__(self):
        self._errors: list[PhaseError] = []
        self._lock = threading.Lock()

    def record(self, phase_id: str, error_type: str, message: str = "", retries: int = 0, severity: str = "warning"):
        with self._lock:
            self._errors.append(
                PhaseError(
                    phase_id=phase_id,
                    error_type=error_type,
                    message=str(message)[:200],
                    retry_count=retries,
                    severity=severity,
                )
            )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_type: dict[str, int] = {}
            by_severity: dict[str, int] = {}
            by_phase: dict[str, int] = {}
            for e in self._errors:
                by_type[e.error_type] = by_type.get(e.error_type, 0) + 1
                by_severity[e.severity] = by_severity.get(e.severity, 0) + 1
                by_phase[e.phase_id] = by_phase.get(e.phase_id, 0) + 1
            return {
                "total_errors": len(self._errors),
                "by_type": by_type,
                "by_severity": by_severity,
                "by_phase": dict(sorted(by_phase.items(), key=lambda x: -x[1])[:10]),
                "last_errors": [
                    {"phase": e.phase_id, "type": e.error_type, "msg": e.message[:80]} for e in self._errors[-5:]
                ],
            }

    @property
    def error_count(self) -> int:
        return len(self._errors)


_instance: PhaseErrorRegistry | None = None
_lock = threading.Lock()


def get_error_registry() -> PhaseErrorRegistry:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PhaseErrorRegistry()
    return _instance
