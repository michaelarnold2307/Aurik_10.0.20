"""
backend/core/expert_feedback_system.py — Expert feedback aggregator
===================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExpertAggregate:
    """Typed aggregate payload for expert feedback scores."""

    scores: dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.scores.get(key, default))


class ExpertFeedbackSystem:
    """Sammelt expert feedback and computes per-dimension averages."""

    def __init__(self) -> None:
        self._feedback: list[dict[str, float]] = []

    def add_feedback(self, expert: str, scores: dict[str, float]) -> None:
        """Fügt hinzu: *scores* from *expert*."""
        self._feedback.append(dict(scores))

    def aggregate(self) -> ExpertAggregate:
        """Gibt mean score per dimension across all expert feedback entries zurück."""
        if not self._feedback:
            return ExpertAggregate()
        keys = self._feedback[0].keys()
        return ExpertAggregate({k: sum(f.get(k, 0.0) for f in self._feedback) / len(self._feedback) for k in keys})


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
import threading as _threading

_expert_feedback_system_instance = None
_expert_feedback_system_lock = _threading.Lock()


def get_expert_feedback_system() -> ExpertFeedbackSystem:
    """Gibt the process-wide singleton ExpertFeedbackSystem instance zurück."""
    global _expert_feedback_system_instance
    if _expert_feedback_system_instance is None:
        with _expert_feedback_system_lock:
            if _expert_feedback_system_instance is None:
                _expert_feedback_system_instance = ExpertFeedbackSystem()
    return _expert_feedback_system_instance
