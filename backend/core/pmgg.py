"""
backend.core.pmgg — Alias-Modul für PerPhaseMusicalGoalsGate (Spec §2.29).

Kanonischer Importpfad laut Pipeline-Spec:
    from backend.core.pmgg import PerPhaseMusicalGoalsGate, get_phase_gate

Implementierung liegt in: backend/core/per_phase_musical_goals_gate.py
"""

from backend.core.per_phase_musical_goals_gate import (
    PerPhaseMusicalGoalsGate,
    PhaseGateLogEntry,
    PhaseGateResult,
    get_phase_gate,
    wrap_phase,
)

__all__ = [
    "PerPhaseMusicalGoalsGate",
    "PhaseGateLogEntry",
    "PhaseGateResult",
    "get_phase_gate",
    "wrap_phase",
]
