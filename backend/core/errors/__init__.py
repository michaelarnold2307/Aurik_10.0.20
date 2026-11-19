"""Aurik Error Handling — Degradierte Ausgabe und Recovery.

§15.8: Fehlertoleranz-Subsystem für 68-Phasen-Pipeline.
Graceful Degradation statt Pipeline-Abbruch bei nicht-kritischen Fehlern.
"""

from backend.core.errors.degraded_output import DegradedOutput
from backend.core.errors.phase_error_guard import phase_error_guard

__all__ = ["DegradedOutput", "phase_error_guard"]
