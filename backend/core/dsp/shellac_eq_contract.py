"""
Python-Interface für historische EQ-Kurven (DSP-Contract) für Schellack
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HistoricalEQCurveContract:
    id: str = "historical_eq_curve"
    category: str = "eq_curve"
    version: str = "1.0.0"
    curve: str = "flat"  # z.B. "columbia_pre_1938", "victor_pre_1938", ...
    turnover_hz: int = 400
    rolloff_db_at_10k: int = -12
    max_hf_boost_db: float = 0.0
    max_lf_boost_db: float = 0.0
    preconditions: dict[str, Any] | None = None
    side_effects: list | None = None
    rollback_strategy: str = "snapshot_restore"

    def is_applicable(self, medium: str) -> bool:
        """
        Prüft, ob die EQ-Kurve für das Medium anwendbar ist. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        """
        try:
            result = medium == "shellac"
            _audit_log("success", f"is_applicable: {result} für medium={medium}")
            return result
        except Exception as e:
            _audit_log("error", f"Fehler bei is_applicable: {e}")
            return False

    def commit_gate(self, anchor_stability_delta: float, hf_harshness_delta: float) -> bool:
        """
        Commit-Gate für EQ-Kurve. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        """
        try:
            result = anchor_stability_delta >= 0.020 and hf_harshness_delta <= 0.0
            _audit_log(
                "success",
                f"commit_gate: {result} (anchor_delta={anchor_stability_delta}, hf_delta={hf_harshness_delta})",
            )
            return result
        except Exception as e:
            _audit_log("error", f"Fehler bei commit_gate: {e}")
            return False


def _audit_log(level: str, message: str) -> None:
    _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
    _fn("[shellac_eq_contract] %s", message)


# Start-Kurven (praxisbewährt)
SHELLAC_EQ_CURVES = [
    {"curve": "columbia_pre_1938", "turnover_hz": 300, "rolloff_db_at_10k": -16},
    {"curve": "victor_pre_1938", "turnover_hz": 500, "rolloff_db_at_10k": -8},
    {"curve": "hmv_pre_1938", "turnover_hz": 250, "rolloff_db_at_10k": -12},
    {"curve": "default", "turnover_hz": 400, "rolloff_db_at_10k": -12},
]
