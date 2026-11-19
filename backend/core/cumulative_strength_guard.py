"""
backend/core/cumulative_strength_guard.py — CumulativeStrengthGuard (Muster G)
================================================================================

Verhindert dass mehrere Guards (AFG, PMGG, IAD, UQ-Drive, Vintage-Caps) eine Phase
kumulativ auf < 0.60 effektive Stärke dämpfen.

Jeder Guard für sich ist sinnvoll — aber ihr Produkt kann eine Phase faktisch
deaktivieren (z. B. 0.75 × 0.69 × 0.96 ≈ 0.50). Der CumulativeStrengthGuard
erkennt solche Fälle und begrenzt die Gesamtdämpfung auf maximal 40 %
(= minimale effektive Stärke 0.60).

Verwendung:
    guard = CumulativeStrengthGuard()
    guard.register("AFG", 0.75)       # AFG backoff → 75%
    guard.register("UQ-Drive", 0.82)  # UQ scalar → 82%
    cumulative = guard.cumulative      # = 0.75 * 0.82 = 0.615
    ok, clamped = guard.clamp(0.60)
    # ok=True, clamped=False (615 >= 600)

    # Bei Über-Dämpfung:
    guard.register("PMGG", 0.50)      # PMGG strength → 50%
    cumulative = guard.cumulative      # = 0.75 * 0.82 * 0.50 = 0.307
    ok, clamped = guard.clamp(0.60)
    # ok=False, clamped=0.60 → muss von 0.307 auf 0.60 angehoben werden
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

CUMULATIVE_MINIMUM: float = 0.60
"""Untere Grenze für die effektive Gesamtstärke einer Phase.
Kein einzelner Guard-Eintrag darf das Produkt aller Guards unter diesen Wert drücken."""


@dataclass
class StrengthReport:
    """Bericht über die kumulative Stärke einer Phase."""

    phase_id: str
    """Name der Phase (z. B. 'phase_23_spectral_repair')."""

    reductions: dict[str, float] = field(default_factory=dict)
    """guard_name → reduction_factor ∈ (0, 1].
    Einträge in der Reihenfolge ihrer Registrierung."""

    cumulative: float = 1.0
    """Produkt aller reduction_factor."""

    clamped: float | None = None
    """Auf CUMULATIVE_MINIMUM angehobener Wert (wenn cumulative < Minimum)."""

    is_ok: bool = True
    """True wenn cumulative >= CUMULATIVE_MINIMUM (kein Eingriff nötig)."""

    @property
    def total_reduction_pct(self) -> float:
        """Gesamtdämpfung in Prozent (0 = keine Dämpfung, 40 = 40% Dämpfung)."""
        return float((1.0 - self.cumulative) * 100)


class CumulativeStrengthTracker:
    """Verfolgt die kumulative Dämpfung einer Phase durch mehrere Guards.

    Pro restore()-Aufruf wird eine Instanz erstellt. Jeder Guard registriert
    seinen Reduktionsfaktor. Am Ende kann die kumulative Stärke abgefragt
    und ggf. auf den Mindestwert angehoben werden.
    """

    def __init__(self, phase_id: str = "unknown") -> None:
        self.phase_id: str = phase_id
        self.reductions: dict[str, float] = {}
        self._clamped: float | None = None

    def register(self, guard_name: str, reduction_factor: float) -> None:
        """Registriert eine Guard-Entscheidung.

        Args:
            guard_name: Name des Guards (z. B. 'AFG', 'PMGG', 'IAD', 'UQ-Drive').
            reduction_factor: Reduktionsfaktor ∈ (0, 1].
                              1.0 = keine Reduktion.
        """
        rf = float(np.clip(reduction_factor, 0.01, 1.0))
        self.reductions[guard_name] = rf

    @property
    def cumulative(self) -> float:
        """Produkt aller registrierten Reduktionsfaktoren."""
        if not self.reductions:
            return 1.0
        prod = 1.0
        for rf in self.reductions.values():
            prod *= rf
        return float(prod)

    def clamp(self, minimum: float = CUMULATIVE_MINIMUM) -> tuple[bool, float]:
        """Prüft und korrigiert die kumulative Stärke.

        Returns:
            (is_ok, clamped_value)
            is_ok: True wenn cumulative >= minimum (kein Eingriff nötig).
            clamped_value: minimum falls gecapped, sonst cumulative.
        """
        cum = self.cumulative
        if cum >= minimum:
            self._clamped = None
            return True, cum
        self._clamped = minimum
        logger.warning(
            "§G CumulativeStrengthGuard: %s cumulative=%.3f < %.2f — "
            "muss um Faktor %.2f angehoben werden. "
            "Reduktionen: %s",
            self.phase_id,
            cum,
            minimum,
            minimum / max(cum, 1e-6),
            {k: f"{v:.3f}" for k, v in sorted(self.reductions.items())},
        )
        return False, minimum

    def report(self) -> StrengthReport:
        """Erzeugt einen StrengthReport für Logging/Telemetrie."""
        is_ok, clamped_val = self.clamp()
        return StrengthReport(
            phase_id=self.phase_id,
            reductions=dict(self.reductions),
            cumulative=self.cumulative,
            clamped=clamped_val if not is_ok else None,
            is_ok=is_ok,
        )

    def strongest_reduction_guard(self) -> tuple[str, float] | None:
        """Gibt (guard_name, factor) des am stärksten reduzierenden Guards zurück."""
        if not self.reductions:
            return None
        return min(self.reductions.items(), key=lambda kv: kv[1])


class GlobalCumulativeGuard:
    """Fasse mehrere CumulativeStrengthTracker pro restore()-Aufruf zusammen."""

    def __init__(self) -> None:
        self._trackers: dict[str, CumulativeStrengthTracker] = {}

    def get_tracker(self, phase_id: str) -> CumulativeStrengthTracker:
        """Holt oder erstellt einen Tracker für eine Phase."""
        if phase_id not in self._trackers:
            self._trackers[phase_id] = CumulativeStrengthTracker(phase_id=phase_id)
        return self._trackers[phase_id]

    def register(self, phase_id: str, guard_name: str, reduction: float) -> None:
        """Komfort-Methode: registriert eine Reduktion für eine Phase."""
        self.get_tracker(phase_id).register(guard_name, reduction)

    def all_reports(self) -> dict[str, StrengthReport]:
        """Reports aller getrackten Phasen."""
        return {pid: tracker.report() for pid, tracker in self._trackers.items()}

    def phases_below_threshold(self, minimum: float = CUMULATIVE_MINIMUM) -> list[tuple[str, float]]:
        """Listet Phasen auf, deren kumulative Stärke unter dem Minimum liegt.

        Returns:
            Liste von (phase_id, cumulative_strength) sortiert nach Stärke (niedrigste zuerst).
        """
        result: list[tuple[str, float]] = []
        for pid, tracker in self._trackers.items():
            cum = tracker.cumulative
            if cum < minimum:
                result.append((pid, cum))
        result.sort(key=lambda x: x[1])
        return result

    def log_summary(self) -> None:
        """Protokolliert eine Zusammenfassung aller Phasen mit Dämpfung."""
        weak = self.phases_below_threshold()
        if not weak:
            return
        logger.warning(
            "§G CumulativeStrengthGuard: %d/%d Phasen unter %.2f:\n%s",
            len(weak),
            len(self._trackers),
            CUMULATIVE_MINIMUM,
            "\n".join(
                f"  {pid}: cum={cum:.3f} (Dämpfung %{1 - cum:.1%}) "
                "— stärkster Guard: "
                f"{self._trackers[pid].strongest_reduction_guard()}"
                for pid, cum in weak
            ),
        )
