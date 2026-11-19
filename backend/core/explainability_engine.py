"""
backend/core/explainability_engine.py — Audio restoration explainability module
================================================================================

Provides natural-language explanations for restoration decisions.
"""

from __future__ import annotations


class ExplainabilityEngine:
    """Generiert German explanation strings for audio restoration phases."""

    def explain(
        self,
        phase: str,
        context: dict,
        metrics: dict,
    ) -> str:
        """Gibt a German explanation for a restoration phase zurück.

        Args:
            phase:   Name of the applied phase (e.g. "declicking", "denoising").
            context: Optional context dict (material, era, etc.).
            metrics: Metrics produced by the phase.

        Returns:
            German explanation string.
        """
        if phase == "declicking":
            cr = float(metrics.get("click_reduction", 0.0))
            if cr >= 0.5:
                return (
                    "Es wurden viele Störimpulse (Klicks) erkannt und entfernt "
                    f"(Reduktion: {cr:.0%}). Das Material wies eine hohe Klick-Dichte auf."
                )
            return f"Klick-Entfernung war kaum nötig (Reduktion: {cr:.0%}). Das Material ist in gutem Zustand."

        if phase == "denoising":
            nr = float(metrics.get("noise_reduction", 0.0))
            if nr >= 0.4:
                return (
                    "Es wurde ein hoher Rauschpegel festgestellt und reduziert "
                    f"(NR: {nr:.0%}). Das Signal-Rausch-Verhältnis wurde verbessert."
                )
            return f"Rauschreduktion war minimal erforderlich (NR: {nr:.0%}). Das Material ist weitgehend rauschfrei."

        if phase == "eq":
            dev = float(metrics.get("bark_band_deviation", 0.0))
            if dev >= 0.1:
                return (
                    "Es wurden spektrale Unausgewogenheiten im Frequenzgang "
                    f"festgestellt und korrigiert (Abweichung: {dev:.2f})."
                )
            return f"Frequenzgangkorrektur war kaum nötig (Abweichung: {dev:.2f}). Das Spektrum ist ausgewogen."

        return (
            f"Phase '{phase}' wurde ausgeführt. "
            "Alle restaurativen Schritte wurden gemäß dem aktuellen Analyseergebnis angewandt."
        )


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------
import threading as _threading

_explainability_engine_instance = None
_explainability_engine_lock = _threading.Lock()


def get_explainability_engine() -> ExplainabilityEngine:
    """Gibt the process-wide singleton ExplainabilityEngine instance zurück."""
    global _explainability_engine_instance
    if _explainability_engine_instance is None:
        with _explainability_engine_lock:
            if _explainability_engine_instance is None:
                _explainability_engine_instance = ExplainabilityEngine()
    return _explainability_engine_instance
