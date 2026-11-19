"""
DEPRECATION SHIM – backend.core.evaluation.quality_control
=========================================================
Diese Datei war ein älteres Subset von backend.quality_control
(635 Zeilen vs. 854 Zeilen — gleiche Klassen, gleiche Methoden,
aber ohne ML-Plugin-Integration und use_ml_plugins-Parameter).

Alle Symbole werden jetzt aus dem kanonischen Modul geliefert.

Kanonisch: backend/quality_control.py
"""

from __future__ import annotations

import warnings

warnings.warn(
    "backend.core.evaluation.quality_control ist veraltet. Importiere direkt aus backend.quality_control.",
    DeprecationWarning,
    stacklevel=2,
)

from backend.quality_control import (
    CASScoreCalculator,
    QualityControl,
    QualityGates,
)

__all__ = ["CASScoreCalculator", "QualityControl", "QualityGates"]
