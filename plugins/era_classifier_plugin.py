"""
EraClassifier Plugin — Spec §2.1 / §2.14 Delegation-Stub
=========================================================

Dieses Plugin erfüllt den in Spec §2.1 vorgeschriebenen Pfad
``plugins/era_classifier_plugin.py`` und delegiert vollständig an
die kanonische Implementierung in ``backend/core/era_classifier.py``.

Keine eigene Logik — Single Source of Truth bleibt ``backend/core/era_classifier``.
"""

from backend.core.era_classifier import (
    EraClassifier,
    EraResult,
    classify_era,
    get_era_classifier,
)

__all__ = [
    "EraClassifier",
    "EraResult",
    "classify_era",
    "get_era_classifier",
]
