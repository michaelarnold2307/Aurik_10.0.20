"""
backend.core.german_schlager_classifier — Alias-Modul (Spec §2.1 Pipeline).

Kanonischer Importpfad laut Pipeline-Spec:
    from backend.core.german_schlager_classifier import GermanSchlagerClassifier

Implementierung liegt in: backend/core/genre_classifier.py
"""

from backend.core.genre_classifier import (
    GermanSchlagerClassifier,
    SchlagerClassificationResult,
    get_genre_classifier,
    get_restoration_profile,
)

# Convenience-Alias: get_german_schlager_classifier → get_genre_classifier
get_german_schlager_classifier = get_genre_classifier

__all__ = [
    "GermanSchlagerClassifier",
    "SchlagerClassificationResult",
    "get_genre_classifier",
    "get_german_schlager_classifier",
    "get_restoration_profile",
]
