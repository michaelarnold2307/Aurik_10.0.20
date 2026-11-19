"""
backend/carrier_forensics.py — Kompatibilitäts-Shim (Aurik 6.0 → 9.x)
=============================================

Dieses Modul ist ein reiner Re-Export-Shim für
``backend.core.medium_classifier``.

Migrationsanleitung::

    # Alt (Aurik 6.0):
    from backend.carrier_forensics import CarrierForensics
    # Neu (Aurik 10.0.0.x):
    from backend.core.medium_classifier import MediumClassifier, classify_medium

Referenz: §2.1 Aurik-9-Spec, MediumClassifier (§6.1 MaterialType)
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "backend.carrier_forensics ist veraltet (Aurik 6.0). "
    "Verwende 'from backend.core.medium_classifier import MediumClassifier, classify_medium'.",
    DeprecationWarning,
    stacklevel=2,
)

from backend.core.medium_classifier import (
    ClassificationResult,
    MediumClassifier,
    classify_medium,
    get_medium_classifier,
)

# Aurik-6.0-kompatibler Alias
CarrierForensics = MediumClassifier


def analyze_carrier_forensics(mono, sr: int) -> dict:
    """Legacy compatibility shim (Aurik 6.0 → 9.x).

    Delegates to ``classify_medium`` and returns a dict compatible with the
    Aurik 6.0 ``analyze_carrier_forensics`` signature.

    Returns:
        dict with keys ``carrier_forensic`` (str), ``score`` (float),
        ``features`` (dict).
    """
    import numpy as _np

    result = classify_medium(_np.asarray(mono), sr)
    return {
        "carrier_forensic": str(
            result.material_type.value if hasattr(result.material_type, "value") else result.material_type
        ),
        "score": float(result.confidence),
        "features": {},
    }


__all__ = [
    "CarrierForensics",
    "ClassificationResult",
    "MediumClassifier",
    "analyze_carrier_forensics",
    "classify_medium",
    "get_medium_classifier",
]

# --- Aurik-6.0-Original-Code entfernt (2026-03-11, §9.4 Anti-Parallelwelten) ---
# Originaldatei war: carrier_forensics.py für Aurik 6.0
# Nachfolger: backend.core.medium_classifier (MediumClassifier)
