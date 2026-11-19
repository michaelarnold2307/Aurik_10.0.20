"""
backend/ethics_engine.py — Deprecation-Shim
============================================

.. deprecated::
    Dieses Modul ist ein Kompatibilitäts-Shim.
    Die kanonische Implementierung liegt in:
        backend.core.epistemic_gate.ethics_engine

    Alle Importe bitte auf das kanonische Modul umstellen::

        from backend.core.epistemic_gate.ethics_engine import (
            EpistemicDecision, ProcessingMode, AuthenticityConstraints,
            EthicsReport, EthicsEngine, integrate_ethics_into_pipeline,
        )

Autor: AURIK Team (Shim seit v9.x)
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "backend.ethics_engine ist veraltet. Bitte 'from backend.core.epistemic_gate.ethics_engine import ...' verwenden.",
    DeprecationWarning,
    stacklevel=2,
)

from backend.core.epistemic_gate.ethics_engine import (
    AuthenticityConstraints,
    EpistemicDecision,
    EthicsEngine,
    EthicsReport,
    ProcessingMode,
    integrate_ethics_into_pipeline,
)

__all__ = [
    "AuthenticityConstraints",
    "EpistemicDecision",
    "EthicsEngine",
    "EthicsReport",
    "ProcessingMode",
    "integrate_ethics_into_pipeline",
]
