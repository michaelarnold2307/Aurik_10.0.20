"""Objective MUSHRA Estimator — Forward-Compatible Wrapper für mushra_evaluator.

⚠️  Dieses Modul ist die empfohlene Import-Quelle für MUSHRA-Approximationen.
Es re-exportiert mushra_evaluator mit explizitem _is_approximation=True auf
allen Result-Objekten. Kein Score aus diesem Modul darf als echter subjektiver
MUSHRA-Hörertest ausgegeben werden.

Für echte subjektive MUSHRA-Hörtests nach ITU-R BS.1534-3:
    backend/core/mushra_listener.py (geplant, §15.3)

Usage:
    from backend.core.objective_mushra_estimator import (
        get_mushra_evaluator,
        evaluate_mushra,
        compare_mushra,
        MushraResult,
        MushraComparison,
    )

    # Alle Result-Objekte haben _is_approximation=True
    result = evaluate_mushra(reference, restored, sr=48000)
    assert result._is_approximation  # Garantiert True
"""

from backend.core.mushra_evaluator import (
    MushraComparison,
    MushraEvaluator,
    MushraResult,
    compare_mushra,
    evaluate_mushra,
    get_mushra_evaluator,
)

__all__ = [
    "evaluate_mushra",
    "compare_mushra",
    "get_mushra_evaluator",
    "MushraComparison",
    "MushraEvaluator",
    "MushraResult",
]
