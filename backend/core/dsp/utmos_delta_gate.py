"""C1-Consumer: UTMOS-Delta-Veto im laufenden Restaurierungs-Regelkreis.

UTMOS ist ein ZEUGE, kein Richter (Hörordnung §8): Nur eine klare
MOS-Regression gegenüber der Referenz (delta < −threshold) reduziert die
verantwortliche Phase-Familie SOFORT über das zentrale Kalibrierungs-
profil. Ist das UTMOS-Modell nicht ladbar (CI ohne ML-Gewichte), wird
KEIN Veto ausgelöst — Metriken ohne Messung dürfen nie bestrafen.

Deterministisch: Die UTMOS-Inferenz selbst ist modell-deterministisch
(gleiche Gewichte + gleicher Input), hier wird nichts gesampelt.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_UTMOS_THRESHOLD = 0.05  # MOS-Delta, konsistent mit evaluation_system._UTMOS_IMPROVE


def utmos_delta_veto(
    profile: Any,
    reference: np.ndarray,
    candidate: np.ndarray,
    sr: int,
    stage: str = "utmos_gate",
    threshold: float = _UTMOS_THRESHOLD,
) -> Any | None:
    """Misst das UTMOS-MOS-Delta (Referenz → Kandidat) und vetot bei Regression.

    Rückgabe: WitnessVetoResult bei angewendetem Veto, sonst None
    (auch wenn UTMOS nicht verfügbar ist — kein Messwert, kein Veto).
    """
    try:
        from backend.core.evaluation_system import _compute_utmos_delta  # pylint: disable=import-outside-toplevel

        delta = _compute_utmos_delta(np.asarray(reference), np.asarray(candidate), sr)
    except Exception as _exc:  # pylint: disable=broad-except
        logger.debug("UTMOS-Delta nicht messbar (%s) — kein Veto", _exc)
        return None
    if delta is None:
        return None
    try:
        from backend.core.dsp.witness_correction_loop import (  # pylint: disable=import-outside-toplevel
            apply_external_metric_veto,
        )

        res = apply_external_metric_veto(
            profile,
            name="UTMOS",
            delta=float(delta),
            threshold=float(threshold),
            stage=stage,
            family="enhancement",
        )
        return res if res.applied else None
    except Exception as _exc:  # pylint: disable=broad-except
        logger.debug("UTMOS-Veto nicht anwendbar (%s) — kein Veto", _exc)
        return None
