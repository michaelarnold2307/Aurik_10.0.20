"""C1-Consumer: UTMOS-Delta als ZEUGE im laufenden Restaurierungs-Regelkreis.

UTMOSv2 ist auf **Sprache** (BVCC) trainiert. Nach §III.11
(copilot-instructions.md) darf ein sprachtrainiertes Orakel bei Musik/Gesang
kein Veto-Recht ausüben — die Domäne der Bewertung ist nicht die Domäne des
Materials. Das Delta wird deshalb IMMER gemessen und protokolliert (Zeuge,
Hörordnung §8); ein Veto (Reduktion der Phase-Familie) greift nur, wenn ein
musik-kalibriertes MOS freigeschaltet ist (`music_model_flags.use_utmos_music`,
F8/TODO-SOTA). Ist das Modell nicht ladbar (CI ohne ML-Gewichte), wird KEIN
Veto ausgelöst — Metriken ohne Messung dürfen nie bestrafen.

Deterministisch: Die UTMOS-Inferenz selbst ist modell-deterministisch
(gleiche Gewichte + gleicher Input), hier wird nichts gesampelt.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_UTMOS_THRESHOLD = 0.05  # MOS-Delta, konsistent mit evaluation_system._UTMOS_IMPROVE


def _domain_veto_allowed() -> bool:
    """Veto-Recht nur mit musik-kalibriertem MOS (§III.11 copilot-instructions.md).

    Flag wird zur Laufzeit am Modul gelesen (nicht via `from … import`), damit
    Tests und Konfiguration es wirksam umschalten können.
    """
    try:
        import backend.core.music_model_flags as _mmf  # pylint: disable=import-outside-toplevel

        return bool(getattr(_mmf, "use_utmos_music", False))
    except Exception as _exc:  # pylint: disable=broad-except
        logger.warning("MOS-Domänen-Flag nicht lesbar (%s) — kein Veto (§V6 copilot-instructions.md)", _exc)
        return False


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
    if float(delta) < -float(threshold) and not _domain_veto_allowed():
        logger.info(
            "UTMOS-Veto unterdrückt: Sprach-MOS (BVCC) ohne Musik-Kalibrierung darf bei "
            "Musik/Gesang nicht richten (delta=%.3f, stage=%s, §III.11 copilot-instructions.md; F8 offen)",
            float(delta),
            stage,
        )
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
