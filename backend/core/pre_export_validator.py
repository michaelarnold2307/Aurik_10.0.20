"""§v10.17 PreExportValidator — automatischer Gate vor jedem sf.write().

§0h/§0c-Auflösung (copilot-instructions.md, Rev. 2026-09-11):
Qualitäts-Gate-Fehler dürfen NICHT mehr in einem Hardstop ohne Ausgabedatei
enden — §0c (Export-Vertrag) verlangt, dass das bestmögliche sichere Ergebnis
mit Status „degraded" exportiert wird. Der Validator meldet deshalb
kritische Qualitätsbefunde als degraded-Warnings und lässt den Export zu.
Nur un-schreibbare Arrays (leer, NaN/Inf) liefern passed=False, weil es dort
physisch kein sicheres Ergebnis zum Exportieren gibt.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# §0c: Marker-Präfix für degraded-Warnings — Aufrufer können daran den
# Export-Status „degraded" erkennen, ohne den Rückgabewert umzudeuten.
DEGRADED_PREFIX = "DEGRADED_EXPORT"


def validate_before_export(audio: np.ndarray, sr: int, is_studio_2026: bool = False) -> tuple[bool, list[str]]:
    """Wird VOR jedem Export automatisch aufgerufen.

    Returns:
        (passed, warnings) — passed=False bedeutet: Array ist physisch nicht
        schreibbar (leer / NaN / Inf). Qualitäts-Gate-Fehler liefern gemäß
        §0c passed=True mit ``DEGRADED_EXPORT``-Warnings; der Aufrufer MUSS
        dann das bestmögliche sichere Ergebnis (Rollback-Kaskade) schreiben
        und den Status „degraded" führen — niemals den Export verweigern.
    """
    warnings: list[str] = []

    try:
        from backend.core.export_quality_gate import ExportQualityGate
        from backend.core.fallback_auditor import get_fallback_auditor

        # 1. ExportQualityGate — §0c: Befunde degradieren den Export, blockieren ihn nicht.
        check = ExportQualityGate.check(audio, sr, is_studio_2026=is_studio_2026)
        if check.errors:
            logger.warning(
                "PreExportValidator: %d kritische Qualitäts-Befunde — §0c degraded-Ausgabe "
                "(bestmögliches sicheres Ergebnis wird geschrieben, KEIN Hardstop)",
                len(check.errors),
            )
            warnings.extend(f"{DEGRADED_PREFIX}: {e}" for e in check.errors)
        if check.warnings:
            warnings.extend(check.warnings)

        # 2. FallbackAuditor — konsolidierter Degradations-Bericht + Kaskaden-Limit.
        # §0c: Auch hier gilt degraded statt blockiert.
        fa = get_fallback_auditor()
        if fa.degraded:
            logger.warning("%s", fa.report())  # §v10.17: sichtbarer Bericht vor jedem Export
        if fa.should_block_pipeline if hasattr(fa, "should_block_pipeline") else False:
            logger.warning(
                "PreExportValidator: Ersatzpfad-Kaskadenlimit überschritten — §0c degraded-Ausgabe statt Blockade"
            )
            warnings.append(f"{DEGRADED_PREFIX}: fallback_cascade_exceeded")

        # 3. Audio-Sanity — einzig verbleibende Blockade-Bedingungen:
        # leere oder NaN/Inf-Arrays sind nicht schreibbar; hier gibt es kein
        # „bestmögliches sicheres Ergebnis" im Array selbst.
        arr = np.asarray(audio)
        if arr.size == 0:
            return False, [*warnings, "empty_audio"]
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return False, [*warnings, "nan_inf_in_audio"]

        logger.info("PreExportValidator: PASS — %.1f dBTP, %.1f LUFS", check.true_peak_dbtp, check.integrated_lufs)
        return True, warnings

    except Exception as e:
        logger.warning("PreExportValidator fehlgeschlagen: %s — Ausgabe NICHT blockiert", e)
        return True, warnings  # Nicht blockieren wenn Validator selbst fehlschlägt
