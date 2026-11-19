"""
Aurik 6.0 - Artefakt- und Bias-Detektion (SOTA-Produktivmodul)
Automatisierte Erkennung und Audit-Log-Dokumentation von Artefakten und Bias im Audiosignal.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def detect_clipping(audio, threshold=0.99):
    try:
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            raise ValueError("Ungültige Eingabe für detect_clipping")
        result = np.any(np.abs(audio) >= threshold)
        _audit_log({"clipping": bool(result), "threshold": threshold})
        return result
    except Exception as e:
        logger.error("[Clipping-Erkennung][Fehler] %s", e)
        _audit_log({"clipping": False, "error": str(e)})
        return False


def detect_dc_offset(audio, tolerance=0.01):
    try:
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            raise ValueError("Ungültige Eingabe für detect_dc_offset")
        result = abs(np.mean(audio)) > tolerance
        _audit_log({"dc_offset": bool(result), "tolerance": tolerance})
        return result
    except Exception as e:
        logger.error("[DC-Offset-Erkennung][Fehler] %s", e)
        _audit_log({"dc_offset": False, "error": str(e)})
        return False


def detect_bias(audio, sr):
    try:
        if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
            raise ValueError("Ungültige Eingabe für detect_bias")
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        total: float = float(np.sum(spec))
        for f_low, f_high in [(0, 200), (200, 2000), (2000, 6000), (6000, 12000)]:
            band = spec[(freqs >= f_low) & (freqs < f_high)]
            if np.sum(band) / (total + 1e-9) > 0.5:
                _audit_log({"bias": True, "band": (f_low, f_high)})
                return True, (f_low, f_high)
        _audit_log({"bias": False})
        return False, None
    except Exception as e:
        logger.error("[Bias-Erkennung][Fehler] %s", e)
        _audit_log({"bias": False, "error": str(e)})
        return False, None


def _audit_log(result):
    logger.info("[AuditLog][artifact_bias_detection] Ergebnis: %s", result)


# Integration in Pipeline: Nach jedem Verarbeitungsschritt aufrufen und ins Audit-Log schreiben!
