"""
Mono-Summierungslogik für Schellack (Kanalwahl/Summierung)
"""

import logging

logger = logging.getLogger(__name__)


def compute_disruption_score(
    click_density: float,
    crackle_density: float,
    hf_integrity: float,
    noise_floor: float,
    weights: tuple[float, float, float, float] = (0.5, 0.3, 0.2, 0.0),
) -> float:
    """
    Berechnet einen Disruption-Score für einen Kanal. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
    """
    try:
        for val in [click_density, crackle_density, hf_integrity, noise_floor]:
            if not isinstance(val, (float, int)):
                _audit_log("error", f"Input {val} ist kein float/int")
                raise ValueError("Alle Inputs müssen float/int sein")
        score = (
            weights[0] * click_density
            + weights[1] * noise_floor
            + weights[2] * hf_integrity
            + weights[3] * crackle_density
        )
        _audit_log("success", f"Disruption-Score berechnet: {score:.3f}")
        return score
    except Exception as e:
        _audit_log("error", f"Fehler bei Disruption-Score: {e}")
        return 0.0


def select_channel_or_sum(scores: dict[str, float], threshold: float = 0.20) -> str:
    """
    Entscheidet, ob L, R oder Summe verwendet wird. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
    """
    try:
        score_L = scores.get("L", 0.0)
        score_R = scores.get("R", 0.0)
        result = ("L" if score_L < score_R else "R") if abs(score_L - score_R) >= threshold else "mono_sum"
        _audit_log("success", f"Kanalwahl: {result} (L={score_L:.3f}, R={score_R:.3f})")
        return result
    except Exception as e:
        _audit_log("error", f"Fehler bei Kanalwahl: {e}")
        return "mono_sum"


def mono_sum(L, R):
    """
    Einfache Mono-Summierung. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
    """
    try:
        if not (isinstance(L, (float, int)) and isinstance(R, (float, int))):
            _audit_log("error", "Inputs sind nicht float/int")
            raise ValueError("Inputs müssen float/int sein")
        result = 0.5 * (L + R)
        _audit_log("success", f"Mono-Summe: {result:.3f}")
        return result
    except Exception as e:
        _audit_log("error", f"Fehler bei Mono-Summierung: {e}")
        return 0.0


# Gate: HF-Coherence nach Summierung prüfen


def hf_coherence_gate(hf_coherence_before: float, hf_coherence_after: float, tolerance: float = 0.02) -> bool:
    """
    Prüft, ob die HF-Coherence nach der Summierung im Toleranzbereich liegt. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
    """
    try:
        if not (isinstance(hf_coherence_before, (float, int)) and isinstance(hf_coherence_after, (float, int))):
            _audit_log("error", "Inputs sind nicht float/int")
            raise ValueError("Inputs müssen float/int sein")
        result = hf_coherence_after >= (hf_coherence_before - tolerance)
        _audit_log("success", f"HF-Coherence-Gate: {result}")
        return result
    except Exception as e:
        _audit_log("error", f"Fehler bei HF-Coherence-Gate: {e}")
        return False


def _audit_log(level: str, message: str) -> None:
    _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
    _fn("[shellac_mono_strategy] %s", message)
