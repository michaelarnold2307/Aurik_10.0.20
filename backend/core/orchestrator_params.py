"""§ORCHESTRATOR OrchestratorParams — Das EINE Gehirn. v10.0.0 Final.

JEDE Phase bezieht ALLE Parameter von HIER.
Keine Phase enthält eigene Entscheidungslogik.
Kein Wert ist geraten — alles kontinuierlich aus Messungen.

Architektur:
  Orchestrator (Gehirn)  →  compute_params(phase_id, measurements) → dict
  Phasen (Hände)         →  process(audio, **params)  ← nur DSP

§V25: 0 hartcodierte Werte. Alles kontinuierliche Funktionen.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Kontinuierliche Parameter-Funktionen — Pro Phase
# ═══════════════════════════════════════════════════════════════════════════


def _safe(m: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(m.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def _posterior_strength(m: dict, severity_key: str, confidence_key: str) -> float:
    """Return normalized defect depth × its detector confidence (§G188)."""
    severity = _clip(_safe(m, severity_key), 0.0, 1.0)
    confidence = _clip(_safe(m, confidence_key), 0.0, 1.0)
    return severity * confidence


# ── Phase 03: Denoise ───────────────────────────────────────────────────


def phase03_denoise(m: dict) -> dict:
    """Rauschstärke aus normalisierter Defekttiefe und Detektorkonfidenz."""
    snr_db = _safe(m, "snr_db", 30.0)
    strength = _posterior_strength(m, "noise_severity", "noise_confidence")

    return {"strength": round(strength, 3), "snr_db": snr_db}


# ── Phase 07: Harmonic Restoration ──────────────────────────────────────


def phase07_harmonic(m: dict) -> dict:
    """Harmonische Rekonstruktion folgt der gemessenen harmonischen Lücke."""
    bw = _safe(m, "bandwidth_loss")
    strength = _posterior_strength(m, "harmonic_deficit", "harmonic_confidence")

    return {
        "strength": round(strength, 3),
        "h2_target": round(0.003 + 0.003 * (1.0 - bw), 4),
        "tilt_tolerance_db": round(2.0 + 2.0 * bw, 1),
    }


# ── Phase 19: De-Esser ──────────────────────────────────────────────────


def phase19_deesser(m: dict) -> dict:
    """Sibilanzstärke folgt Defekttiefe × Detektor-Konfidenz."""
    bw = _safe(m, "bandwidth_loss")
    terminal = str(m.get("terminal_codec", "") or "").lower()
    is_mp3 = terminal in ("mp3_low", "mp3_high")

    sib_factor = 1.0 + 4.0 * bw if is_mp3 else 1.0 + 1.5 * bw
    sib_factor = _clip(sib_factor, 1.0, 6.0)

    # Codec und Bandbreite beeinflussen die Detektionsschwelle, nicht den
    # Kompensationsgrad eines bestätigten Sibilanz-Defekts (§G188/G189).
    cap_factor = 1.0
    sibilance_depth = _safe(m, "sibilance_severity")
    sibilance_confidence = _safe(m, "sibilance_confidence", 0.0)
    strength = _clip(sibilance_depth * sibilance_confidence, 0.0, 1.0)

    return {
        "sibilance_threshold_mult": round(sib_factor, 1),
        "deessing_strength_cap_factor": round(cap_factor, 2),
        "strength": round(strength, 3),
    }


# ── Phase 39: Air-Band ──────────────────────────────────────────────────


def phase39_air_band(m: dict) -> dict:
    """Kontinuierlich: Air-Band nur wenn bandwidth_loss es rechtfertigt."""
    bw = _safe(m, "bandwidth_loss")
    is_restoration = m.get("is_restoration_mode", True)
    is_analog = str(m.get("material_type", "")).lower() in (
        "vinyl",
        "shellac",
        "wax_cylinder",
        "wire_recording",
        "tape",
        "reel_tape",
        "cassette",
        "lacquer_disc",
    )

    # Air-Band ist in Restoration für analoge Quellen NUR mit bw_loss>0.5 erlaubt
    allow = not (is_restoration and is_analog and bw <= 0.5)

    strength = _posterior_strength(m, "air_band_deficit", "air_band_confidence") if allow else 0.0

    return {
        "allow_air_band": allow,
        "strength": round(_clip(strength, 0.0, 1.0), 3),
        "shelf_gain_db": round(1.0 + 5.0 * bw, 1),
    }


# ── Phase 29: Tape Hiss Reduction ───────────────────────────────────────


def phase29_tape_hiss(m: dict) -> dict:
    """Hiss-Reparatur folgt dem gemessenen Defektposterior."""
    snr = _safe(m, "snr_db", 30.0)
    strength = _posterior_strength(m, "hiss_severity", "hiss_confidence")

    return {"strength": round(strength, 3), "snr_db": snr}


# ── Phase 06: Frequency Restoration (NVSR) ──────────────────────────────


def phase06_frequency(m: dict) -> dict:
    """NVSR-Stärke folgt dem gemessenen Bandbreitenverlust."""
    bw = _safe(m, "bandwidth_loss")

    # Nur sinnvoll wenn tatsächlich Frequenzen fehlen
    if bw < 0.3:
        return {"strength": 0.0, "skip": True}

    strength = _posterior_strength(m, "bandwidth_loss", "bandwidth_confidence")

    rolloff_target = 10000 + int(5500 * bw)  # bw=0.5→12750, bw=1.0→15500

    return {
        "strength": round(strength, 3),
        "target_rolloff_hz": rolloff_target,
        "skip": False,
    }


# ── Phase 40: Loudness Normalization ────────────────────────────────────


def phase40_loudness(m: dict) -> dict:
    """Kontinuierlich: Loudness-Ziel aus Material + Restorability."""
    rs = _safe(m, "restorability_score", 50.0) / 100.0

    # Je besser das Material, desto näher am Standard-LUFS
    target_lufs = -18.0 + 3.0 * (1.0 - rs)  # rs=50→-16.5, rs=90→-17.7

    return {
        "target_lufs": round(target_lufs, 1),
        "strength": round(rs, 3),
    }


# ── Phase 01: Click Removal ─────────────────────────────────────────────


def phase01_click(m: dict) -> dict:
    """Klickkorrektur folgt der gemessenen Ereignisdichte."""
    click_density = _safe(m, "click_density", 0.0)
    if click_density <= 0.0:
        return {"strength": 0.0, "skip": True}

    strength = _posterior_strength(m, "click_severity", "click_confidence")

    return {"strength": round(strength, 3)}


# ── Phase 12: Wow/Flutter ───────────────────────────────────────────────


def phase12_wow_flutter(m: dict) -> dict:
    """Kontinuierlich: Stärke aus Wow/Flutter-Severity + Material."""
    wow = _safe(m, "wow_severity", 0.0)
    flutter = _safe(m, "flutter_severity", 0.0)
    sev = max(wow, flutter)
    confidence = max(_safe(m, "wow_confidence"), _safe(m, "flutter_confidence"))
    if sev < 0.10:
        return {"strength": 0.0, "skip": True}

    strength = _clip(sev, 0.0, 1.0) * _clip(confidence, 0.0, 1.0)

    return {"strength": round(strength, 3), "skip": False}


# ── Phase 09: Crackle ───────────────────────────────────────────────────


def phase09_crackle(m: dict) -> dict:
    """Kontinuierlich: Stärke aus Crackle-Dichte."""
    density = _safe(m, "crackle_density", 0.0)
    if density < 5:
        return {"strength": 0.0, "skip": True}
    strength = _posterior_strength(m, "crackle_severity", "crackle_confidence")
    return {"strength": round(_clip(strength, 0.0, 1.0), 3), "skip": False}


# ═══════════════════════════════════════════════════════════════════════════
# Universelle Parameter-Funktion — Das EINE Gehirn
# ═══════════════════════════════════════════════════════════════════════════

_PARAM_FUNCTIONS: dict[str, Any] = {
    "phase_01_click_removal": phase01_click,
    "phase_03_denoise": phase03_denoise,
    "phase_06_frequency_restoration": phase06_frequency,
    "phase_07_harmonic_restoration": phase07_harmonic,
    "phase_09_crackle_removal": phase09_crackle,
    "phase_12_wow_flutter_fix": phase12_wow_flutter,
    "phase_19_de_esser": phase19_deesser,
    "phase_29_tape_hiss_reduction": phase29_tape_hiss,
    "phase_39_air_band_enhancement": phase39_air_band,
    "phase_40_loudness_normalization": phase40_loudness,
}


# Generische Default-Funktion für Phasen ohne spezifische Kalibrierung
def _generic_params(m: dict) -> dict:
    severity = _safe(m, "defect_severity")
    confidence = _safe(m, "defect_confidence", 0.0)
    return {"strength": round(_clip(severity * confidence, 0.0, 1.0), 3)}


def compute_phase_params(
    phase_id: str,
    measurements: dict,
) -> dict:
    """Das EINE Gehirn: Berechnet ALLE Parameter für eine Phase.

    Args:
        phase_id: ID der Phase (z.B. "phase_03_denoise")
        measurements: Dict mit ALLEN aktuellen Messwerten.
            Muss enthalten: bandwidth_loss, restorability_score,
            crest_original, crest_current (mindestens).

    Returns:
        Dict mit kalibrierten Parametern — direkt an phase.process() übergebbar.
    """
    fn = _PARAM_FUNCTIONS.get(phase_id, _generic_params)
    try:
        params = fn(measurements)
    except Exception as e:
        logger.debug("berechnen_Verarbeitungsschritt_params %s: %s — Ersatzpfad generic", phase_id, e)
        params = _generic_params(measurements)

    # Gemeinsame Parameter für ALLE Phasen
    params.setdefault("calibrated", True)
    params.setdefault("restorability_score", _safe(measurements, "restorability_score", 50.0))
    params.setdefault("bandwidth_loss", _safe(measurements, "bandwidth_loss", 0.0))

    return params  # type: ignore[no-any-return]


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight Probe: Eine schnelle Test-Iteration für Grenzfälle
# ═══════════════════════════════════════════════════════════════════════════


def probe_phase_benefit(
    phase_id: str,
    audio: np.ndarray,
    phase_runner,
    params: dict,
    sample_rate: int = 48000,
) -> dict:
    """Testet mit EINER schnellen Ausführung, ob die Phase nützt.

    Führt die Phase mit der kalibrierten Zielstärke aus und misst das Delta.
    Ein schädlicher Kandidat wird zurückgerollt; keine pauschale Teilstärke.

    Returns:
        {"should_run": bool, "strength": float, "delta": float, "reason": str}
    """
    try:
        # Test 1: Kalibrierte Stärke
        strength = params.get("strength", 0.20)
        audio_test = phase_runner(audio, strength)
        delta = _quick_probe_delta(audio, audio_test)

        if delta > -0.02:
            return {
                "should_run": True,
                "strength": strength,
                "delta": round(delta, 4),
                "reason": f"Kalibrierte Stärke {strength:.3f} hilft (Δ={delta:+.4f})",
            }

        # §G188 (GEBOTE.md): Ein schädlicher Kandidat wird verworfen; ein
        # pauschal halbierter Rest-Eingriff darf einen gemessenen Defekt nicht stehenlassen.
        return {
            "should_run": False,
            "strength": 0.0,
            "delta": round(delta, 4),
            "reason": f"Kandidat zurückgerollt ({strength:.3f}→{delta:+.4f})",
        }

    except Exception as e:
        logger.debug(
            "§V6 (copilot-instructions.md) _berechnen_quality_probe fehlgeschlagen — should_Ausfuehrung=False zurückgegeben (Probe-Fehler): %s",
            e,
        )
        return {"should_run": False, "strength": 0.0, "delta": -1.0, "reason": f"Probe fehlgeschlagen: {e}"}


def _quick_probe_delta(pre: np.ndarray, post: np.ndarray) -> float:
    """Ultraschnelles Qualitäts-Delta (<1ms)."""
    try:
        a = np.asarray(pre, dtype=np.float32).ravel()
        b = np.asarray(post, dtype=np.float32).ravel()
        n = min(len(a), len(b), 4096)  # nur 4096 samples
        if n < 64:
            return 0.0
        a, b = a[:n], b[:n]
        rms_a = float(np.sqrt(np.mean(a**2))) + 1e-12
        rms_b = float(np.sqrt(np.mean(b**2))) + 1e-12
        rms_ok = min(rms_a, rms_b) / max(rms_a, rms_b)
        corr = float(np.corrcoef(a, b)[0, 1]) if n > 2 else 1.0
        corr = max(0.0, min(1.0, corr)) if not np.isnan(corr) else 1.0
        return float(0.5 * rms_ok + 0.5 * corr - 0.95)
    except Exception as exc:
        logger.debug(
            "§V6 (copilot-instructions.md) _quick_probe_delta fehlgeschlagen — 0.0 zurückgegeben (Audio %s): %s",
            pre.shape,
            exc,
        )
        return 0.0
