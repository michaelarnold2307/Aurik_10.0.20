from __future__ import annotations

"""Normative CI-Contract: preventive DSP guards must exist in critical timing phases.

Purpose:
- Enforce "Prävention vor Reparatur" for the historically most fragile timing paths.
- Prevent regressions where only downstream gates remain while phase-local protection
  disappears.

Scope (v2):
- phase_31_speed_pitch_correction: phase-local damage shield must be present and used
- phase_12_wow_flutter_fix: stereo temporal re-alignment + percentile peak cap must be present
- phase_42_vocal_enhancement: stem re-alignment + zero-phase additive bands
- phase_44_guitar_enhancement: zero-phase filters + percentile peak guard
- phase_45_brass_enhancement: zero-phase filters + percentile peak guard
- phase_46_spatial_enhancement: IACC guard + percentile peak guard
- phase_48_stereo_width_enhancer: IACC guard + percentile peak guard
"""


from pathlib import Path

import pytest

_PHASE31 = Path("backend/core/phases/phase_31_speed_pitch_correction.py")
_PHASE12 = Path("backend/core/phases/phase_12_wow_flutter_fix.py")
_PHASE42 = Path("backend/core/phases/phase_42_vocal_enhancement.py")
_PHASE44 = Path("backend/core/phases/phase_44_guitar_enhancement.py")
_PHASE45 = Path("backend/core/phases/phase_45_brass_enhancement.py")
_PHASE46 = Path("backend/core/phases/phase_46_spatial_enhancement.py")
_PHASE48 = Path("backend/core/phases/phase_48_stereo_width_enhancer.py")


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase31_has_preventive_damage_shield_contract() -> None:
    assert _PHASE31.exists(), f"Fehlt: {_PHASE31}"
    text = _PHASE31.read_text(encoding="utf-8")

    assert "def _apply_preventive_damage_shield(" in text, (
        "Phase 31 muss eine lokale Präventionsschicht besitzen (damage shield, nicht nur downstream gate)."
    )
    assert "result_audio, shield_meta = self._apply_preventive_damage_shield(" in text, (
        "Phase 31 muss den Damage-Shield im process()-Pfad aufrufen."
    )
    assert "compute_gated_rms_dbfs" in text, "Phase 31 Damage-Shield braucht RMS-Anstiegsbegrenzung."
    assert "correct_interchannel_delay(" in text, "Phase 31 Damage-Shield braucht L/R-Zeitkohärenz-Korrektur."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase12_has_preventive_timing_and_peak_contract() -> None:
    assert _PHASE12.exists(), f"Fehlt: {_PHASE12}"
    text = _PHASE12.read_text(encoding="utf-8")

    assert "_MAX_PERCENTILE_PEAK" in text, "Phase 12 braucht einen fixen p99.9-Peak-Ceiling-Contract."
    assert "correct_interchannel_delay(" in text, (
        "Phase 12 Loudness-Pfad muss vor Pegelentscheidung eine L/R-Re-Alignment-Stufe enthalten."
    )
    assert "np.percentile(np.abs(proc), 99.9)" in text, "Phase 12 muss p99.9-basierten Peak-Guard verwenden."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase42_has_preventive_alignment_and_zero_phase_contract() -> None:
    assert _PHASE42.exists(), f"Fehlt: {_PHASE42}"
    text = _PHASE42.read_text(encoding="utf-8")

    assert "align_stem_to_reference(" in text, "Phase 42 muss Stem-Latenz präventiv ausrichten, bevor Re-Mix erfolgt."
    assert "signal.sosfiltfilt(" in text, "Phase 42 additive Bandpfade müssen zero-phase filtern."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase44_has_zero_phase_and_peak_guard_contract() -> None:
    assert _PHASE44.exists(), f"Fehlt: {_PHASE44}"
    text = _PHASE44.read_text(encoding="utf-8")

    assert "sig.filtfilt(" in text, "Phase 44 muss zero-phase IIR-Filternutzung erzwingen."
    assert "sig.sosfiltfilt(" in text, "Phase 44 additive Bandpfade müssen sosfiltfilt nutzen."
    assert "np.percentile(np.abs(audio), 99.9)" in text, "Phase 44 braucht p99.9 input peak guard."
    assert "np.percentile(np.abs(x), 99.9)" in text, "Phase 44 braucht p99.9 output peak guard."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase45_has_zero_phase_and_peak_guard_contract() -> None:
    assert _PHASE45.exists(), f"Fehlt: {_PHASE45}"
    text = _PHASE45.read_text(encoding="utf-8")

    assert "sig.filtfilt(" in text, "Phase 45 muss zero-phase IIR-Filternutzung erzwingen."
    assert "sig.sosfiltfilt(" in text, "Phase 45 additive Bandpfade müssen sosfiltfilt nutzen."
    assert "np.percentile(np.abs(audio), 99.9)" in text, "Phase 45 braucht p99.9 input peak guard."
    assert "np.percentile(np.abs(x), 99.9)" in text, "Phase 45 braucht p99.9 output peak guard."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase46_has_iacc_and_peak_guard_contract() -> None:
    assert _PHASE46.exists(), f"Fehlt: {_PHASE46}"
    text = _PHASE46.read_text(encoding="utf-8")

    assert "_IACC_MIN" in text, "Phase 46 braucht einen festen IACC-Minimalwert als Präventionsgrenze."
    assert "_compute_iacc(" in text, "Phase 46 muss IACC aktiv messen."
    assert "side_reduction" in text, "Phase 46 muss Side-Anteil bei IACC-Verletzung reduzieren."
    assert "np.percentile(np.abs(audio), 99.9)" in text, "Phase 46 braucht p99.9 input peak guard."
    assert "np.percentile(np.abs(processed), 99.9)" in text, "Phase 46 braucht p99.9 output peak guard."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase48_has_iacc_and_peak_guard_contract() -> None:
    assert _PHASE48.exists(), f"Fehlt: {_PHASE48}"
    text = _PHASE48.read_text(encoding="utf-8")

    assert "_IACC_MIN" in text, "Phase 48 braucht einen festen IACC-Minimalwert als Präventionsgrenze."
    assert "_compute_iacc(" in text, "Phase 48 muss IACC aktiv messen."
    assert "side_reduction" in text, "Phase 48 muss Side-Anteil bei IACC-Verletzung reduzieren."
    assert "np.percentile(np.abs(audio), 99.9)" in text, "Phase 48 braucht p99.9 input peak guard."
    assert "np.percentile(np.abs(processed), 99.9)" in text, "Phase 48 braucht p99.9 output peak guard."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase31_stereo_simultaneous_processing_invariant_static() -> None:
    """§2.51: Phase 31 muss Stereo-Simultaneous-Processing als normative Klasseninvariante tragen.

    Pitch-Detektion IMMER auf Mono-Mix; Stretch-Parameter EINMAL berechnet,
    IDENTISCH auf L und R angewendet — Zeitversatz zwischen den Kanälen ist VERBOTEN.
    """
    assert _PHASE31.exists(), f"Fehlt: {_PHASE31}"
    text = _PHASE31.read_text(encoding="utf-8")

    assert "_STEREO_SIMULTANEOUS_PROCESSING: bool = True" in text, (
        "Phase 31 muss _STEREO_SIMULTANEOUS_PROCESSING = True als Klassenattribut tragen "
        "(normative Verriegelung §2.51 — kein Inter-Kanal-Zeitversatz)."
    )
    assert "assert self._STEREO_SIMULTANEOUS_PROCESSING" in text, (
        "Phase 31 muss _STEREO_SIMULTANEOUS_PROCESSING in den Stretch-Methoden aktiv prüfen."
    )
    # PSOLA: Shared-period helpers müssen vorhanden sein (Stereo via Mono-Mix)
    assert "_psola_compute_periods_mono(" in text, (
        "Phase 31 PSOLA muss _psola_compute_periods_mono() besitzen: Period-Berechnung auf Mono-Mix, niemals pro Kanal."
    )
    assert "_psola_apply_mono(" in text, (
        "Phase 31 PSOLA muss _psola_apply_mono() besitzen: "
        "gleicher period_samps-Array für L und R (§2.51 Simultaneous-Processing)."
    )
    # Pitch detection must use mono mix (never per-channel raw audio)
    assert "np.mean(audio, axis=1)" in text, (
        "Pitch-Detektion in Phase 31 muss stets np.mean(audio, axis=1) für den Mono-Mix nutzen — "
        "niemals per-Kanal (§2.51)."
    )


@pytest.mark.normative
@pytest.mark.timeout(30)
def test_phase31_wsola_identical_channels_yield_identical_output() -> None:
    """Behavioral §2.51: WSOLA auf L=R-Stereo-Input muss L=R-Output liefern.

    Verifiziert, dass _correct_wsola L und R mit identischen Parametern verarbeitet
    und kein Zeitversatz zwischen den Kanälen eingeführt wird.
    """
    import numpy as np

    from backend.core.phases.phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase

    phase = SpeedPitchCorrectionPhase()
    assert phase._STEREO_SIMULTANEOUS_PROCESSING is True, "_STEREO_SIMULTANEOUS_PROCESSING muss zur Laufzeit True sein."

    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
    mono = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])  # L == R — identischer Input

    result = phase._correct_wsola(stereo, ratio=1.02, params={"formant_preserve": 0.8})

    assert result.ndim == 2, "WSOLA muss Stereo-Array (2D) zurückgeben."
    assert result.shape[1] == 2, "WSOLA Output muss 2 Kanäle haben."
    np.testing.assert_allclose(
        result[:, 0],
        result[:, 1],
        atol=1e-6,
        err_msg=(
            "WSOLA §2.51: L und R müssen bei identischem Input identischen Output liefern — "
            "kein Inter-Kanal-Zeitversatz durch unabhängige Parameterberechnung."
        ),
    )
