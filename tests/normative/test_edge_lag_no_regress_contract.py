from __future__ import annotations

"""[RELEASE_MUST] §2.63 Edge/Lag No-Regress Contract.

Sichert die strukturellen Invarianten gegen Re-Introduktion:
- Intro/Outro-Peaks werden präventiv vermieden (Context-Padding vor ML).
- Kein neuer L/R-Lag durch kanalweise Drift im Phase-23-ML-Pfad.
"""


from pathlib import Path

import pytest

_SPEC02 = Path(".github/specs/02_pipeline_architecture.md")
_PHASE03 = Path("backend/core/phases/phase_03_denoise.py")
_PHASE23 = Path("backend/core/phases/phase_23_spectral_repair.py")


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_spec02_contains_edge_lag_release_must_section() -> None:
    assert _SPEC02.exists(), f"Fehlt: {_SPEC02}"
    text = _SPEC02.read_text(encoding="utf-8")

    assert "§2.63 [RELEASE_MUST] Intro/Outro-Edge-Safety + Stereo-Lag-Invariante" in text, (
        "Spec 02 muss den §2.63-Abschnitt als Release-Must enthalten."
    )
    assert "Kontext-Padding" in text and "deterministisch" in text, (
        "§2.63 muss Kontext-Padding + deterministisches Strippen normativ erzwingen."
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase03_uses_preventive_context_padding() -> None:
    assert _PHASE03.exists(), f"Fehlt: {_PHASE03}"
    text = _PHASE03.read_text(encoding="utf-8")

    assert "Context-Padding" in text, "Phase 03 muss präventives Context-Padding enthalten."
    assert "boundary artefacts" in text, "Phase 03 muss Boundary-Artefakte explizit adressieren."


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase23_uses_ms_path_and_avoids_lr_independent_ml_repair() -> None:
    assert _PHASE23.exists(), f"Fehlt: {_PHASE23}"
    text = _PHASE23.read_text(encoding="utf-8")

    assert "M/S domain" in text or "M/S" in text, "Phase 23 muss den M/S-Stereo-Pfad dokumentieren."
    assert "do NOT repair L/R independently" in text, "Phase 23 muss unabhängige L/R-ML-Reparatur explizit verbieten."

    # Hard anti-regression checks for the old independent-L/R call pattern.
    assert "_repair_single_channel(audio_arr[ch])" not in text, (
        "Regression erkannt: unabhängiger L/R-ML-Pfad (channels-first) wurde wieder eingeführt."
    )
    assert "_repair_single_channel(audio_arr[:, ch])" not in text, (
        "Regression erkannt: unabhängiger L/R-ML-Pfad (samples-first) wurde wieder eingeführt."
    )


@pytest.mark.normative
@pytest.mark.timeout(20)
def test_phase23_bw_ceiling_skips_harmonic_ceiling_violation_check() -> None:
    """§6.2c + §2.46e False-Positive-Bug v10.0.0:
    Wenn _apply_material_bw_ceiling applied=True zurückgibt, darf
    check_hallucination KEIN material_bw_ceiling_hz erhalten — sonst
    triggert der 6th-Order-Butterworth-Rolloff ein false-positive
    ceiling_band_ratio > 8× → vollständiger phase_23-Rollback für
    Kassetten-Material (AudioSR-Arbeit vollständig verworfen).
    """
    assert _PHASE23.exists(), f"Fehlt: {_PHASE23}"
    text = _PHASE23.read_text(encoding="utf-8")

    # Der Fix muss den _hg_ceiling_hz23-Guard enthalten (None wenn applied)
    assert "_hg_ceiling_hz23" in text, (
        "§6.2c/§2.46e Fix fehlt: _hg_ceiling_hz23-Variable muss in phase_23 existieren. "
        "Ohne diesen Guard triggert der 6th-Order-Butterworth-Rolloff false-positive "
        "harmonic_ceiling_violation für Kassetten-Material (cassette BW-Ceiling 12 kHz)."
    )
    assert "None if _bw_ceiling_applied23 else" in text, (
        "§6.2c/§2.46e Fix fehlt: material_bw_ceiling_hz muss None sein wenn "
        "_bw_ceiling_applied23=True — sonst false-positive Rollback aller phase_23-Arbeit."
    )
    # Blend-Pfad muss ebenfalls geschützt sein
    assert "_cand_hg_ceiling_hz23" in text, (
        "§6.2c/§2.46e Fix unvollständig: _cand_hg_ceiling_hz23 fehlt im Blend-Retry-Pfad. "
        "Alle 3 Blend-Versuche müssten ebenfalls None übergeben wenn Ceiling angewendet."
    )
    assert "None if _blend_ceiling_applied23 else" in text, (
        "§6.2c/§2.46e Fix unvollständig: Blend-Retry übergibt _bw23 statt None nach "
        "_apply_material_bw_ceiling → gleicher false-positive wie Hauptpfad."
    )
