"""tests/unit/test_phase_07_harmonic_restoration.py — §SOTA-PSY-A1 (HarmonicRestorationPhase)."""

import numpy as np

from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase


def _make_harmonic_audio(sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    """Reiner Sinus ohne Obertöne + leichtes Rauschen (löst Harmonik-Synthese aus)."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.4 + rng.randn(int(sr * duration)) * 0.005).astype(np.float32)


def _make_phase(sr: int = 48000) -> HarmonicRestorationPhase:
    return HarmonicRestorationPhase(sample_rate=sr)


# ---------------------------------------------------------------------------
# §SOTA-PSY-A1: subaudibles Harmonik-Delta wird gegated (Dry-Passthrough)
# ---------------------------------------------------------------------------


def test_psy_a1_subaudible_delta_passthrough(monkeypatch):
    """Gate meldet skippable ⇒ Phase gibt Dry zurück (§SOTA-PSY-A1)."""
    import backend.core.dsp.audibility_gate as ag

    sr = 48000
    phase = _make_phase(sr)
    audio = _make_harmonic_audio(sr)

    monkeypatch.setattr(
        ag,
        "defect_audibility",
        lambda *a, **kw: {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True},
    )

    result = phase.process(audio.copy(), sample_rate=sr, material_type="shellac")
    assert result.metadata.get("subaudible_defects_skipped") is True
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=5)


def test_psy_a1_real_band_does_not_deactivate_phase():
    """Realer Gegentest: hörbare Harmonik-Synthese im richtigen Band (2–16 kHz)
    ⇒ Phase wird NICHT übersprungen (Output ≠ Dry). Beweist, dass die Bandwahl korrekt ist.
    """
    sr = 48000
    phase = _make_phase(sr)
    audio = _make_harmonic_audio(sr)
    result = phase.process(audio.copy(), sample_rate=sr, material_type="shellac")
    # Kein PSY-A1-Skip-Flag; die Phase hat echte Harmonik hinzugefügt (Output ≠ Dry).
    assert result.metadata.get("subaudible_defects_skipped") is not True
    assert result.modifications.get("harmonic_restored") is True
    assert not np.allclose(result.audio.astype(np.float32), audio.astype(np.float32), atol=1e-6)


def test_psy_a1_subaudible_delta_stereo_layout_restored(monkeypatch):
    """Stereo-Layout-Invariante: Gate-Frühreturn restauriert (C, N) statt (N, C).

    Fängt Mean-Achsen-/Layout-Kollaps-Bugklasse ab (Befund a8929178). Phase 07
    normalisiert intern auf channels-last (to_channels_last) und MUSS den
    Frühreturn mit restore_layout(..., _p07_transposed) zurückführen.
    """
    import backend.core.dsp.audibility_gate as ag

    sr = 48000
    phase = _make_phase(sr)
    mono = _make_harmonic_audio(sr)
    audio = np.vstack([mono, mono * 0.9]).astype(np.float32)  # channels-first (2, N)

    monkeypatch.setattr(
        ag,
        "defect_audibility",
        lambda *a, **kw: {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True},
    )

    result = phase.process(audio.copy(), sample_rate=sr, material_type="shellac")
    assert result.metadata.get("subaudible_defects_skipped") is True
    assert result.audio.shape == audio.shape, f"Layout-Kollaps: {result.audio.shape} statt {audio.shape}"
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=5)
