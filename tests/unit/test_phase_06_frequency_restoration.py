"""tests/unit/test_phase_06_frequency_restoration.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase


@pytest.fixture
def phase():
    return FrequencyRestorationPhase(sample_rate=48000)


@pytest.fixture
def audio():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5 + rng.randn(48000) * 0.01).astype(np.float32)


def test_returns_ndarray(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert isinstance(result.audio, np.ndarray)


def test_no_nan_inf(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert np.isfinite(result.audio).all()


def test_not_silent(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert float(np.sqrt(np.mean(result.audio**2))) > 1e-10


def test_length_preserved(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert len(result.audio) == len(audio)


# ---------------------------------------------------------------------------
# §SOTA-PSY-A1: subaudibles HF-Delta wird gegated (Dry-Passthrough)
# ---------------------------------------------------------------------------


def _make_rolled_off_audio(sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    """Breitbandiges Musik-Signal mit künstlichem Rolloff unterhalb 8 kHz.

    Erzeugt echten HF-Verlust, damit phase_06 tatsächlich HF synthetisiert
    (echter Gegentest gegen falsche Bandwahl).
    """
    import scipy.signal as _sps

    rng = np.random.RandomState(42)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float64)
    x = np.zeros(int(sr * duration), dtype=np.float64)
    for f in (220.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0, 14000.0):
        x += 0.15 * np.sin(2 * np.pi * f * t)
    x += rng.randn(int(sr * duration)) * 0.01
    sos = _sps.butter(8, 9000.0 / (sr / 2.0), btype="low", output="sos")
    return _sps.sosfiltfilt(sos, x).astype(np.float32)


def test_psy_a1_subaudible_delta_passthrough(monkeypatch):
    """Gate meldet skippable ⇒ Phase gibt Dry zurück (§SOTA-PSY-A1)."""
    import backend.core.dsp.audibility_gate as ag

    sr = 48000
    phase = FrequencyRestorationPhase(sample_rate=sr)
    audio = _make_rolled_off_audio(sr)

    monkeypatch.setattr(
        ag,
        "defect_audibility",
        lambda *a, **kw: {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True},
    )

    result = phase.process(audio.copy(), sample_rate=sr, material_type="unknown")
    assert result.metadata.get("subaudible_defects_skipped") is True
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=5)


def test_psy_a1_real_band_does_not_deactivate_phase():
    """Realer Gegentest: hörbares HF-Delta im richtigen Band (8–20 kHz) ⇒ Phase
    wird NICHT übersprungen (Output ≠ Dry). Beweist, dass die Bandwahl korrekt ist.
    """
    sr = 48000
    phase = FrequencyRestorationPhase(sample_rate=sr)
    audio = _make_rolled_off_audio(sr)
    result = phase.process(audio.copy(), sample_rate=sr, material_type="unknown")
    # Kein PSY-A1-Skip-Flag und der Output weicht vom Dry-Signal ab (echte HF-Synthese).
    assert result.metadata.get("subaudible_defects_skipped") is not True
    assert not np.allclose(result.audio.astype(np.float32), audio.astype(np.float32), atol=1e-6)


def test_psy_a1_subaudible_delta_stereo_layout_preserved(monkeypatch):
    """Stereo-Layout-Invariante: Gate-Frühreturn behält (C, N) statt (N, C).

    Fängt Mean-Achsen-/Layout-Kollaps-Bugklasse ab (Befund a8929178).
    """
    import backend.core.dsp.audibility_gate as ag

    sr = 48000
    phase = FrequencyRestorationPhase(sample_rate=sr)
    mono = _make_rolled_off_audio(sr)
    audio = np.vstack([mono, mono * 0.9]).astype(np.float32)  # channels-first (2, N)

    monkeypatch.setattr(
        ag,
        "defect_audibility",
        lambda *a, **kw: {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True},
    )

    result = phase.process(audio.copy(), sample_rate=sr, material_type="unknown")
    assert result.metadata.get("subaudible_defects_skipped") is True
    assert result.audio.shape == audio.shape, f"Layout-Kollaps: {result.audio.shape} statt {audio.shape}"
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=5)
