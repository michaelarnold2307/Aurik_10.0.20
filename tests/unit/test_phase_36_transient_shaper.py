"""tests/unit/test_phase_36_transient_shaper.py — §SOTA-PSY-A1 (TransientShaper)."""

import numpy as np

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_36_transient_shaper import TransientShaper


def _make_transient_audio(sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    """Signal mit Transienten (Impulse) + Grundton — löst Attack-Synthesis aus.

    Peak unter 1.0 halten, damit der Dry-Rückgabe-Vergleich im Skip-Test kein
    Hard-Clipping sieht (Befund der Layout-/Clip-Bugklasse).
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    x = (np.sin(2 * np.pi * 440 * t) * 0.25).astype(np.float32)
    for i in range(0, int(sr * duration), int(0.1 * sr)):
        x[i : i + 50] += 0.5
    return x


def _make_phase() -> TransientShaper:
    return TransientShaper()


# ---------------------------------------------------------------------------
# §SOTA-PSY-A1: subaudibles Transienten-Delta wird gegated (Dry-Passthrough)
# ---------------------------------------------------------------------------


def test_psy_a1_subaudible_delta_passthrough(monkeypatch):
    """Gate meldet skippable ⇒ Phase gibt Dry zurück (§SOTA-PSY-A1)."""
    import backend.core.dsp.audibility_gate as ag

    sr = 48000
    phase = _make_phase()
    audio = _make_transient_audio(sr)

    monkeypatch.setattr(
        ag,
        "defect_audibility",
        lambda *a, **kw: {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True},
    )

    result = phase.process(audio.copy(), sample_rate=sr, material=MaterialType.CD_DIGITAL)
    assert result.metadata.get("subaudible_defects_skipped") is True
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=5)


def test_psy_a1_real_band_does_not_deactivate_phase():
    """Realer Gegentest: hörbares Transienten-Delta im richtigen Band (800 Hz–10 kHz)
    ⇒ Phase wird NICHT übersprungen (Output ≠ Dry). Beweist, dass die Bandwahl korrekt ist.
    """
    sr = 48000
    phase = _make_phase()
    audio = _make_transient_audio(sr)
    result = phase.process(audio.copy(), sample_rate=sr, material=MaterialType.CD_DIGITAL)
    # Kein PSY-A1-Skip-Flag und der Output weicht vom Dry-Signal ab (echtes Shaping).
    assert result.metadata.get("subaudible_defects_skipped") is not True
    assert not np.allclose(result.audio.astype(np.float32), audio.astype(np.float32), atol=1e-6)


def test_psy_a1_subaudible_delta_stereo_layout_preserved(monkeypatch):
    """Stereo-Layout-Invariante: Gate-Frühreturn behält (C, N) statt (N, C).

    Fängt Mean-Achsen-/Layout-Kollaps-Bugklasse ab (Befund a8929178).
    Phase 36 arbeitet channels-first (C, N) ohne Layout-Normalisierung.
    """
    import backend.core.dsp.audibility_gate as ag

    sr = 48000
    phase = _make_phase()
    mono = _make_transient_audio(sr)
    audio = np.vstack([mono, mono * 0.9]).astype(np.float32)  # channels-first (2, N)

    monkeypatch.setattr(
        ag,
        "defect_audibility",
        lambda *a, **kw: {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True},
    )

    result = phase.process(audio.copy(), sample_rate=sr, material=MaterialType.CD_DIGITAL)
    assert result.metadata.get("subaudible_defects_skipped") is True
    assert result.audio.shape == audio.shape, f"Layout-Kollaps: {result.audio.shape} statt {audio.shape}"
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=5)
