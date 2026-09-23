"""tests/unit/test_phase_36_transient_shaper.py — §SOTA-PSY-A1 (TransientShaper)."""

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# §SR-CG Crackle-Guard: Knistern/Click/Pop-Events nicht mitboosten (vinyl)
# ---------------------------------------------------------------------------


def test_crackle_guard_zeroes_delta_inside_events():
    """Innerhalb der Knistern-Events wird das Transienten-Delta ausgeblendet —
    außerhalb bleibt das Shaping vollständig erhalten."""
    sr = 48000
    n = sr
    original = np.zeros(n, dtype=np.float32)
    shaped = np.ones(n, dtype=np.float32) * 0.5
    kwargs = {"defect_locations": {"crackle": [(0.10, 0.20)], "click": [(0.50, 0.55)]}}
    out = TransientShaper._apply_crackle_guard(shaped, original, sr, kwargs)
    # Kern des Events: komplett zurück auf Original (0)
    assert float(np.max(np.abs(out[int(0.13 * sr) : int(0.17 * sr)]))) < 1e-6
    assert float(np.max(np.abs(out[int(0.52 * sr) : int(0.53 * sr)]))) < 1e-6
    # Außerhalb: Shaping voll erhalten
    assert float(out[0]) == pytest.approx(0.5, abs=1e-6)
    assert float(out[int(0.90 * sr)]) == pytest.approx(0.5, abs=1e-6)


def test_crackle_guard_fail_open_without_events():
    """Ohne defect_locations (oder leer) ist der Guard die Identität —
    kein Verhalten ohne Defekt-Scan geändert."""
    sr = 48000
    n = sr // 4
    original = np.zeros(n, dtype=np.float32)
    shaped = np.ones(n, dtype=np.float32) * 0.5
    for kwargs in ({}, {"defect_locations": None}, {"defect_locations": {"crackle": []}}):
        out = TransientShaper._apply_crackle_guard(shaped, original, sr, kwargs)
        np.testing.assert_array_equal(out, shaped)


def test_crackle_guard_stereo_channels_first_and_deterministic():
    """Stereo-Layout-Invariante (C, N) bleibt erhalten; deterministisch (§G5 copilot-instructions.md)."""
    sr = 48000
    n = sr // 2
    original = np.zeros((2, n), dtype=np.float32)
    shaped = np.ones((2, n), dtype=np.float32) * 0.5
    kwargs = {"defect_locations": {"crackle": [(0.10, 0.15)]}}
    out1 = TransientShaper._apply_crackle_guard(shaped, original, sr, kwargs)
    out2 = TransientShaper._apply_crackle_guard(shaped, original, sr, kwargs)
    assert out1.shape == shaped.shape
    np.testing.assert_array_equal(out1, out2)
    # Beide Kanäle identisch ausgeblendet (kein L/R-Zeitversatz)
    np.testing.assert_array_equal(out1[0], out1[1])
    assert float(np.max(np.abs(out1[0][int(0.12 * sr) : int(0.13 * sr)]))) < 1e-6
    assert float(out1[0][int(0.40 * sr)]) == pytest.approx(0.5, abs=1e-6)
