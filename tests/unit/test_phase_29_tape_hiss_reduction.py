"""tests/unit/test_phase_29_tape_hiss_reduction.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase


@pytest.fixture
def phase():
    return TapeHissReductionPhase()


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


def test_ana11_consonant_protection_keeps_region_closer_to_input(monkeypatch):
    """§SOTA-Analogie-Korrektur 2026-09-17 (ANA-11): Die DSP-
    Konsonanten-Schutzmaske (phoneme_boundary_detector) war in der Produktion
    UNGENUTZT — jetzt blendet phase_29 das NR-Ergebnis in Konsonanten-Frames
    sanft Richtung Input zurück. Relative Gegenprobe: Mit Schutzmaske muss
    die Masken-Region näher am Input liegen als ohne (gemessen +6,9 % auf
    synthetischem Band-Hiss)."""
    from scipy.signal import butter, sosfilt

    import backend.core.dsp.phoneme_boundary_detector as pbd

    SR = 48000
    rng = np.random.default_rng(4)
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    sos = butter(4, [8000 / (SR / 2), 18000 / (SR / 2)], btype="bandpass", output="sos")
    hiss = sosfilt(sos, rng.standard_normal(n)).astype(np.float32)
    hiss *= 0.25 / np.abs(hiss).max()
    x = (0.15 * np.sin(2 * np.pi * 440 * t)).astype(np.float32) + hiss
    mask = np.zeros(n, dtype=bool)
    mask[int(0.4 * n) : int(0.45 * n)] = True

    def _run(_m: np.ndarray) -> np.ndarray:
        monkeypatch.setattr(pbd, "detect_phoneme_protection_mask_dsp", lambda a, sr, hop_length=512: _m)
        p = TapeHissReductionPhase()
        return p.process(x, sample_rate=SR, material_type="tape", strength=1.0).audio

    out_prot = _run(mask)
    out_none = _run(np.zeros(n, dtype=bool))
    d_prot = float(np.abs(out_prot[mask] - x[mask]).mean())
    d_none = float(np.abs(out_none[mask] - x[mask]).mean())
    assert d_prot <= d_none, f"Schutz wirkungslos/negativ: {d_prot:.5f} vs {d_none:.5f}"
