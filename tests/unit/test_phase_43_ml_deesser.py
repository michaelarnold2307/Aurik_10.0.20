"""tests/unit/test_phase_43_ml_deesser.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_43_ml_deesser import AdaptiveDeEsserPhase


@pytest.fixture
def phase():
    return AdaptiveDeEsserPhase()


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


class _FakeSibSeg:
    def __init__(self, start_s: float, end_s: float) -> None:
        self.start_s = start_s
        self.end_s = end_s


class _FakePTL:
    """Minimales PhonemeTimeline-Stub für das Segment-Gate."""

    def __init__(self, segs: list[_FakeSibSeg]) -> None:
        self._segs = segs

    def sibilant_segments(self):
        return list(self._segs)

    def sibilant_band_hz(self):
        return (4000.0, 12000.0)

    def __getattr__(self, name):
        return None


def test_psy_a1_subaudible_sibilant_segment_skipped(phase):
    """§SOTA-PSY-A1 (2026-09-16): Ein Sibilanten-Segment unter der
    Maskierungsschwelle bleibt ungezähmt (gate=0 → Original) und wird im
    Metadaten-Zähler subaudible_sibilants_skipped geführt (§4-Vertrag)."""
    sr = 48000
    n = int(sr * 1.0)
    rng = np.random.RandomState(7)
    t = np.arange(n) / sr
    x = (rng.randn(n) * 1e-4).astype(np.float32)  # leiser Grundpegel
    # Segment B: lauter 6-kHz-Sibilanten-Burst (hörbar über der Schwelle)
    b0, b1 = int(0.6 * sr), int(0.65 * sr)
    x[b0:b1] += (0.3 * np.sin(2 * np.pi * 6000 * t[b0:b1])).astype(np.float32)
    ptl = _FakePTL([_FakeSibSeg(0.20, 0.25), _FakeSibSeg(0.60, 0.65)])
    res = phase.process(x, sample_rate=sr, material_type="vinyl", phoneme_timeline=ptl)
    assert int(res.metadata.get("subaudible_sibilants_skipped", 0)) == 1
    a0, a1 = int(0.2 * sr), int(0.25 * sr)
    assert np.allclose(res.audio[a0:a1], x[a0:a1], atol=1e-4)
