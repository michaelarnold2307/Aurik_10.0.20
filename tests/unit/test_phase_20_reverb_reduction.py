"""tests/unit/test_phase_20_reverb_reduction.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_20_reverb_reduction import ReverbReduction


@pytest.fixture
def phase():
    return ReverbReduction()


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


def _neutral_nmr(a: np.ndarray, sr: int) -> object:
    """NMR neutralisieren: delta=0 — RT60-Delta isoliert messbar."""

    class _Neutral:
        ok = True
        recommended_nr_strength_delta = 0.0

    return _Neutral()


def _patch_rt60(monkeypatch: pytest.MonkeyPatch, rt60: float, conf: float) -> None:
    """Fake-DFN-Witness mit festem (rt60, conf) injizieren."""

    class _FakeDfn:
        def estimate_rt60_sec(self, a: np.ndarray, sr: int) -> tuple[float, float]:
            return (rt60, conf)

    monkeypatch.setattr("plugins.deepfilternet_v3_ii_plugin.get_loaded_deepfilternet_plugin", lambda: _FakeDfn())
    monkeypatch.setattr("backend.core.dsp.nmr_feedback.compute_nmr_score", _neutral_nmr)


def test_rt60_witness_raises_effective_strength(phase, audio, monkeypatch):
    """§SOTA-DR-V1: RT60=1,2 s + conf 0,9 ⇒ Delta +0,14 auf die effektive Stärke."""
    _patch_rt60(monkeypatch, rt60=1.2, conf=0.9)
    result = phase.process(audio, sample_rate=48000, material_type="vinyl", strength=0.3)
    md = result.metadata or {}
    assert md.get("rt60_estimate_sec") == pytest.approx(1.2)
    assert md.get("rt60_confidence") == pytest.approx(0.9)
    assert md.get("effective_strength", 0.0) == pytest.approx(0.44, abs=0.02)


def test_rt60_low_confidence_no_delta(phase, audio, monkeypatch):
    """§SOTA-DR-V1: conf < 0,5 ⇒ kein Eingriff — trockenes Material bleibt unberührt."""
    _patch_rt60(monkeypatch, rt60=1.5, conf=0.1)
    result = phase.process(audio, sample_rate=48000, material_type="vinyl", strength=0.3)
    md = result.metadata or {}
    assert md.get("effective_strength", 0.0) == pytest.approx(0.3, abs=0.02)
