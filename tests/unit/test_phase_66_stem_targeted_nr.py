"""tests/unit/test_phase_66_stem_targeted_nr.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_66_stem_targeted_nr import StemTargetedNRPhase


@pytest.fixture
def phase():
    return StemTargetedNRPhase()


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
# §SOTA-PSY-A1: subaudibles Stem-NR-Delta wird gegated (Dry-Passthrough)
# ---------------------------------------------------------------------------


class _FakeStemResult:
    def __init__(self, stems):
        self.sdri_db = 12.0
        self.confidence = 0.9
        self.stems = stems


class _FakeSeparator:
    def separate(self, audio, sample_rate, stems=None):
        n = audio.shape[0] if audio.ndim == 1 else audio.shape[-1]
        voc = np.zeros(n, dtype=np.float32)
        other = np.zeros(n, dtype=np.float32)
        return _FakeStemResult({"vocals": voc, "other": other})


class _FakeDFN:
    def enhance(self, stem, sr, energy_bias_db=0.0):
        return np.asarray(stem, dtype=np.float32).copy()


def test_psy_a1_subaudible_delta_passthrough(phase, monkeypatch):
    """Gate meldet skippable ⇒ Phase gibt Dry zurück (PSY-A1-Rollback)."""
    sr = 48000
    rng = np.random.RandomState(0)
    audio = (rng.randn(sr * 4) * 0.05).astype(np.float32)  # ≥ 3 s für Separator-Gate

    monkeypatch.setattr(phase, "_get_separator", lambda: (_FakeSeparator(), "fake"))
    monkeypatch.setattr(phase, "_get_dfn", lambda: _FakeDFN())

    import backend.core.dsp.audibility_gate as ag

    monkeypatch.setattr(
        ag,
        "defect_audibility",
        lambda *a, **kw: {"audible": False, "delta_db": 0.0, "threshold_db": 0.0, "skippable": True},
    )

    result = phase.process(
        audio.copy(),
        sr,
        material_type="vinyl",
        quality_mode="restoration",
        panns_singing=0.8,
        strength=1.0,
    )
    assert result.success is True
    assert "PSY-A1" in (result.metadata.get("rollback_reason") or "")
    np.testing.assert_array_almost_equal(result.audio, audio, decimal=5)
