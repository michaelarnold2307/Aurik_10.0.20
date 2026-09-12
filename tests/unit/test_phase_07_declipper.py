"""tests/unit/test_phase_07_declipper.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_07_declipper import DeclipperPhase


@pytest.fixture
def phase():
    return DeclipperPhase()


@pytest.fixture
def audio():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    clipped = np.clip(np.sin(2 * np.pi * 440 * t) * 1.5, -1.0, 1.0)
    return clipped.astype(np.float32)


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


def _clipped_bursts(amp: float) -> np.ndarray:
    """0.8-Sinus mit drei 25-ms-Bursts ×amp, hart bei ±1.0 geclippt."""
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 440 * t) * 0.8
    burst = int(sr * 0.025)
    for onset_s in (0.25, 0.5, 0.75):
        s0 = int(onset_s * sr)
        x[s0 : s0 + burst] *= amp
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def test_mild_clipping_sparse_rejected_pchip_stays(phase):
    """Evidenz (DECLIPPER_SOTA_PLAN.md Slice A): mildes 2.5× — Sparse verschlechtert, PCHIP bleibt."""
    x = _clipped_bursts(2.5)
    result = phase.process(x, sample_rate=48000)
    assert result.metrics.get("declip_applied") is True
    assert result.metrics.get("sparse_used") is False


def test_strong_clipping_uses_sparse_path(phase):
    """Evidenz: starkes 6× — Sparse übernimmt (Harmonik-Proxy sinkt)."""
    x = _clipped_bursts(6.0)
    result = phase.process(x, sample_rate=48000)
    assert result.metrics.get("declip_applied") is True
    assert result.metrics.get("sparse_used") is True


def test_sparse_path_deterministic(phase):
    x = _clipped_bursts(6.0)
    r1 = phase.process(x, sample_rate=48000)
    r2 = phase.process(x, sample_rate=48000)
    np.testing.assert_array_equal(r1.audio, r2.audio)  # §G5 (copilot-instructions.md)


def test_sparse_path_reduces_thd(phase):
    """THD-Verhältnis (Harmonische 2–10 vs. Grundton) sinkt durch den Sparse-Zweig."""
    x = _clipped_bursts(6.0)
    result = phase.process(x, sample_rate=48000)
    assert result.metrics.get("sparse_used") is True

    def _thd(y: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(y.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(y), 1.0 / 48000)
        fund = float(spec[np.argmin(np.abs(freqs - 440))]) + 1e-12
        harm = 0.0
        for h in range(2, 11):
            _i = int(np.argmin(np.abs(freqs - 440 * h)))
            harm += float(spec[_i]) ** 2
        return float(np.sqrt(harm) / fund)

    assert _thd(result.audio) < _thd(x)


def test_sparse_fallback_to_pchip(phase, monkeypatch):
    """Sparse-Modul nicht verfügbar → PCHIP bleibt, kein Crash (§V6 (copilot-instructions.md))."""
    x = _clipped_bursts(6.0)

    def _boom(_a, _sr, _t, **_kw):
        raise RuntimeError("sparse unavailable")

    monkeypatch.setattr("backend.core.dsp.sparse_declipper.sparse_declip", _boom)
    result = phase.process(x, sample_rate=48000)
    assert result.metrics.get("declip_applied") is True
    assert result.metrics.get("sparse_used") is False
    assert np.isfinite(result.audio).all()
