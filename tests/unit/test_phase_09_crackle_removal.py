"""tests/unit/test_phase_09_crackle_removal.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_09_crackle_removal import CrackleRemovalPhase


@pytest.fixture
def phase():
    return CrackleRemovalPhase()


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


def test_banquet_stereo_per_channel_layout(phase, monkeypatch, audio):
    """BANQUET-Stereo: kanalweise Inferenz, Layout identisch zum Input (§v10.95)."""
    import backend.core.phases.phase_09_crackle_removal as _mod

    class _FakeSession:
        class _In:
            name = "input_fixed"
            shape = [1, 48000]

        def get_inputs(self):
            return [_FakeSession._In()]

        def run(self, _, feeds):
            return [np.asarray(feeds["input_fixed"], dtype=np.float32)]

    monkeypatch.setattr(_mod, "_get_banquet_onnx_session", lambda: _FakeSession())

    stereo_cf = np.stack([audio, audio], axis=0).astype(np.float32)
    out_cf = phase._remove_crackle_onnx_direct(stereo_cf, 48000, {})
    assert out_cf.shape == (2, len(audio))
    np.testing.assert_allclose(out_cf[0], out_cf[1], atol=1e-6)

    stereo_cl = stereo_cf.T.copy()
    out_cl = phase._remove_crackle_onnx_direct(stereo_cl, 48000, {})
    assert out_cl.shape == (len(audio), 2)
    np.testing.assert_allclose(out_cl[:, 0], out_cl[:, 1], atol=1e-6)
