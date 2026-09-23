"""tests/unit/test_phase_12_wow_flutter_fix.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix


@pytest.fixture
def phase():
    return WowFlutterFix()


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


def test_melody_guard_sets_refusal_flag(phase):
    """§WF-V2-Kopplung: Musikalische Tonhöhen-Spanne (> Limit) muss
    `_melody_guard_refused` setzen und flache Stretch-Faktoren liefern —
    die Spektral-Warp-Versorgung darf diesen Guard nicht überschreiben
    (Produktionsbefund „Vogel der Nacht": pitch_instability nach phase_12
    trotz wow=0.00 und Melodie-Guard-Ablehnung)."""
    sr = 48000
    n = sr * 2
    t = np.arange(n) / sr
    f0 = 220.0 * 2.0 ** (np.sin(2 * np.pi * 0.5 * t) * 12.0 / 12.0)  # ±1 Oktave Melodie
    conf = np.ones(n)
    phase._wow_sev_for_stretch = 0.0
    phase._melody_guard_refused = False
    sf = phase._calculate_stretch_factors(f0, conf, 0.5, max_stretch_delta=0.05)
    assert np.allclose(sf, 1.0)
    assert phase._melody_guard_refused is True
