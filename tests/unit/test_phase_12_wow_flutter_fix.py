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


class TestSmoothStretchFactors:
    """§WF-V3: Trajektorien-Glättung vor der Zeitstreckung.

    Produktionsbefund vinyl/1970: Stufen in den Stretch-Faktoren erzeugen
    Zeitwarp-Sprünge → Energie-Sprünge >6 dB/100 ms (TemporalConsistencyGuard),
    Pre-Echo-Befunde und timbre_authentizitaet-Degradation nach phase_12.
    """

    def test_step_is_spread_and_slope_limited(self, phase):
        factors = np.array([1.0] * 12 + [1.20] * 12, dtype=np.float32)
        out = phase._smooth_stretch_factors(factors)
        assert out.shape == factors.shape
        assert out.dtype == np.float32
        # Slope-Limit: kein Fenster-Delta über 2,5 %
        assert float(np.max(np.abs(np.diff(out)))) <= 0.025 + 1e-6
        # Sprung wird über mehrere Fenster verteilt statt hart übernommen
        assert float(out[13]) < 1.06
        # Endwert erreicht das Ziel (kein Overshoot, kein Dauerfehler)
        assert float(out[-1]) == pytest.approx(1.20, abs=1e-5)
        # Begrenzt auf [min, max] der Eingabe
        assert float(out.min()) >= 1.0 - 1e-6
        assert float(out.max()) <= 1.20 + 1e-6

    def test_smooth_trajectory_nearly_identity_and_deterministic(self, phase):
        smooth = np.array([1.0, 1.001, 0.999, 1.002, 1.0, 0.998], dtype=np.float32)
        out1 = phase._smooth_stretch_factors(smooth)
        out2 = phase._smooth_stretch_factors(smooth)
        # Glatte Trajektorie bleibt nahezu unverändert
        assert np.allclose(out1, smooth, atol=2e-3)
        # Deterministisch (§G5 copilot-instructions.md)
        assert np.array_equal(out1, out2)

    def test_passthrough_short_or_non_1d(self, phase):
        short = np.array([1.0, 1.3], dtype=np.float32)
        assert np.array_equal(phase._smooth_stretch_factors(short), short)
        empty = np.array([], dtype=np.float32)
        assert phase._smooth_stretch_factors(empty).size == 0
        two_d = np.ones((4, 4), dtype=np.float32)
        assert np.array_equal(phase._smooth_stretch_factors(two_d), two_d)

    def test_wow_band_survives(self, phase):
        # Legitimes 4-Hz-Wow (±2 %) ist die Identität der Projektion
        # (max. Steigung ~2,15 %/Fenster < 2,5 %-Limit).
        n = 240
        x = np.arange(n)
        wow = 1.0 + 0.02 * np.sin(2 * np.pi * 4.0 * x * 42.7e-3)
        out = phase._smooth_stretch_factors(wow.astype(np.float32))
        assert float(np.max(np.abs(out - wow))) < 1e-6
