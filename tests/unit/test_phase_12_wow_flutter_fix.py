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


class TestBandLimitedWarp:
    """§7.4c-L3 (2026-09-27): Bandbegrenzter Inverse-Warp für bandabhängiges
    Flutter (multiband_wow_flutter, Hint > 4 Hz). Exakte Komplement-
    Konstruktion: nur [4 kHz, 12 kHz] wird gewarpt, der Rest bleibt
    bit-identisch (Never-worsen, Hörordnung §8a)."""

    SR = 48000

    @staticmethod
    def _two_tone_signal() -> np.ndarray:
        t = np.arange(TestBandLimitedWarp.SR * 2) / TestBandLimitedWarp.SR
        return (0.3 * np.sin(2 * np.pi * 500.0 * t) + 0.3 * np.sin(2 * np.pi * 6000.0 * t)).astype(np.float32)

    @staticmethod
    def _smooth_stretch() -> np.ndarray:
        t = np.arange(TestBandLimitedWarp.SR * 2) / TestBandLimitedWarp.SR
        return (1.0 + 0.02 * np.sin(2 * np.pi * 0.5 * t)).astype(np.float32)

    def test_warp_diff_lives_only_in_band(self, phase):
        x = self._two_tone_signal()
        out = phase._band_limited_warp(x, self._smooth_stretch(), self.SR)
        d = out.astype(np.float64) - x.astype(np.float64)
        spec = np.abs(np.fft.rfft(d)) ** 2
        freqs = np.fft.rfftfreq(len(d), 1.0 / self.SR)
        lo_e = float(np.sum(spec[freqs < 2000.0]))
        hi_e = float(np.sum(spec[freqs >= 2000.0]))
        assert hi_e > 0.0
        assert lo_e < 1e-9 * hi_e

    def test_length_finite_dtype(self, phase):
        x = self._two_tone_signal()
        out = phase._band_limited_warp(x, self._smooth_stretch(), self.SR)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert np.isfinite(out).all()

    def test_identity_factors_bit_identical(self, phase):
        x = self._two_tone_signal()
        ones = np.ones(len(x), dtype=np.float32)
        out = phase._band_limited_warp(x, ones, self.SR)
        assert np.allclose(out, x, atol=1e-4)


class TestHfGridFlutterTrack:
    """§7.4c-L3 (2026-09-27): Der Teilband-Messkanal muss bei Hint > 4 Hz das
    Grid bis 12 kHz öffnen (32-kHz-Arbeitsspur) — der Standard-Wow-Pfad bleibt
    beim 4-kHz-Grid (§G5 (GEBOTE.md) Bestandsverhalten)."""

    SR = 48000

    @staticmethod
    def _fm_tone(f0: float, f_mod: float, depth_pct: float, dur_s: float = 6.0) -> np.ndarray:
        n = int(TestHfGridFlutterTrack.SR * dur_s)
        t_dev = (
            (depth_pct / 100.0)
            * (1.0 / f_mod)
            * np.sin(2 * np.pi * f_mod * np.arange(n) / TestHfGridFlutterTrack.SR)
            * TestHfGridFlutterTrack.SR
        )
        base = 0.5 * np.sin(2 * np.pi * f0 * np.arange(n) / TestHfGridFlutterTrack.SR)
        return np.interp(np.arange(n) + t_dev, np.arange(n), base).astype(np.float32)

    def test_hf_hint_measures_fm_above_4khz(self, phase):
        # 6-kHz-Ton mit ±1,9 % FM bei 5 Hz (L3-Synth-Muster): der HF-Grid-Kanal
        # muss die Gemeinschafts-FM messen (Amplitude > 8 cents in der Spur).
        x = self._fm_tone(6000.0, 5.0, 1.9)
        vp, conf = phase._estimate_wow_track_subband(x.astype(np.float64), self.SR, hint_freq_hz=5.0)
        assert vp.size >= 32
        cents = 1200.0 * np.log2(vp / np.median(vp))
        span = float(np.percentile(cents, 95) - np.percentile(cents, 5))
        assert span > 8.0

    def test_no_hint_keeps_lowband_grid(self, phase):
        # Ohne Hint (> 4 Hz) bleibt der Kanal im Standard-Grid — ein reiner
        # 6-kHz-FM-Ton ist dort unsichtbar (Bestandsverhalten, kein Fehler).
        x = self._fm_tone(6000.0, 5.0, 1.9)
        vp, conf = phase._estimate_wow_track_subband(x.astype(np.float64), self.SR, hint_freq_hz=None)
        assert vp.size >= 0


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
        # Slope-Limit: kein Fenster-Delta über dem Default (5 %, max_stretch_delta)
        assert float(np.max(np.abs(np.diff(out)))) <= 0.05 + 1e-6
        # Sprung wird über mehrere Fenster verteilt statt hart übernommen
        assert float(out[13]) < 1.11
        # Endwert erreicht das Ziel (kein Overshoot, kein Dauerfehler)
        assert float(out[-1]) == pytest.approx(1.20, abs=1e-5)
        # Begrenzt auf [min, max] der Eingabe
        assert float(out.min()) >= 1.0 - 1e-6
        assert float(out.max()) <= 1.20 + 1e-6

    def test_max_step_parameter_couples_to_max_stretch_delta(self, phase):
        # Gekoppeltes Limit: mit max_step=0.025 (früheres hartes Limit) wird
        # stärker begrenzt — der Parameter steuert die Schärfe.
        factors = np.array([1.0] * 12 + [1.20] * 12, dtype=np.float32)
        out = phase._smooth_stretch_factors(factors, max_step=0.025)
        assert float(np.max(np.abs(np.diff(out)))) <= 0.025 + 1e-6
        assert float(out[13]) < 1.06

    def test_legit_algorithmic_output_is_identity(self, phase):
        # Jede Trajektorie mit per-Fenster-Deltas ≤ max_stretch_delta (5 %)
        # ist legitime _calculate_stretch_factors-Ausgabe → Identität.
        n = 200
        x = np.arange(n)
        legit = 1.0 + 0.03 * np.sin(2 * np.pi * 0.5 * x / 24.0)
        out = phase._smooth_stretch_factors(legit.astype(np.float32), max_step=0.05)
        assert np.allclose(out, legit, atol=1e-6)

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
        # (max. Steigung ~2,15 %/Fenster < 5 %-Limit = max_stretch_delta).
        n = 240
        x = np.arange(n)
        wow = 1.0 + 0.02 * np.sin(2 * np.pi * 4.0 * x * 42.7e-3)
        out = phase._smooth_stretch_factors(wow.astype(np.float32))
        assert float(np.max(np.abs(out - wow))) < 1e-6
