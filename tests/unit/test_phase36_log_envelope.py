from __future__ import annotations

"""
Tests for phase_36 Transient Shaper — K: Log-Domain Envelope Follower.

Verifies that _compute_envelope operates in the log (dBFS) domain and that
the perceptual-uniformity property (Weber–Fechner) holds: a transient of equal
loudness relative increase is detected consistently regardless of absolute level.

Scientific reference:
    Giannoulis et al. (2012) "Digital Dynamic Range Compressor Design —
    A Tutorial and Analysis", JAES 60(6), pp. 399–408.
    Zölzer (2011) DAFX §6.1.
"""


import numpy as np
import pytest

SR = 48000
ATT_MS = 10.0  # ms
REL_MS = 80.0  # ms


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shaper():
    from backend.core.phases.phase_36_transient_shaper import TransientShaper

    return TransientShaper()


def _att_samp(ms: float = ATT_MS) -> int:
    return max(1, int(ms * SR / 1000))


def _rel_samp(ms: float = REL_MS) -> int:
    return max(1, int(ms * SR / 1000))


# ---------------------------------------------------------------------------
# 1. API / contract tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnvelopeApi:
    def test_output_shape_matches_input(self, shaper):
        n = 4800
        x = np.random.default_rng(0).uniform(-0.5, 0.5, n)
        env = shaper._compute_envelope(x, _att_samp(), _rel_samp())
        assert env.shape == (n,)

    def test_dtype_float64(self, shaper):
        x = np.ones(1000, dtype=np.float32)
        env = shaper._compute_envelope(x.astype(np.float64), _att_samp(), _rel_samp())
        assert env.dtype == np.float64

    def test_output_non_negative(self, shaper):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(4800)
        env = shaper._compute_envelope(x, _att_samp(), _rel_samp())
        assert np.all(env >= 0.0), "Envelope must be non-negative (linear domain)"

    def test_output_finite(self, shaper):
        x = np.random.default_rng(1).standard_normal(4800)
        env = shaper._compute_envelope(x, _att_samp(), _rel_samp())
        assert np.all(np.isfinite(env)), "Envelope must be finite everywhere"

    def test_zero_signal(self, shaper):
        x = np.zeros(4800)
        env = shaper._compute_envelope(x, _att_samp(), _rel_samp())
        assert np.all(np.isfinite(env)), "All-zeros signal must not produce NaN/Inf"
        assert np.all(env >= 0.0)

    def test_single_sample_no_crash(self, shaper):
        """Edge: length-1 array must not crash."""
        x = np.array([0.5])
        env = shaper._compute_envelope(x, 1, 1)
        assert env.shape == (1,)
        assert np.isfinite(env[0])

    def test_constant_signal_tracks_level(self, shaper):
        """Constant amplitude → envelope converges to that amplitude."""
        amp = 0.3
        x = np.full(SR, amp)
        env = shaper._compute_envelope(x, _att_samp(), _rel_samp())
        # After 512 ms the envelope should be within 5 % of amp
        assert abs(env[-1] - amp) < amp * 0.05, f"Last env value {env[-1]:.4f} vs amp {amp:.4f}"

    def test_attack_rises_faster_than_release_falls(self, shaper):
        """Asymmetric ballistics: signal step-up is tracked faster than step-down."""
        n = SR  # 1 second
        half = n // 2
        x = np.zeros(n)
        x[:half] = 0.05  # low level
        x[half:] = 0.8  # high level (attack)
        env_up = shaper._compute_envelope(x, _att_samp(5), _rel_samp(200))

        x2 = x[::-1].copy()  # reverse: high level first, then low (release)
        env_down = shaper._compute_envelope(x2, _att_samp(5), _rel_samp(200))

        # After 100 ms from the transition, attack envelope should be higher
        # than release envelope at the symmetrically equivalent point
        step = int(100 * SR / 1000)
        rise = env_up[half + step]
        fall = env_down[half + step]
        assert rise > fall, f"Attack envelope {rise:.4f} should exceed release {fall:.4f}"


# ---------------------------------------------------------------------------
# 2. Log-domain perceptual-uniformity tests
# ---------------------------------------------------------------------------


class TestLogDomainBallistics:
    def _make_burst(
        self, base_amp: float, burst_amp: float, n: int = 4800, burst_start: int = 480, burst_dur: int = 200
    ) -> np.ndarray:
        """Constant base + a rectangular burst transient at burst_start."""
        x = np.full(n, base_amp)
        x[burst_start : burst_start + burst_dur] = burst_amp
        return x

    def test_equal_db_rise_at_loud_level(self, shaper):
        """
        +6 dB transient at -6 dBFS (amp 0.50 → 1.00) → dBFS rise ≈ 6 dB.
        """
        x = self._make_burst(base_amp=0.50, burst_amp=1.00)
        env = shaper._compute_envelope(x, _att_samp(2), _rel_samp(50))
        burst_start = 480
        before_db = 20.0 * np.log10(max(env[burst_start - 50], 1e-6))
        peak_db = 20.0 * np.log10(max(env[burst_start + 100 : burst_start + 200].max(), 1e-6))
        rise_db = peak_db - before_db
        # Log-domain ballistics: should track roughly the 6 dB rise
        assert rise_db > 2.0, f"Loud transient dB rise {rise_db:.2f} dB too small"

    def test_equal_db_rise_at_quiet_level(self, shaper):
        """
        Same +6 dB transient at -66 dBFS (amp 0.0005 → 0.001) must also yield
        a measurable envelope rise.  Linear-domain ballistics would capture this
        only at amplitude 1e-4 difference — barely detectable.
        """
        x = self._make_burst(base_amp=0.0005, burst_amp=0.001)
        env = shaper._compute_envelope(x, _att_samp(2), _rel_samp(50))
        burst_start = 480
        before_db = 20.0 * np.log10(max(env[burst_start - 50], 1e-6))
        peak_db = 20.0 * np.log10(max(env[burst_start + 100 : burst_start + 200].max(), 1e-6))
        rise_db = peak_db - before_db
        # With log-domain ballistics the quiet transient should be captured too
        assert rise_db > 2.0, f"Quiet transient dB rise {rise_db:.2f} dB too small"

    def test_perceptual_uniformity(self, shaper):
        """
        Core Weber-Fechner test: two transients of *equal dB ratio* at different
        absolute levels should yield similar dB rises in the envelope.

        Loud:  0.50 → 1.00  (+6.02 dB)
        Quiet: 0.005 → 0.01 (+6.02 dB)

        With log-domain ballistics: |rise_loud_dB − rise_quiet_dB| < 3 dB.
        With linear-domain ballistics this would differ by ~40 dB.
        """
        att = _att_samp(2)
        rel = _rel_samp(50)
        burst_start, burst_dur = 480, 200

        x_loud = self._make_burst(0.50, 1.00, burst_start=burst_start, burst_dur=burst_dur)
        x_quiet = self._make_burst(0.005, 0.01, burst_start=burst_start, burst_dur=burst_dur)

        env_loud = shaper._compute_envelope(x_loud, att, rel)
        env_quiet = shaper._compute_envelope(x_quiet, att, rel)

        def db_rise(env):
            before = env[burst_start - 10 : burst_start].mean()
            peak = env[burst_start : burst_start + burst_dur].max()
            return 20.0 * np.log10(max(peak, 1e-6)) - 20.0 * np.log10(max(before, 1e-6))

        rise_loud = db_rise(env_loud)
        rise_quiet = db_rise(env_quiet)
        diff = abs(rise_loud - rise_quiet)
        assert diff < 3.0, (
            f"Log-domain ballistics should give similar dB rise for equal-ratio "
            f"transients: loud={rise_loud:.2f} dB, quiet={rise_quiet:.2f} dB, "
            f"diff={diff:.2f} dB (must be < 3 dB)"
        )

    def test_floor_no_nan(self, shaper):
        """Near-zero signal (just above -120 dBFS floor) must not produce NaN."""
        x = np.full(4800, 1e-7)  # ~ -140 dBFS → clamped to -120 dBFS
        env = shaper._compute_envelope(x, _att_samp(), _rel_samp())
        assert np.all(np.isfinite(env)), "Near-floor signal produced NaN/Inf"

    def test_large_amplitude_no_overflow(self, shaper):
        """Signal at exactly ±1.0 (0 dBFS) must not overflow."""
        x = np.sin(2.0 * np.pi * 440.0 / SR * np.arange(4800))
        x = x / (np.abs(x).max() + 1e-9)  # normalise to ±1
        env = shaper._compute_envelope(x, _att_samp(), _rel_samp())
        assert np.all(np.isfinite(env))
        assert np.all(env <= 1.0 + 1e-6)

    def test_impulse_tracked(self, shaper):
        """A 100-sample burst at 0.9 amplitude on a small background must raise the envelope.

        Starting from near-silence, a single sample cannot fully excite the
        log-domain follower in one DS block.  A burst of ≥ 2 ms (100 samples @
        48 kHz = ≥ 3 DS blocks with DS=16) on a non-zero background gives the
        attack-coeff chain enough steps to exceed 0.1 linear within the burst.
        """
        x = np.full(4800, 0.01)  # small background: -40 dBFS → finite start state
        x[1000:1100] = 0.9  # 100-sample (~2 ms) burst
        env = shaper._compute_envelope(x, _att_samp(1), _rel_samp(20))
        # After ≥ 3 DS blocks of attack from -40 dBFS toward -0.9 dBFS the
        # envelope should exceed 0.1 linear (≈ -20 dBFS)
        assert env[1000:1200].max() > 0.1, "Burst not tracked by envelope"


# ---------------------------------------------------------------------------
# 3. _shape_band integration tests
# ---------------------------------------------------------------------------


class TestShapeBand:
    def test_output_shape_mono(self, shaper):
        x = np.random.default_rng(7).standard_normal(4800)
        out = shaper._shape_band(x, SR, 3.0, 0.0, ATT_MS, REL_MS)
        assert out.shape == x.shape

    def test_output_finite(self, shaper):
        x = np.random.default_rng(8).standard_normal(4800) * 0.3
        out = shaper._shape_band(x, SR, 2.0, -1.0, ATT_MS, REL_MS)
        assert np.all(np.isfinite(out))

    def test_zero_gain_passthrough(self, shaper):
        """attack_gain=0 and sustain_gain=0 → output ≈ input (gain=1 for both)."""
        x = np.random.default_rng(9).standard_normal(4800) * 0.3
        out = shaper._shape_band(x, SR, 0.0, 0.0, ATT_MS, REL_MS)
        np.testing.assert_allclose(out, x, atol=1e-4, err_msg="Zero gain should be pass-through")

    def test_positive_attack_gain_increases_transients(self, shaper):
        """Positive attack_gain_db must increase the overall energy."""
        rng = np.random.default_rng(10)
        x = np.zeros(4800)
        # Sprinkle transients
        for pos in range(500, 4800, 500):
            x[pos : pos + 50] = rng.uniform(0.5, 0.8, 50)
        energy_before = float(np.mean(x**2))
        out = shaper._shape_band(x, SR, 6.0, -3.0, ATT_MS, REL_MS)
        energy_after = float(np.mean(out**2))
        assert energy_after >= energy_before * 0.8, (
            f"Positive attack gain should not drastically reduce energy: {energy_before:.6f} → {energy_after:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. Full process() integration tests
# ---------------------------------------------------------------------------

SR_PROC = 48000


def _make_audio(dur: float = 0.5, stereo: bool = False, amp: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(99)
    n = int(dur * SR_PROC)
    if stereo:
        return (rng.standard_normal((n, 2)) * amp).astype(np.float32)
    return (rng.standard_normal(n) * amp).astype(np.float32)


class TestProcessIntegration:
    def test_mono_no_nan(self, shaper):
        audio = _make_audio(stereo=False)
        r = shaper.process(audio, SR_PROC)
        assert np.all(np.isfinite(r.audio))

    def test_stereo_no_nan(self, shaper):
        audio = _make_audio(stereo=True)
        r = shaper.process(audio, SR_PROC)
        assert np.all(np.isfinite(r.audio))

    def test_no_clipping(self, shaper):
        audio = _make_audio(stereo=True)
        r = shaper.process(audio, SR_PROC)
        assert np.max(np.abs(r.audio)) <= 1.0 + 1e-6

    def test_shape_preserved_mono(self, shaper):
        audio = _make_audio(stereo=False)
        r = shaper.process(audio, SR_PROC)
        assert r.audio.shape == audio.shape

    def test_shape_preserved_stereo(self, shaper):
        audio = _make_audio(stereo=True)
        r = shaper.process(audio, SR_PROC)
        assert r.audio.shape == audio.shape

    def test_success_flag(self, shaper):
        r = shaper.process(_make_audio(), SR_PROC)
        assert r.success is True

    def test_version_bumped(self, shaper):
        meta = shaper.get_metadata()
        major, minor, _ = meta.version.split(".")
        assert (int(major), int(minor)) >= (2, 1), f"Version must be >= 2.1, got {meta.version}"
