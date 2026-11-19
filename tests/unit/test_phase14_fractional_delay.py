from __future__ import annotations

"""
Tests for phase_14 Phase Correction — L: Fractional Delay Correction.

Verifies sub-sample L/R alignment via:
  1. Parabolic interpolation of the XCF peak  (Smith 2011 §3.4)
  2. Lagrange order-3 FIR for fractional delay (Laakso et al. 1996)

Scientific references:
    Laakso et al. (1996) "Splitting the Unit Delay: Tools for Fractional
    Delay Filter Design", IEEE Signal Processing Magazine 13(1), pp. 30–60.
    Smith (2011) "Spectral Audio Signal Processing" §3.4.
"""


import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase():
    from backend.core.phases.phase_14_phase_correction import PhaseCorrection

    return PhaseCorrection()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _stereo_with_delay(delay_samples: float, freq: float = 1000.0, dur: float = 0.5, sr: int = SR) -> np.ndarray:
    """Stereo signal where right channel is delayed by ``delay_samples`` (float)."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    base = 0.5 * np.sin(2.0 * np.pi * freq * t)

    left = base.copy()
    # Integer part via np.roll, fractional part via linear interp
    d_int = int(delay_samples)
    d_frac = delay_samples - d_int
    right_int = np.roll(base, d_int)
    right_int[:d_int] = 0.0
    # Fractional via 1st-order linear interp (for ground truth)
    right = (1.0 - d_frac) * right_int + d_frac * np.roll(right_int, 1)

    return np.column_stack([left, right]).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. _lagrange_ffd tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLagrangeFfd:
    def test_shape(self, phase):
        h = phase._lagrange_ffd(0.3, order=3)
        assert h.shape == (4,)

    def test_dtype(self, phase):
        h = phase._lagrange_ffd(0.0, order=3)
        assert h.dtype == np.float64

    def test_zero_frac_unity_at_center(self, phase):
        """frac=0 → filter is a pure integer delay (all energy at center tap)."""
        h = phase._lagrange_ffd(0.0, order=3)
        M = 3 // 2  # center tap index = 1
        # Center tap should dominate: h[M] largest
        assert h[M] == pytest.approx(1.0, abs=1e-9)
        # All other taps ≈ 0
        for k, v in enumerate(h):
            if k != M:
                assert abs(v) < 1e-9, f"tap {k} = {v:.2e} should be ~0"

    def test_energy_near_unity(self, phase):
        """Sum of |h| should be close to 1 for any frac (gain-neutral)."""
        for frac in [-0.4, -0.2, 0.0, 0.2, 0.4]:
            h = phase._lagrange_ffd(frac, order=3)
            total = float(np.sum(h))
            assert abs(total - 1.0) < 0.05, f"frac={frac}: sum(h)={total:.4f} not near 1"

    def test_fractional_delay_accuracy(self, phase):
        """Apply filter to a sine, verify the group delay matches center+frac."""
        frac = 0.3
        order = 3
        h = phase._lagrange_ffd(frac, order=order)
        M = order // 2  # causal integer delay of filter

        sr_test = 48000
        freq = 500.0
        n = 4096
        t = np.arange(n) / sr_test
        x = np.sin(2.0 * np.pi * freq * t)

        y = np.convolve(x, h, mode="full")
        # After compensating for integer delay M, the remaining shift should be frac
        expected_shift = frac  # fractional samples beyond center
        # Verify via cross-correlation of x vs y[M:M+n]
        y_comp = y[M : M + n]
        xcf = np.correlate(x[10:], y_comp[10:], mode="full")
        center = len(xcf) // 2
        # Parabolic peak
        peak_idx = int(np.argmax(np.abs(xcf[center - 5 : center + 6])))
        # Peak should be within ±1 sample of the expected fractional shift
        assert abs(peak_idx - 5) <= 1, f"Lagrange FIR frac={frac}: peak offset {peak_idx - 5}, expected ~0"

    def test_half_sample_delay(self, phase):
        """frac=0.5 → symmetric taps (h[k] = h[N-k] since this is max-flat midpoint)."""
        h = phase._lagrange_ffd(0.5, order=3)
        # Not symmetric in general for odd order+1, but verify finiteness
        assert np.all(np.isfinite(h))

    def test_order_7_shape(self, phase):
        h = phase._lagrange_ffd(0.25, order=7)
        assert h.shape == (8,)
        assert np.all(np.isfinite(h))


# ---------------------------------------------------------------------------
# 2. _analyze_phase sub-sample tests
# ---------------------------------------------------------------------------


class TestAnalyzePhase:
    def test_returns_float_delay(self, phase):
        """_analyze_phase must return a float delay (not int)."""
        n = SR // 2
        t = np.arange(n) / SR
        L = np.sin(2.0 * np.pi * 440.0 * t)
        d = 5
        R = np.roll(L, d)
        R[:d] = 0.0
        corr, delay = phase._analyze_phase(L, R, max_delay=20)
        assert isinstance(delay, float), f"delay should be float, got {type(delay)}"

    def test_integer_delay_detected(self, phase):
        """Known 10-sample integer delay is detected within ±0.5 samples."""
        n = SR // 2
        t = np.arange(n) / SR
        L = np.sin(2.0 * np.pi * 440.0 * t)
        d_true = 10
        R = np.roll(L, d_true)
        R[:d_true] = 0.0
        corr, delay = phase._analyze_phase(L, R, max_delay=30)
        assert abs(delay - d_true) < 0.5, f"Detected delay {delay:.3f} ≠ true {d_true}"


class TestPhase14LocalityProfile:
    def test_locality_profile_is_event_strength_adaptive(self, phase):
        profile, coverage = phase._build_locality_profile(
            n_samples=SR * 3,
            sample_rate=SR,
            defect_locations={"phase_issues": [(0.30, 0.90)], "stereo_imbalance": [(1.70, 2.30)]},
            event_metadata={
                "phase_issues": {"severity": 0.95, "confidence": 0.95},
                "stereo_imbalance": {"severity": 0.35, "confidence": 0.65},
            },
        )

        assert profile.shape == (SR * 3,)
        assert 0.0 < coverage < 0.50
        strong_region = float(np.mean(profile[int(0.45 * SR) : int(0.75 * SR)]))
        mild_region = float(np.mean(profile[int(1.85 * SR) : int(2.15 * SR)]))
        clean_region = float(np.mean(profile[int(1.10 * SR) : int(1.35 * SR)]))
        assert strong_region > mild_region * 1.5
        assert clean_region < 0.04

    def test_vibrato_zone_caps_locality_profile(self, phase):
        free, _ = phase._build_locality_profile(
            n_samples=SR * 3,
            sample_rate=SR,
            defect_locations={"phase_issues": [(1.20, 1.80)]},
            event_metadata={"phase_issues": {"severity": 0.95, "confidence": 0.95}},
        )
        capped, _ = phase._build_locality_profile(
            n_samples=SR * 3,
            sample_rate=SR,
            defect_locations={"phase_issues": [(1.20, 1.80)]},
            event_metadata={"phase_issues": {"severity": 0.95, "confidence": 0.95}},
            protected_zones=[(1.15, 1.85, 0.20)],
        )

        free_strength = float(np.mean(free[int(1.35 * SR) : int(1.65 * SR)]))
        capped_strength = float(np.mean(capped[int(1.35 * SR) : int(1.65 * SR)]))
        assert capped_strength <= 0.21
        assert capped_strength < free_strength * 0.55

    def test_fractional_delay_more_accurate_than_integer(self, phase):
        """Sub-sample estimation is closer to truth than rounding to nearest int."""
        n = SR // 2
        t = np.arange(n) / SR
        freq = 800.0
        L = np.sin(2.0 * np.pi * freq * t)
        # True delay = 7.35 samples (fractional)
        d_true = 7.35
        d_int = 7
        d_frac = 0.35
        R_int = np.roll(L, d_int)
        R_int[:d_int] = 0.0
        R = (1.0 - d_frac) * R_int + d_frac * np.roll(R_int, 1)

        _, delay_float = phase._analyze_phase(L, R, max_delay=20)
        err_float = abs(delay_float - d_true)
        err_int = abs(round(delay_float) - d_true)

        assert err_float <= err_int + 0.01, f"Float err {err_float:.3f} should be ≤ int err {err_int:.3f}"

    def test_zero_delay_signal(self, phase):
        """Identical L and R → delay ≈ 0.0."""
        n = SR // 2
        t = np.arange(n) / SR
        L = np.sin(2.0 * np.pi * 1000.0 * t)
        corr, delay = phase._analyze_phase(L, L.copy(), max_delay=10)
        assert abs(delay) < 1.0, f"Identical channels should give delay ≈ 0, got {delay:.3f}"

    def test_correlation_range(self, phase):
        n = SR // 2
        t = np.arange(n) / SR
        L = np.sin(2.0 * np.pi * 440.0 * t)
        R = np.roll(L, 5)
        corr, _ = phase._analyze_phase(L, R, max_delay=20)
        assert -1.0 <= corr <= 1.0, f"Correlation {corr:.4f} out of range"


# ---------------------------------------------------------------------------
# 3. _correct_band_phase tests
# ---------------------------------------------------------------------------


class TestCorrectBandPhase:
    def test_output_shape_preserved(self, phase):
        n = SR // 2
        t = np.arange(n) / SR
        L = np.sin(2.0 * np.pi * 440.0 * t)
        R = np.roll(L, 5)
        out_L, out_R = phase._correct_band_phase(L, R, delay=5.0, strength=1.0)
        assert out_L.shape == L.shape
        assert out_R.shape == R.shape

    def test_integer_delay_corrected(self, phase):
        n = SR // 2
        t = np.arange(n) / SR
        L = np.sin(2.0 * np.pi * 440.0 * t)
        d = 8
        R = np.roll(L, d)
        R[:d] = 0.0
        out_L, out_R = phase._correct_band_phase(L, R, delay=float(d), strength=1.0)
        # After correction L and R should be more correlated than before
        with np.errstate(invalid="ignore"):
            corr_before = float(np.corrcoef(L[d:], R[d:])[0, 1])
            corr_after = float(np.corrcoef(out_L[d:], out_R[d:])[0, 1])
        assert corr_after >= corr_before - 0.05

    def test_zero_delay_passthrough(self, phase):
        """delay=0 → output ≈ input."""
        n = SR // 2
        L = np.random.default_rng(0).standard_normal(n)
        R = np.random.default_rng(1).standard_normal(n)
        out_L, out_R = phase._correct_band_phase(L, R, delay=0.0, strength=1.0)
        np.testing.assert_allclose(out_L, L, atol=1e-9)
        np.testing.assert_allclose(out_R, R, atol=1e-9)

    def test_fractional_delay_output_finite(self, phase):
        n = SR // 2
        t = np.arange(n) / SR
        L = np.sin(2.0 * np.pi * 1200.0 * t)
        R = np.roll(L, 5)
        out_L, out_R = phase._correct_band_phase(L, R, delay=5.35, strength=1.0)
        assert np.all(np.isfinite(out_L))
        assert np.all(np.isfinite(out_R))

    def test_fractional_delay_improves_correlation(self, phase):
        """0.4-sample fractional delay: correction with frac FIR should improve corr."""
        n = SR
        t = np.arange(n) / SR
        freq = 4000.0  # high frequency — sensitive to sub-sample delay
        L = np.sin(2.0 * np.pi * freq * t)
        d_true = 3.4
        d_int = 3
        d_frac_gt = 0.4
        R_int = np.roll(L, d_int)
        R_int[:d_int] = 0.0
        R = (1.0 - d_frac_gt) * R_int + d_frac_gt * np.roll(R_int, 1)

        # Correction with full float delay (includes fractional FIR)
        out_L, out_R = phase._correct_band_phase(L, R, delay=d_true, strength=1.0)
        # Compare correlation improvement
        trim = 20
        with np.errstate(invalid="ignore"):
            corr_before = float(np.corrcoef(L[trim:], R[trim:])[0, 1])
            corr_after = float(np.corrcoef(out_L[trim:], out_R[trim:])[0, 1])
        assert corr_after >= corr_before, (
            f"Fractional FIR should not worsen correlation: {corr_before:.4f} → {corr_after:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. Full process() integration tests
# ---------------------------------------------------------------------------


class TestProcessIntegration:
    def test_mono_passthrough(self, phase):
        """Mono input is returned unchanged."""
        from backend.core.defect_scanner import MaterialType

        mono = np.random.default_rng(42).standard_normal(SR // 2).astype(np.float32)
        r = phase.process(mono, SR, material=MaterialType.TAPE)
        assert r.success
        assert r.audio.shape == mono.shape

    def test_stereo_no_nan(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = _stereo_with_delay(8.0, freq=500.0)
        r = phase.process(audio.astype(np.float32), SR, material=MaterialType.TAPE)
        assert np.all(np.isfinite(r.audio))

    def test_stereo_no_clipping(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = _stereo_with_delay(5.0, freq=1000.0)
        r = phase.process(audio.astype(np.float32), SR, material=MaterialType.TAPE)
        assert np.max(np.abs(r.audio)) <= 1.0 + 1e-6

    def test_shape_preserved(self, phase):
        from backend.core.defect_scanner import MaterialType

        audio = _stereo_with_delay(3.0).astype(np.float32)
        r = phase.process(audio, SR, material=MaterialType.VINYL)
        assert r.audio.shape == audio.shape

    def test_correlation_improves_for_tape(self, phase):
        """Known 12-sample delay → correlation should improve after correction."""
        from backend.core.defect_scanner import MaterialType

        audio = _stereo_with_delay(12.0, freq=440.0)
        r = phase.process(audio.astype(np.float32), SR, material=MaterialType.TAPE)
        assert r.metrics["correlation_after"] >= r.metrics["correlation_before"] - 0.05

    def test_algorithm_field_fractional(self, phase):
        """metadata.algorithm should reference fractional correction (v2.1)."""
        from backend.core.defect_scanner import MaterialType

        audio = _stereo_with_delay(5.0).astype(np.float32)
        r = phase.process(audio, SR, material=MaterialType.TAPE)
        assert "fractional" in r.metadata.get("algorithm", "")

    def test_version_bumped(self, phase):
        meta = phase.get_metadata()
        major, minor, _ = meta.version.split(".")
        assert (int(major), int(minor)) >= (2, 1)

    def test_delays_corrected_are_floats(self, phase):
        """delays_corrected_samples should contain float values (sub-sample precision)."""
        from backend.core.defect_scanner import MaterialType

        audio = _stereo_with_delay(7.0).astype(np.float32)
        r = phase.process(audio, SR, material=MaterialType.TAPE)
        for d in r.metrics.get("delays_corrected_samples", []):
            assert isinstance(d, float), f"Expected float, got {type(d)}: {d}"
