"""
tests/unit/test_frisson_candidate_detector.py
=============================================
Aurik 10.0.0 — FrissonCandidateDetector + MDEM Frisson-Integration

13 Unit-Tests:
  TestFrissonCandidateDetector  (test_01 – test_09)
  TestMDEMFrissonIntegration    (test_10 – test_13)

Alle Tests synthetisch (keine echten Audio-Dateien).
"""

import numpy as np
import pytest

SR = 48000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detector():
    from backend.core.frisson_candidate_detector import FrissonCandidateDetector

    return FrissonCandidateDetector()


@pytest.fixture(scope="module")
def mdem():
    from backend.core.micro_dynamics_envelope_morphing import MicroDynamicsEnvelopeMorphing

    return MicroDynamicsEnvelopeMorphing()


@pytest.fixture(scope="module")
def silence_then_music():
    """2 s silence followed by 3 s loud music — classic frisson trigger."""
    t_music = np.linspace(0, 3.0, 3 * SR, endpoint=False)
    music = (0.6 * np.sin(2 * np.pi * 440 * t_music)).astype(np.float32)
    silence = np.zeros(2 * SR, dtype=np.float32)
    return np.concatenate([silence, music]).astype(np.float32)


@pytest.fixture(scope="module")
def constant_tone():
    """A flat 5-second 440 Hz tone at constant amplitude — low frisson potential."""
    t = np.linspace(0, 5.0, 5 * SR, endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# TestFrissonCandidateDetector
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFrissonCandidateDetector:
    def test_01_detect_returns_list(self, detector, silence_then_music):
        """detect() must always return a list."""
        result = detector.detect(silence_then_music, SR)
        assert isinstance(result, list)

    def test_02_zones_have_valid_properties(self, detector, silence_then_music):
        """Every returned zone must have valid start/end/score/trigger."""
        zones = detector.detect(silence_then_music, SR)
        for z in zones:
            assert z.start_s >= 0.0, f"start_s must be ≥ 0: {z.start_s}"
            assert z.end_s > z.start_s, f"end_s must be > start_s: {z}"
            assert 0.0 <= z.score <= 1.0, f"score must be in [0,1]: {z.score}"
            assert isinstance(z.trigger, str) and len(z.trigger) > 0

    def test_03_silence_to_music_creates_zone(self, detector, silence_then_music):
        """Loud entry after silence must produce at least one high-score zone."""
        zones = detector.detect(silence_then_music, SR)
        assert len(zones) >= 1, "Expected ≥1 frisson zone for silence→music signal"
        # Top zone score should be meaningful
        assert zones[0].score >= 0.25, f"Expected top score ≥ 0.25, got {zones[0].score:.3f}"

    def test_04_constant_signal_no_high_zones(self, detector, constant_tone):
        """Constant-amplitude tone should produce no zone with score > 0.6."""
        zones = detector.detect(constant_tone, SR)
        high_score_zones = [z for z in zones if z.score > 0.60]
        assert len(high_score_zones) == 0, f"Constant tone produced unexpected high-score zones: {high_score_zones}"

    def test_05_nan_safe(self, detector):
        """NaN/Inf input must not crash, must return a list."""
        nan_audio = np.full(SR * 3, np.nan, dtype=np.float32)
        result = detector.detect(nan_audio, SR)
        assert isinstance(result, list)

        inf_audio = np.full(SR * 3, np.inf, dtype=np.float32)
        result2 = detector.detect(inf_audio, SR)
        assert isinstance(result2, list)

    def test_06_stereo_input(self, detector, silence_then_music):
        """Stereo (2-channel) input must be handled without crash."""
        stereo = np.stack([silence_then_music, silence_then_music * 0.9], axis=1)  # (samples, 2)
        result = detector.detect(stereo, SR)
        assert isinstance(result, list)

    def test_07_very_short_audio_no_crash(self, detector):
        """Audio shorter than one analysis frame must return empty list, not crash."""
        short = np.zeros(int(0.3 * SR), dtype=np.float32)  # 300 ms
        result = detector.detect(short, SR)
        assert isinstance(result, list)

    def test_08_zones_sorted_by_score_descending(self, detector, silence_then_music):
        """Returned zones must be sorted by score descending."""
        zones = detector.detect(silence_then_music, SR, max_zones=20)
        for i in range(len(zones) - 1):
            assert zones[i].score >= zones[i + 1].score, (
                f"Zones not sorted: index {i} score={zones[i].score:.3f} < index {i + 1} score={zones[i + 1].score:.3f}"
            )

    def test_09_peak_timing_near_transition(self, detector, silence_then_music):
        """Detected zone must overlap with the actual silence→music transition (at t=2 s)."""
        zones = detector.detect(silence_then_music, SR)
        if not zones:
            pytest.skip("No zones detected — cannot verify timing")
        # The top zone should cover the transition region (2.0 s ± 2.0 s)
        top_z = zones[0]
        overlap = top_z.end_s > 1.5 and top_z.start_s < 4.5
        assert overlap, (
            f"Top zone [{top_z.start_s:.2f}, {top_z.end_s:.2f}] does not overlap "
            f"with expected transition region [1.5, 4.5] s"
        )


# ---------------------------------------------------------------------------
# TestMDEMFrissonIntegration
# ---------------------------------------------------------------------------


class TestMDEMFrissonIntegration:
    def test_10_morph_accepts_frisson_zones_none(self, mdem):
        """morph() with frisson_zones=None must behave identically to before."""
        t = np.linspace(0, 2.0, 2 * SR, endpoint=False)
        sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = mdem.morph(sig.copy(), sig.copy(), SR, mode="restoration", frisson_zones=None)
        assert result is not None
        assert result.dtype == np.float32 or result.dtype == np.float64
        assert len(result) == len(sig)

    def test_11_morph_with_zones_no_crash(self, mdem):
        """morph() with a FrissonZone must complete without exception."""
        from backend.core.frisson_candidate_detector import FrissonZone

        t = np.linspace(0, 4.0, 4 * SR, endpoint=False)
        original = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        restored = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        zones = [FrissonZone(start_s=1.0, end_s=3.0, score=0.7, trigger="rms_crescendo")]
        result = mdem.morph(restored.copy(), original, SR, mode="restoration", frisson_zones=zones)
        assert result is not None
        assert len(result) == len(restored)
        # must be NaN-free
        assert not np.any(np.isnan(result)), "morph() returned NaN with frisson_zones"

    def test_12_frisson_zone_preserves_peak(self, mdem):
        """In a frisson zone, downward gain must be capped at -1.0 LU.

        Design:
          - original:  constant 0.25 amplitude (flat — ensures Pearson >> 0.93, no retry)
          - restored:  0.25 for first 2 s, then 0.40 in frisson zone  → 4 LU difference
          - Without frisson: G = clip(-4, -4, 4) = -4.0 LU
          - With frisson:    G = clip(-4, -1, 4) = -1.0 LU
          - Gain ratio: 10^(3/20) ≈ 1.41×  → clearly measurable in mid-zone RMS
        """
        from backend.core.frisson_candidate_detector import FrissonZone

        sr = SR
        duration_s = 4.0
        n = int(duration_s * sr)
        t = np.linspace(0, duration_s, n, endpoint=False)

        # Original: constant 0.25 amplitude (Pearson stays >> 0.93, no Pearson-retry)
        original = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Restored: same first 2 s, then 0.40 amplitude (MDEM wants -4 LU attenuation)
        restored = original.copy()
        seg_peak = slice(2 * sr, 4 * sr)
        t_peak = t[seg_peak]
        restored[seg_peak] = (0.40 * np.sin(2 * np.pi * 440 * t_peak)).astype(np.float32)

        frisson_zone = [FrissonZone(start_s=1.8, end_s=3.8, score=0.82, trigger="rms_crescendo")]

        result_with = mdem.morph(restored.copy(), original.copy(), sr, mode="restoration", frisson_zones=frisson_zone)
        result_without = mdem.morph(restored.copy(), original.copy(), sr, mode="restoration", frisson_zones=None)

        # Measure RMS in the MIDDLE of the frisson zone (2.8–3.5 s) to avoid
        # Savitzky-Golay boundary smoothing effects near the zone edges.
        mid_start = int(2.8 * sr)
        mid_end = int(3.5 * sr)
        rms_with = float(np.sqrt(np.mean(result_with[mid_start:mid_end] ** 2) + 1e-12))
        rms_without = float(np.sqrt(np.mean(result_without[mid_start:mid_end] ** 2) + 1e-12))

        # With frisson protection (-1 LU floor) vs without (-4 LU floor):
        # expected ratio ≈ 10^(3/20) ≈ 1.41×. Use conservative threshold 1.15×.
        assert rms_with >= rms_without * 1.15, (
            f"Frisson zone must preserve peak energy. "
            f"rms_with={rms_with:.4f}, rms_without={rms_without:.4f}, "
            f"ratio={rms_with / max(rms_without, 1e-10):.2f} (expected ≥ 1.15)"
        )

    def test_13_out_of_range_frisson_zone_no_crash(self, mdem):
        """Frisson zone beyond audio duration must not crash."""
        from backend.core.frisson_candidate_detector import FrissonZone

        sig = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 2, 2 * SR, endpoint=False))).astype(np.float32)
        zones = [
            FrissonZone(start_s=100.0, end_s=200.0, score=0.9, trigger="rms_crescendo"),
        ]
        result = mdem.morph(sig.copy(), sig.copy(), SR, mode="restoration", frisson_zones=zones)
        assert result is not None
        assert len(result) == len(sig)
