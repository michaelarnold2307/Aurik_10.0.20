"""v10.17 HHCG + MQA + Groove Tests (fixed)
===========================================
Run: python3 -m pytest backend/tests/test_bugfixes.py -v
"""

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════
# Naturalness Self-Calibration
# ═══════════════════════════════════════════════════════════════════


class TestNaturalnessSelfCalibration:
    """Naturalness calibrates threshold from reference audio."""

    def test_without_reference(self):
        from backend.core.quality_prediction import QualityAnalyzer

        qa = QualityAnalyzer()
        sr = 48000
        t = np.arange(sr * 2) / sr
        audio = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(sr * 2)).astype(np.float32)
        est = qa.analyze_quality(audio, sr)
        assert 0.0 <= est.naturalness <= 1.0

    def test_with_reference_not_worse(self):
        """Self-calibration: reference must not reduce score vs no-ref."""
        from backend.core.quality_prediction import QualityAnalyzer

        qa = QualityAnalyzer()
        sr = 48000
        t = np.arange(sr * 2) / sr
        degraded = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(sr * 2)).astype(np.float32)
        restored = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.randn(sr * 2)).astype(np.float32)
        est1 = qa.analyze_quality(restored, sr)
        est2 = qa.analyze_quality(restored, sr, reference=degraded)
        assert est2.naturalness >= est1.naturalness, f"{est1.naturalness:.3f} >= {est2.naturalness:.3f}?"

    def test_silence_default(self):
        from backend.core.quality_prediction import QualityAnalyzer

        qa = QualityAnalyzer()
        est = qa.analyze_quality(np.zeros(48000, dtype=np.float32), 48000)
        assert est.naturalness == 0.75


# ═══════════════════════════════════════════════════════════════════
# MQA Minimum Improvement
# ═══════════════════════════════════════════════════════════════════


class TestMQAMinimumImprovement:
    """MQA requires improvement for quality-guaranteed."""

    def test_improvement_computed(self):
        """musical_improvement must be computed and source has threshold check."""
        from backend.core.musical_quality_assurance import (
            MediumType,
            MusicalQualityAssurance,
            ProcessingMode,
        )

        mqa = MusicalQualityAssurance()
        sr = 48000
        t = np.arange(sr * 2) / sr
        degraded = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.random.randn(len(t))).astype(np.float32)
        clean = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.03 * np.random.randn(len(t))).astype(np.float32)
        report = mqa.validate_final_quality(
            degraded,
            clean,
            sr,
            medium_type=MediumType.UNKNOWN,
            processing_mode=ProcessingMode.RESTORATION,
            modules_applied=["test"],
        )
        assert isinstance(report.musical_improvement, float)
        assert -1.0 <= report.musical_improvement <= 1.0, (
            f"improvement={report.musical_improvement * 100:.1f}% out of range"
        )

    def test_source_has_minimal_improvement(self):
        """Source code must check _minimal_improvement."""
        import inspect

        from backend.core.musical_quality_assurance import MusicalQualityAssurance

        source = inspect.getsource(MusicalQualityAssurance.validate_final_quality)
        assert "_minimal_improvement" in source


# ═══════════════════════════════════════════════════════════════════
# GrooveMetric Onset-Guard
# ═══════════════════════════════════════════════════════════════════


class TestGrooveOnsetGuard:
    """Onset-Guard with DTW-rms threshold."""

    def test_source_has_dtw_rms_check(self):
        """Source must check dtw_rms_ms before guard fires."""
        import inspect

        from backend.core.musical_goals.musical_goals_metrics import GrooveMetric

        source = inspect.getsource(GrooveMetric._measure_with_dtw)
        assert "dtw_rms_ok" in source or "dtw_rms_ms" in source, "Guard needs DTW-rms check"
        assert "0.95" in source, "Threshold must be 0.95"
        assert "0.75" in source or "0.60" in source, "Score cap must be < 1.0"


# ═══════════════════════════════════════════════════════════════════
# HumanHearingComfortGuard
# ═══════════════════════════════════════════════════════════════════


class TestComfortGuard:
    """HHCG overshoot margin."""

    def test_margin_is_2_5(self):
        """Source must have envelope_margin_db = 2.50."""
        import inspect

        from backend.core.dsp.human_hearing_comfort_guard import _attenuate_peak_overshoot

        source = inspect.getsource(_attenuate_peak_overshoot)
        assert "envelope_margin_db = 2.50" in source, "Must be 2.50 (was 1.25)"

    def test_no_overshoot_passthrough(self):
        """Identical audio → no correction needed."""
        from backend.core.dsp.human_hearing_comfort_guard import apply_human_hearing_comfort_guard

        sr = 48000
        t = np.arange(sr * 2) / sr
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = apply_human_hearing_comfort_guard(audio, audio, sr)
        assert result.max_peak_overshoot_db <= 3.0
        assert result.peak_overshoot_frames == 0

    def test_loud_candidate_attenuated(self):
        """Loud candidate must be attenuated."""
        from backend.core.dsp.human_hearing_comfort_guard import apply_human_hearing_comfort_guard

        sr = 48000
        t = np.arange(sr * 2) / sr
        ref = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        loud = (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # +6dB
        result = apply_human_hearing_comfort_guard(ref, loud, sr)
        # At least some overshoot detected
        assert result.max_peak_overshoot_db > 0.5, f"Expected overshoot >0.5dB, got {result.max_peak_overshoot_db:.2f}"
        # Result RMS should be closer to ref than original loud was
        out_rms = float(np.sqrt(np.mean(result.audio**2)))
        loud_rms = float(np.sqrt(np.mean(loud**2)))
        ref_rms = float(np.sqrt(np.mean(ref**2)))
        assert abs(out_rms - ref_rms) <= abs(loud_rms - ref_rms), "Correction should reduce RMS gap"


# ═══════════════════════════════════════════════════════════════════
# Mono Compatibility Log
# ═══════════════════════════════════════════════════════════════════


class TestMonoLog:
    """Mono log format uses %.2f."""

    def test_log_format(self):
        with open("backend/core/dsp/stereo_guard.py") as f:
            src = f.read()
        assert "Phasenlöschung=%.2f" in src, "Log must use %.2f (was %.1f)"
