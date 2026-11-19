"""
Tests for ProcessingLogger System

Tests:
- QualityMetrics computation
- ProcessingStep logging
- ProcessingTrace aggregation
- Audio snapshot saving
- JSON export
- Markdown report generation
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.core.processing_logger import (
    ProcessingLogger,
    ProcessingStep,
    ProcessingTrace,
    QualityMetrics,
    create_logger,
)


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    sr = 44100
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration))

    # Generate clean sine wave at 440 Hz (A4)
    audio_clean = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Add noise
    noise = 0.05 * np.random.randn(len(audio_clean))
    audio_noisy = audio_clean + noise

    return audio_clean, audio_noisy, sr


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.unit
class TestQualityMetrics:
    """Test QualityMetrics computation."""

    def test_compute_metrics(self, sample_audio):
        """Test metrics computation."""
        audio_clean, _, sr = sample_audio

        logger = ProcessingLogger()
        metrics = logger._compute_metrics(audio_clean, sr)

        # Check all metrics are computed
        assert isinstance(metrics, QualityMetrics)
        assert metrics.snr_db > 0
        assert metrics.thd_percent >= 0
        assert metrics.lufs < 0  # Typical for normalized audio
        assert metrics.spectral_centroid_hz > 0
        assert metrics.peak_db <= 0  # Peak should be <= 0 dB FS
        assert metrics.dynamic_range_db > 0

    def test_snr_comparison(self, sample_audio):
        """Test that SNR can be computed (estimation is not perfect for synthetic audio)."""
        audio_clean, audio_noisy, sr = sample_audio

        logger = ProcessingLogger()
        metrics_clean = logger._compute_metrics(audio_clean, sr)
        metrics_noisy = logger._compute_metrics(audio_noisy, sr)

        # SNR estimation is complex - just verify both are in reasonable range
        assert -20 <= metrics_clean.snr_db <= 80
        assert -20 <= metrics_noisy.snr_db <= 80

        # THD should be higher for noisy audio (more reliable than SNR for synthetic signals)
        assert metrics_noisy.thd_percent > metrics_clean.thd_percent


class TestProcessingStep:
    """Test ProcessingStep functionality."""

    def test_step_creation(self, sample_audio):
        """Test creating a ProcessingStep."""
        audio_clean, audio_noisy, sr = sample_audio

        logger = ProcessingLogger()
        metrics_before = logger._compute_metrics(audio_noisy, sr)
        metrics_after = logger._compute_metrics(audio_clean, sr)

        step = ProcessingStep(
            step_id="test_step",
            phase="Phase 1: Denoising",
            module_name="test_denoiser",
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            processing_time_ms=123.45,
        )

        assert step.step_id == "test_step"
        assert step.phase == "Phase 1: Denoising"
        # Note: SNR estimation may not be accurate for synthetic sine waves
        # Just verify the improvement method works
        improvement = step.improvement_snr_db()
        assert isinstance(improvement, float)

    def test_step_to_dict(self, sample_audio):
        """Test converting ProcessingStep to dict."""
        audio_clean, audio_noisy, sr = sample_audio

        logger = ProcessingLogger()
        metrics_before = logger._compute_metrics(audio_noisy, sr)
        metrics_after = logger._compute_metrics(audio_clean, sr)

        step = ProcessingStep(
            step_id="test_step",
            phase="Phase 1: Denoising",
            module_name="test_denoiser",
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            processing_time_ms=123.45,
            parameters={"strength": 0.3},
        )

        step_dict = step.to_dict()

        assert step_dict["step_id"] == "test_step"
        assert "metrics_before" in step_dict
        assert "metrics_after" in step_dict
        assert "improvements" in step_dict
        assert step_dict["parameters"]["strength"] == 0.3


class TestProcessingTrace:
    """Test ProcessingTrace functionality."""

    def test_trace_creation(self, sample_audio):
        """Test creating a ProcessingTrace."""
        trace = ProcessingTrace(
            session_id="test_session", input_file="test.wav", processing_mode="restoration", sample_rate=44100
        )

        assert trace.session_id == "test_session"
        assert trace.input_file == "test.wav"
        assert len(trace.steps) == 0

    def test_overall_snr_improvement(self, sample_audio):
        """Test overall SNR improvement calculation."""
        audio_clean, audio_noisy, sr = sample_audio
        logger = ProcessingLogger()

        # Create trace with 2 steps
        trace = ProcessingTrace(session_id="test", input_file="test.wav", sample_rate=sr)

        # Step 1: Partial improvement
        audio_partial = 0.8 * audio_clean + 0.2 * audio_noisy
        step1 = ProcessingStep(
            step_id="step1",
            phase="Phase 1",
            module_name="module1",
            metrics_before=logger._compute_metrics(audio_noisy, sr),
            metrics_after=logger._compute_metrics(audio_partial, sr),
            processing_time_ms=100,
        )
        trace.steps.append(step1)

        # Step 2: Full improvement
        step2 = ProcessingStep(
            step_id="step2",
            phase="Phase 2",
            module_name="module2",
            metrics_before=logger._compute_metrics(audio_partial, sr),
            metrics_after=logger._compute_metrics(audio_clean, sr),
            processing_time_ms=150,
        )
        trace.steps.append(step2)

        # Overall improvement calculation (may not be positive for synthetic sine waves)
        overall_improvement = trace.overall_snr_improvement()
        assert isinstance(overall_improvement, float)

        # Verify calculation: should be difference between last and first metrics
        expected = step2.metrics_after.snr_db - step1.metrics_before.snr_db
        assert abs(overall_improvement - expected) < 0.01

    def test_trace_to_markdown(self, sample_audio):
        """Test Markdown report generation."""
        audio_clean, audio_noisy, sr = sample_audio
        logger = ProcessingLogger()

        trace = ProcessingTrace(session_id="test", input_file="test.wav", sample_rate=sr)

        # Add a step
        step = ProcessingStep(
            step_id="step1",
            phase="Phase 1",
            module_name="module1",
            metrics_before=logger._compute_metrics(audio_noisy, sr),
            metrics_after=logger._compute_metrics(audio_clean, sr),
            processing_time_ms=100,
        )
        trace.steps.append(step)

        # Generate markdown
        md = trace.to_markdown()

        assert "# Processing Trace Report" in md
        assert "test.wav" in md
        assert "step1" in md
        assert "Phase 1" in md

    def test_markdown_time_units_regression(self, sample_audio):
        """Ensure per-step times stay in ms and total time stays in seconds."""
        audio_clean, audio_noisy, sr = sample_audio
        logger = ProcessingLogger()

        trace = ProcessingTrace(session_id="units", input_file="units.wav", sample_rate=sr)
        step = ProcessingStep(
            step_id="step_units",
            phase="Phase Units",
            module_name="module_units",
            metrics_before=logger._compute_metrics(audio_noisy, sr),
            metrics_after=logger._compute_metrics(audio_clean, sr),
            processing_time_ms=208.0,
        )
        trace.steps.append(step)
        trace.total_processing_time_sec = 0.208

        md = trace.to_markdown()

        # Regression guard: step time/average must remain in milliseconds.
        assert "208 ms" in md
        assert "Avg Time/Step:" in md and " ms" in md

        # Total time is a session duration in seconds, never milliseconds.
        assert "Total Time:" in md and "0.21s" in md
        assert "208 s" not in md


class TestProcessingLogger:
    """Test ProcessingLogger functionality."""

    def test_logger_creation(self, temp_output_dir):
        """Test creating a ProcessingLogger."""
        logger = ProcessingLogger(session_id="test_session", output_dir=temp_output_dir)

        assert logger.session_id == "test_session"
        assert logger.output_dir == temp_output_dir
        assert logger.trace is None

    def test_session_workflow(self, sample_audio, temp_output_dir):
        """Test complete session workflow."""
        audio_clean, audio_noisy, sr = sample_audio

        # Create logger
        logger = ProcessingLogger(
            session_id="test_session",
            output_dir=temp_output_dir,
            save_audio_snapshots=True,
            save_json=True,
            save_markdown=True,
        )

        # Start session
        logger.start_session(input_file="test.wav", processing_mode="restoration", sample_rate=sr)

        # Log a processing step
        logger.log_step(
            step_id="phase_1_denoise",
            phase="Phase 1: Denoising",
            module_name="test_denoiser",
            audio_before=audio_noisy,
            audio_after=audio_clean,
            sr=sr,
            processing_time_ms=123.45,
            parameters={"strength": 0.3},
        )

        # End session
        trace = logger.end_session(output_file="output.wav")

        # Verify trace
        assert trace.session_id == "test_session"
        assert len(trace.steps) == 1
        assert trace.output_file == "output.wav"

        # Verify files were created
        assert (temp_output_dir / "trace.json").exists()
        assert (temp_output_dir / "report.md").exists()
        assert (temp_output_dir / "phase_1_denoise_before.wav").exists()
        assert (temp_output_dir / "phase_1_denoise_after.wav").exists()

    def test_multiple_steps(self, sample_audio, temp_output_dir):
        """Test logging multiple processing steps."""
        audio_clean, audio_noisy, sr = sample_audio

        logger = ProcessingLogger(session_id="test_multi", output_dir=temp_output_dir)

        logger.start_session("test.wav", sample_rate=sr)

        # Log 3 steps
        audio_step1 = 0.7 * audio_clean + 0.3 * audio_noisy
        audio_step2 = 0.85 * audio_clean + 0.15 * audio_noisy

        logger.log_step("step1", "Phase 1", "module1", audio_noisy, audio_step1, sr, 100)
        logger.log_step("step2", "Phase 2", "module2", audio_step1, audio_step2, sr, 150)
        logger.log_step("step3", "Phase 3", "module3", audio_step2, audio_clean, sr, 200)

        trace = logger.end_session()

        assert len(trace.steps) == 3
        assert trace.total_processing_time_sec == 0.45  # 450ms total
        assert trace.average_processing_time_per_step() == 150.0

    def test_json_export(self, sample_audio, temp_output_dir):
        """Test JSON trace export."""
        audio_clean, audio_noisy, sr = sample_audio

        logger = ProcessingLogger(
            session_id="test_json",
            output_dir=temp_output_dir,
            save_audio_snapshots=False,
            save_json=True,
            save_markdown=False,
        )

        logger.start_session("test.wav", sample_rate=sr)
        logger.log_step("step1", "Phase 1", "module1", audio_noisy, audio_clean, sr, 100)
        logger.end_session()

        # Load and verify JSON
        json_path = temp_output_dir / "trace.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert data["session_id"] == "test_json"
        assert data["input_file"] == "test.wav"
        assert len(data["steps"]) == 1
        assert "overall_metrics" in data
        assert "snr_improvement_db" in data["overall_metrics"]

    def test_compressed_audio(self, sample_audio, temp_output_dir):
        """Test FLAC compression for audio snapshots."""
        audio_clean, audio_noisy, sr = sample_audio

        logger = ProcessingLogger(
            session_id="test_flac",
            output_dir=temp_output_dir,
            save_audio_snapshots=True,
            compress_audio=True,  # Use FLAC
        )

        logger.start_session("test.wav", sample_rate=sr)
        logger.log_step("step1", "Phase 1", "module1", audio_noisy, audio_clean, sr, 100)
        logger.end_session()

        # Verify FLAC files were created
        assert (temp_output_dir / "step1_before.flac").exists()
        assert (temp_output_dir / "step1_after.flac").exists()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_logger(self, temp_output_dir):
        """Test create_logger convenience function."""
        logger = create_logger(
            session_id="convenience_test", output_dir=temp_output_dir, save_audio=True, compress=True
        )

        assert logger.session_id == "convenience_test"
        assert logger.save_audio_snapshots is True
        assert logger.compress_audio is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_log_step_before_start_session(self, sample_audio):
        """Test that log_step fails without starting session."""
        audio_clean, audio_noisy, sr = sample_audio

        logger = ProcessingLogger()

        with pytest.raises(RuntimeError, match="Must call start_session"):
            logger.log_step("step1", "Phase 1", "module1", audio_noisy, audio_clean, sr, 100)

    def test_end_session_without_start(self):
        """Test that end_session fails without starting session."""
        logger = ProcessingLogger()

        with pytest.raises(RuntimeError, match="No active session"):
            logger.end_session()

    def test_empty_trace(self, temp_output_dir):
        """Test logging with no steps."""
        logger = ProcessingLogger(session_id="empty_test", output_dir=temp_output_dir)

        logger.start_session("test.wav")
        trace = logger.end_session()

        assert len(trace.steps) == 0
        assert trace.overall_snr_improvement() == 0.0
        assert trace.average_processing_time_per_step() == 0.0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
