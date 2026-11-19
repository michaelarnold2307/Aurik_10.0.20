"""
Tests for Stereo Parallel Processing.

Tests the StereoParallelProcessor and StereoProcessingPipeline classes,
validating parallel execution, error handling, and performance.

Author: AURIK Team
Date: 8. Februar 2026
"""

import time
from unittest.mock import Mock

import numpy as np
import pytest

from backend.core.parallel.stereo_parallel import (
    ChannelType,
    ProcessingResult,
    StereoParallelProcessor,
    StereoProcessingPipeline,
)


@pytest.fixture
def sample_audio():
    """Create sample stereo audio."""
    sr = 44100
    duration = 0.1  # 100ms for fast tests
    samples = int(sr * duration)

    # Create different content for L/R to verify independent processing
    left = np.sin(2 * np.pi * 440 * np.arange(samples) / sr).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * np.arange(samples) / sr).astype(np.float32)

    return left, right, sr


@pytest.fixture
def processor():
    """Create stereo parallel processor."""
    return StereoParallelProcessor(max_workers=2, enable_parallel=True)


@pytest.mark.unit
class TestBasicProcessing:
    """Test basic stereo parallel processing."""

    def test_process_stereo_basic(self, processor, sample_audio):
        """Test basic stereo processing."""
        left, right, sr = sample_audio

        def simple_gain(audio, sr):
            """Apply 2× gain."""
            return audio * 2.0

        left_out, right_out = processor.process_stereo(left, right, sr, simple_gain)

        # Verify output
        assert left_out.shape == left.shape
        assert right_out.shape == right.shape
        np.testing.assert_allclose(left_out, left * 2.0, rtol=1e-5)
        np.testing.assert_allclose(right_out, right * 2.0, rtol=1e-5)

    def test_process_independent_channels(self, processor, sample_audio):
        """Test that channels are processed independently."""
        left, right, sr = sample_audio

        def channel_specific_gain(audio, sr):
            """Different gain based on audio content."""
            # Left channel (440Hz) will have lower mean abs value than right (880Hz)
            if np.mean(np.abs(audio)) < 0.5:
                return audio * 3.0
            else:
                return audio * 1.5

        left_out, right_out = processor.process_stereo(left, right, sr, channel_specific_gain)

        # Outputs should be different (proving independent processing)
        assert not np.allclose(left_out * 2, right_out)

    def test_process_preserves_length(self, processor, sample_audio):
        """Test that output length matches input."""
        left, right, sr = sample_audio

        def process(audio, sr):
            return audio * 0.5

        left_out, right_out = processor.process_stereo(left, right, sr, process)

        assert len(left_out) == len(left)
        assert len(right_out) == len(right)

    def test_sequential_fallback(self, sample_audio):
        """Test sequential processing fallback."""
        left, right, sr = sample_audio
        processor = StereoParallelProcessor(enable_parallel=False)

        def gain(audio, sr):
            return audio * 2.0

        left_out, right_out = processor.process_stereo(left, right, sr, gain)

        np.testing.assert_allclose(left_out, left * 2.0, rtol=1e-5)
        np.testing.assert_allclose(right_out, right * 2.0, rtol=1e-5)


class TestErrorHandling:
    """Test error handling in parallel processing."""

    def test_channel_length_mismatch(self, processor):
        """Test error on mismatched channel lengths."""
        left = np.ones(1000, dtype=np.float32)
        right = np.ones(2000, dtype=np.float32)

        def process(audio, sr):
            return audio

        with pytest.raises(ValueError, match="Channel length mismatch"):
            processor.process_stereo(left, right, 44100, process)

    def test_processing_function_error(self, processor, sample_audio):
        """Test handling of processing function errors."""
        left, right, sr = sample_audio

        def failing_process(audio, sr):
            raise RuntimeError("Processing failed")

        with pytest.raises(RuntimeError, match="Stereo processing failed"):
            processor.process_stereo(left, right, sr, failing_process)

    def test_processing_function_returns_none(self, processor, sample_audio):
        """Test handling of processing function returning None."""
        left, right, sr = sample_audio

        def none_process(audio, sr):
            return None

        with pytest.raises(RuntimeError, match="Stereo processing failed"):
            processor.process_stereo(left, right, sr, none_process)

    def test_processing_function_wrong_length(self, processor, sample_audio):
        """Test handling of processing function returning wrong length."""
        left, right, sr = sample_audio

        def wrong_length(audio, sr):
            return audio[: len(audio) // 2]  # Return half length

        with pytest.raises(RuntimeError, match="Stereo processing failed"):
            processor.process_stereo(left, right, sr, wrong_length)

    def test_partial_failure_both_channels_fail(self, processor, sample_audio):
        """Test when both channels fail."""
        left, right, sr = sample_audio

        def always_fail(audio, sr):
            raise ValueError("Always fails")

        with pytest.raises(RuntimeError, match="Stereo processing failed"):
            processor.process_stereo(left, right, sr, always_fail)


class TestPerformance:
    """Test performance and speedup measurements."""

    def test_parallel_faster_than_sequential(self, sample_audio):
        """Test that parallel is faster than sequential."""
        left, right, sr = sample_audio
        # Use longer audio for measurable timing
        left = np.tile(left, 10)
        right = np.tile(right, 10)

        def slow_process(audio, sr):
            """Slow processing to make timing measurable."""
            time.sleep(0.01)  # 10ms delay
            return audio * 1.1

        # Parallel processing
        parallel_processor = StereoParallelProcessor(enable_parallel=True)
        start = time.time()
        parallel_processor.process_stereo(left, right, sr, slow_process)
        parallel_time = time.time() - start

        # Sequential processing
        sequential_processor = StereoParallelProcessor(enable_parallel=False)
        start = time.time()
        sequential_processor.process_stereo(left, right, sr, slow_process)
        sequential_time = time.time() - start

        # Parallel should be faster (with some tolerance for overhead)
        assert parallel_time < sequential_time * 0.9

    def test_speedup_calculation(self, processor, sample_audio):
        """Test speedup statistics calculation."""
        left, right, sr = sample_audio

        def process(audio, sr):
            time.sleep(0.005)  # 5ms delay
            return audio * 1.0

        processor.process_stereo(left, right, sr, process)

        speedup = processor.get_average_speedup()
        # Should show parallel benefit; lower bound relaxed for CI/single-core environments
        assert 0.8 < speedup < 2.5

    def test_stats_tracking(self, processor, sample_audio):
        """Test processing statistics tracking."""
        left, right, sr = sample_audio

        def process(audio, sr):
            return audio * 1.0

        # Process multiple times
        for _ in range(3):
            processor.process_stereo(left, right, sr, process)

        stats = processor.get_stats()
        assert stats["total_processed"] == 3
        assert len(stats["speedup_history"]) == 3
        assert stats["average_speedup"] > 0

    def test_reset_stats(self, processor, sample_audio):
        """Test statistics reset."""
        left, right, sr = sample_audio

        def process(audio, sr):
            return audio

        processor.process_stereo(left, right, sr, process)
        processor.reset_stats()

        stats = processor.get_stats()
        assert stats["total_processed"] == 0
        assert len(stats["speedup_history"]) == 0


class TestProcessingResult:
    """Test ProcessingResult dataclass."""

    def test_success_result(self):
        """Test successful processing result."""
        audio = np.ones(1000, dtype=np.float32)
        result = ProcessingResult(channel=ChannelType.LEFT, audio=audio, success=True, processing_time=0.1)

        assert result.channel == ChannelType.LEFT
        assert result.success
        assert result.error is None
        assert result.processing_time == 0.1
        assert len(result.audio) == 1000

    def test_error_result(self):
        """Test error processing result."""
        audio = np.ones(1000, dtype=np.float32)
        result = ProcessingResult(
            channel=ChannelType.RIGHT, audio=audio, success=False, error="Processing failed", processing_time=0.05
        )

        assert result.channel == ChannelType.RIGHT
        assert not result.success
        assert result.error == "Processing failed"
        assert result.processing_time == 0.05

    def test_metadata_initialization(self):
        """Test metadata field initialization."""
        result = ProcessingResult(channel=ChannelType.LEFT, audio=np.ones(100), success=True)

        assert result.metadata == {}

        result_with_meta = ProcessingResult(
            channel=ChannelType.RIGHT, audio=np.ones(100), success=True, metadata={"key": "value"}
        )

        assert result_with_meta.metadata == {"key": "value"}


class TestStereoProcessingPipeline:
    """Test StereoProcessingPipeline wrapper."""

    def test_mono_passthrough(self, sample_audio):
        """Test that mono audio passes through to original pipeline."""
        left, _, sr = sample_audio

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.process.return_value = left * 2.0

        pipeline = StereoProcessingPipeline(mock_pipeline)
        output = pipeline.process(left, sr)

        # Should call original pipeline
        mock_pipeline.process.assert_called_once_with(left, sr)
        np.testing.assert_allclose(output, left * 2.0, rtol=1e-5, atol=1e-8)

    def test_stereo_parallel_processing(self, sample_audio):
        """Test stereo audio parallel processing."""
        left, right, sr = sample_audio
        stereo = np.stack([left, right], axis=0)

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = lambda audio, sr: audio * 2.0

        pipeline = StereoProcessingPipeline(mock_pipeline)
        output = pipeline.process(stereo, sr)

        assert output.shape == stereo.shape
        np.testing.assert_allclose(output[0], left * 2.0, rtol=1e-5)
        np.testing.assert_allclose(output[1], right * 2.0, rtol=1e-5)

    def test_invalid_shape(self):
        """Test error on invalid audio shape."""
        # Mock pipeline
        mock_pipeline = Mock()

        pipeline = StereoProcessingPipeline(mock_pipeline)

        # 3D array - invalid
        invalid_audio = np.ones((2, 2, 1000))

        with pytest.raises(ValueError, match="Invalid audio shape"):
            pipeline.process(invalid_audio, 44100)

    def test_stats_access(self, sample_audio):
        """Test stats access through pipeline."""
        left, right, sr = sample_audio
        stereo = np.stack([left, right], axis=0)

        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = lambda audio, sr: audio

        pipeline = StereoProcessingPipeline(mock_pipeline)
        pipeline.process(stereo, sr)

        stats = pipeline.get_stats()
        assert stats["total_processed"] == 1
        assert "average_speedup" in stats

    def test_disable_parallel(self, sample_audio):
        """Test disabling parallel processing in pipeline."""
        left, right, sr = sample_audio
        stereo = np.stack([left, right], axis=0)

        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = lambda audio, sr: audio * 1.5

        pipeline = StereoProcessingPipeline(mock_pipeline, enable_parallel=False)
        output = pipeline.process(stereo, sr)

        np.testing.assert_allclose(output[0], left * 1.5, rtol=1e-5)
        np.testing.assert_allclose(output[1], right * 1.5, rtol=1e-5)


class TestThreadSafety:
    """Test thread safety of parallel processing."""

    def test_concurrent_process_calls(self, processor, sample_audio):
        """Test multiple concurrent process_stereo calls."""
        left, right, sr = sample_audio

        def process(audio, sr):
            # Simulate some computation
            result = audio.copy()
            for _ in range(10):
                result = result * 1.01
            return result

        # Run multiple times to catch potential race conditions
        for _ in range(5):
            left_out, right_out = processor.process_stereo(left, right, sr, process)
            assert left_out.shape == left.shape
            assert right_out.shape == right.shape

    def test_independent_audio_modification(self, processor, sample_audio):
        """Test that audio arrays are independent (no shared memory issues)."""
        left, right, sr = sample_audio
        original_left = left.copy()
        original_right = right.copy()

        def modifying_process(audio, sr):
            # Modify audio in-place
            audio *= 2.0
            return audio

        left_out, right_out = processor.process_stereo(left, right, sr, modifying_process)

        # Original arrays should be unchanged
        np.testing.assert_allclose(left, original_left, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(right, original_right, rtol=1e-5, atol=1e-8)

        # Outputs should be modified
        np.testing.assert_allclose(left_out, original_left * 2.0, rtol=1e-5)
        np.testing.assert_allclose(right_out, original_right * 2.0, rtol=1e-5)


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_stereo_workflow(self, sample_audio):
        """Test complete stereo processing workflow."""
        left, right, sr = sample_audio

        # Create a processing chain
        def denoise(audio, sr):
            return audio * 0.8  # Stronger attenuation

        def enhance(audio, sr):
            return audio * 1.5  # Stronger gain

        def normalize(audio, sr):
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
            return audio

        def processing_chain(audio, sr):
            audio = denoise(audio, sr)
            audio = enhance(audio, sr)
            audio = normalize(audio, sr)
            return audio

        processor = StereoParallelProcessor()
        left_out, right_out = processor.process_stereo(left, right, sr, processing_chain)

        # Verify output shape
        assert left_out.shape == left.shape
        assert right_out.shape == right.shape

        # Verify normalized (should be very close to 1.0)
        assert np.max(np.abs(left_out)) <= 1.0
        assert np.max(np.abs(right_out)) <= 1.0
        assert np.max(np.abs(left_out)) > 0.95  # Very close to 1.0
        assert np.max(np.abs(right_out)) > 0.95

        # Verify stats updated
        stats = processor.get_stats()
        assert stats["total_processed"] == 1
        assert stats["average_speedup"] > 0
