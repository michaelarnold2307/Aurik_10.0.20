"""
Performance Benchmarks for Aurik DSP Modules
Uses pytest-benchmark for consistent, reproducible performance testing
"""

import numpy as np
import pytest


# Fixture: Generate test audio
@pytest.fixture
def audio_signal():
    """Generate 1-second stereo audio at 44.1kHz"""
    sr = 44100
    duration = 1.0
    samples = int(sr * duration)
    # Stereo signal: sine wave with noise
    t = np.linspace(0, duration, samples)
    signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(samples)
    stereo = np.stack([signal, signal])
    return stereo, sr


@pytest.fixture
def long_audio_signal():
    """Generate 10-second stereo audio for heavy operations"""
    sr = 44100
    duration = 10.0
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples)
    signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(samples)
    stereo = np.stack([signal, signal])
    return stereo, sr


class TestFFTBenchmarks:
    """Benchmark FFT operations"""

    def test_fft_performance(self, benchmark, audio_signal):
        """Benchmark standard FFT"""
        audio, sr = audio_signal

        def run_fft():
            return np.fft.fft(audio[0])

        result = benchmark(run_fft)
        assert len(result) == len(audio[0])

    def test_rfft_performance(self, benchmark, audio_signal):
        """Benchmark real FFT (more efficient for real signals)"""
        audio, sr = audio_signal

        def run_rfft():
            return np.fft.rfft(audio[0])

        result = benchmark(run_rfft)
        assert result is not None

    def test_stft_performance(self, benchmark, audio_signal):
        """Benchmark Short-Time Fourier Transform"""
        audio, sr = audio_signal
        from scipy import signal as scipy_signal

        def run_stft():
            return scipy_signal.stft(audio[0], sr, nperseg=2048, noverlap=512)

        result = benchmark(run_stft)
        assert len(result) == 3  # f, t, Zxx


class TestResamplingBenchmarks:
    """Benchmark resampling operations"""

    def test_resample_44100_to_48000(self, benchmark, audio_signal):
        """Benchmark upsampling from 44.1kHz to 48kHz"""
        audio, sr = audio_signal
        from scipy import signal as scipy_signal

        target_sr = 48000
        num_samples = int(len(audio[0]) * target_sr / sr)

        def run_resample():
            return scipy_signal.resample(audio[0], num_samples)

        result = benchmark(run_resample)
        assert len(result) == num_samples

    def test_resample_44100_to_16000(self, benchmark, audio_signal):
        """Benchmark downsampling from 44.1kHz to 16kHz"""
        audio, sr = audio_signal
        from scipy import signal as scipy_signal

        target_sr = 16000
        num_samples = int(len(audio[0]) * target_sr / sr)

        def run_resample():
            return scipy_signal.resample(audio[0], num_samples)

        result = benchmark(run_resample)
        assert len(result) == num_samples


class TestFilteringBenchmarks:
    """Benchmark filtering operations"""

    def test_bandpass_filter(self, benchmark, audio_signal):
        """Benchmark bandpass filter application"""
        audio, sr = audio_signal
        from scipy.signal import butter, sosfilt

        # Design bandpass filter
        sos = butter(4, [100, 5000], btype="band", fs=sr, output="sos")

        def run_filter():
            return sosfilt(sos, audio[0])

        result = benchmark(run_filter)
        assert len(result) == len(audio[0])

    def test_lowpass_filter(self, benchmark, audio_signal):
        """Benchmark lowpass filter application"""
        audio, sr = audio_signal
        from scipy.signal import butter, sosfilt

        sos = butter(8, 8000, btype="low", fs=sr, output="sos")

        def run_filter():
            return sosfilt(sos, audio[0])

        result = benchmark(run_filter)
        assert len(result) == len(audio[0])


class TestArrayOperationsBenchmarks:
    """Benchmark common array operations"""

    def test_rms_calculation(self, benchmark, audio_signal):
        """Benchmark RMS calculation"""
        audio, sr = audio_signal

        def run_rms():
            return np.sqrt(np.mean(audio[0] ** 2))

        result = benchmark(run_rms)
        assert result > 0

    def test_normalization(self, benchmark, audio_signal):
        """Benchmark audio normalization"""
        audio, sr = audio_signal

        def run_normalize():
            max_val = np.abs(audio).max()
            return audio / max_val if max_val > 0 else audio

        result = benchmark(run_normalize)
        assert result.shape == audio.shape

    def test_peak_detection(self, benchmark, audio_signal):
        """Benchmark peak detection"""
        audio, sr = audio_signal
        from scipy.signal import find_peaks

        def run_peak_detection():
            return find_peaks(audio[0], height=0.5, distance=100)

        result = benchmark(run_peak_detection)
        assert len(result) == 2  # peaks, properties


class TestConvolutionBenchmarks:
    """Benchmark convolution operations"""

    def test_direct_convolution(self, benchmark, audio_signal):
        """Benchmark direct convolution"""
        audio, sr = audio_signal
        # Create impulse response
        ir = np.random.randn(1000)

        def run_convolution():
            return np.convolve(audio[0], ir, mode="same")

        result = benchmark(run_convolution)
        assert len(result) == len(audio[0])

    def test_fft_convolution(self, benchmark, audio_signal):
        """Benchmark FFT-based convolution (faster for long kernels)"""
        audio, sr = audio_signal
        from scipy.signal import fftconvolve

        ir = np.random.randn(1000)

        def run_fft_convolution():
            return fftconvolve(audio[0], ir, mode="same")

        result = benchmark(run_fft_convolution)
        assert len(result) == len(audio[0])


# Performance regression test
def test_performance_regression_guard(benchmark):
    """Ensure basic operations stay fast (regression guard)"""
    data = np.random.randn(44100)  # 1 second of audio

    def compute_spectrogram():
        return np.abs(np.fft.rfft(data)) ** 2

    benchmark(compute_spectrogram)

    # Benchmark should complete in < 10ms on modern hardware
    stats = benchmark.stats.stats
    assert stats.mean < 0.01, f"Performance regression detected: {stats.mean:.4f}s"


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-autosave"])
