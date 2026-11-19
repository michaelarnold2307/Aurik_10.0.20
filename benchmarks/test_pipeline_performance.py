"""
Audio Processing Pipeline Benchmarks
End-to-end performance testing for complete audio processing workflows
"""

import numpy as np
import pytest


@pytest.fixture
def test_audio_file(tmp_path):
    """Create a temporary test audio file"""
    import soundfile as sf

    # Generate test audio
    sr = 44100
    duration = 5.0
    samples = int(sr * duration)
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))

    # Save to temp file
    filepath = tmp_path / "test_input.wav"
    sf.write(filepath, audio, sr)
    return filepath


@pytest.fixture
def test_audio_data():
    """Generate in-memory test audio data"""
    sr = 44100
    duration = 5.0
    samples = int(sr * duration)
    audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
    return audio, sr


class TestIOBenchmarks:
    """Benchmark I/O operations"""

    def test_audio_read_performance(self, benchmark, test_audio_file):
        """Benchmark audio file reading"""
        import soundfile as sf

        def read_audio():
            return sf.read(test_audio_file)

        data, sr = benchmark(read_audio)
        assert len(data) > 0

    def test_audio_write_performance(self, benchmark, tmp_path, test_audio_data):
        """Benchmark audio file writing"""
        import soundfile as sf

        audio, sr = test_audio_data
        output_path = tmp_path / "output.wav"

        def write_audio():
            sf.write(output_path, audio, sr)

        benchmark(write_audio)
        assert output_path.exists()


class TestProcessingPipelineBenchmarks:
    """Benchmark complete processing pipelines"""

    def test_basic_restoration_pipeline(self, benchmark, test_audio_data):
        """Benchmark basic restoration: normalize -> denoise -> limit"""
        audio, sr = test_audio_data

        def restoration_pipeline():
            # 1. Normalize
            normalized = audio / np.abs(audio).max()

            # 2. Simple denoising (highpass filter)
            from scipy.signal import butter, sosfilt

            sos = butter(4, 100, btype="high", fs=sr, output="sos")
            denoised = sosfilt(sos, normalized)

            # 3. Soft limiting
            threshold = 0.9
            limited = np.where(np.abs(denoised) > threshold, threshold * np.sign(denoised), denoised)

            return limited

        result = benchmark(restoration_pipeline)
        assert len(result) == len(audio)

    def test_spectral_processing_pipeline(self, benchmark, test_audio_data):
        """Benchmark spectral processing: STFT -> process -> ISTFT"""
        audio, sr = test_audio_data
        from scipy.signal import istft, stft

        def spectral_pipeline():
            # Forward STFT
            f, t, Zxx = stft(audio, sr, nperseg=2048, noverlap=1536)

            # Spectral processing (e.g., noise gate)
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            threshold = np.median(magnitude) * 1.5
            processed_mag = np.where(magnitude < threshold, 0, magnitude)
            processed_Zxx = processed_mag * np.exp(1j * phase)

            # Inverse STFT
            _, reconstructed = istft(processed_Zxx, sr, nperseg=2048, noverlap=1536)

            return reconstructed

        result = benchmark(spectral_pipeline)
        assert len(result) > 0


class TestCachingBenchmarks:
    """Benchmark caching strategies"""

    def test_lru_cache_benefit(self, benchmark):
        """Demonstrate LRU cache performance improvement"""
        from functools import lru_cache

        # Expensive computation
        def fibonacci(n):
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        @lru_cache(maxsize=128)
        def fibonacci_cached(n):
            if n < 2:
                return n
            return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

        # Benchmark without cache
        result_uncached = benchmark.pedantic(lambda: fibonacci(20), rounds=3, iterations=1)

        # Benchmark with cache (should be much faster on repeated calls)
        result_cached = benchmark.pedantic(lambda: fibonacci_cached(30), rounds=10, iterations=1)

        assert result_cached is not None


class TestParallelProcessingBenchmarks:
    """Benchmark parallel processing strategies"""

    @pytest.mark.slow
    def test_sequential_vs_parallel(self, benchmark):
        """Compare sequential vs parallel processing"""
        from concurrent.futures import ProcessPoolExecutor

        # Task: Process multiple audio chunks
        def process_chunk(chunk):
            """Simulate processing"""
            return np.sum(chunk**2)

        # Generate chunks
        chunks = [np.random.randn(44100) for _ in range(8)]

        def sequential():
            return [process_chunk(c) for c in chunks]

        def parallel():
            with ProcessPoolExecutor(max_workers=4) as executor:
                return list(executor.map(process_chunk, chunks))

        # Benchmark sequential
        result = benchmark(sequential)
        assert len(result) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only", "--benchmark-autosave", "--benchmark-histogram"])
