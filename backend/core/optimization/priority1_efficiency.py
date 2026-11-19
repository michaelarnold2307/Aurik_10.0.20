"""
optimization/priority1_efficiency.py – Algorithmische Effizienzoptimierung.
===========================================================================

Provides vectorised multi-core FFT processing and benchmark utilities.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class OptimizedFFT:
    """Configured FFT with sensible defaults for audio at a given sample rate.

    Parameters
    ----------
    sr:
        Sample rate (Hz).  Default: 48 000.
    """

    def __init__(self, sr: int = 48000) -> None:
        self.sr = sr
        self.n_fft = 4096  # 4 K FFT — required invariant from test

    def get_frequency_resolution(self) -> float:
        """Gibt frequency bin size in Hz zurück.

        Resolution = sr / n_fft.  For sr=48000, n_fft=4096 → ~11.7 Hz.
        """
        return self.sr / self.n_fft

    def fft(self, frame: np.ndarray) -> np.ndarray:
        """Berechnet real-valued FFT of *frame*."""
        return np.fft.rfft(frame, n=self.n_fft)  # type: ignore[no-any-return]

    def ifft(self, spectrum: np.ndarray) -> np.ndarray:
        """Berechnet inverse FFT back to time domain."""
        return np.fft.irfft(spectrum, n=self.n_fft)  # type: ignore[no-any-return]


class AlgorithmicEfficiencyOptimizer:
    """Vectorised frame-wise audio processor with optional multi-core support.

    Parameters
    ----------
    sr:
        Sample rate.
    n_cores:
        Number of worker threads for multi-core mode.
    """

    def __init__(self, sr: int = 48000, n_cores: int = 2) -> None:
        self.sr = sr
        self.n_cores = max(1, n_cores)
        self._fft = OptimizedFFT(sr=sr)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        use_multicore: bool = False,
    ) -> np.ndarray:
        """Wendet an: spectral gain normalisation frame by frame.

        Returns an ndarray of the same length and dtype float32.
        """
        audio_f32 = np.asarray(audio, dtype=np.float32)

        if use_multicore and self.n_cores > 1:
            result = self._process_multicore(audio_f32)
        else:
            result = self._process_single(audio_f32)

        return np.nan_to_num(np.clip(result, -1.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)  # type: ignore[no-any-return]

    def _process_single(self, audio: np.ndarray) -> np.ndarray:
        hop = self._fft.n_fft // 2
        out = np.zeros_like(audio)
        win = np.hanning(self._fft.n_fft)
        n = len(audio)
        count = np.zeros_like(audio)

        for start in range(0, n - self._fft.n_fft + 1, hop):
            frame = audio[start : start + self._fft.n_fft] * win
            spec = self._fft.fft(frame)
            mag = np.abs(spec)
            # Gentle spectral normalisation
            mean_mag = np.mean(mag) + 1e-12
            gain = np.where(mag > mean_mag * 0.1, 1.0, mag / (mean_mag * 0.1 + 1e-12))
            spec_out = spec * gain
            restored = self._fft.ifft(spec_out)[: self._fft.n_fft]
            out[start : start + self._fft.n_fft] += restored * win
            count[start : start + self._fft.n_fft] += win**2

        # OLA normalisation
        count = np.where(count > 1e-8, count, 1.0)
        return out / count  # type: ignore[no-any-return]

    def _process_multicore(self, audio: np.ndarray) -> np.ndarray:
        n_chunks = self.n_cores
        chunks = np.array_split(audio, n_chunks)

        def _proc(chunk: np.ndarray) -> np.ndarray:
            return self._process_single(chunk)

        with ThreadPoolExecutor(max_workers=self.n_cores) as ex:
            processed = list(ex.map(_proc, chunks))

        return np.concatenate(processed)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(self, audio: np.ndarray, sr: int, n_iterations: int = 2) -> dict[str, object]:
        """Misst single-core vs. multi-core throughput.

        Returns
        -------
        dict with keys ``"multicore_speedup"`` and ``"n_cores"``.
        """
        audio_f32 = np.asarray(audio, dtype=np.float32)

        # Single-core timing
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            self._process_single(audio_f32)
        t_single = (time.perf_counter() - t0) / n_iterations + 1e-9

        # Multi-core timing
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            self._process_multicore(audio_f32)
        t_multi = (time.perf_counter() - t0) / n_iterations + 1e-9

        speedup = t_single / t_multi

        return {
            "multicore_speedup": float(speedup),
            "n_cores": self.n_cores,
            "single_core_s": float(t_single),
            "multi_core_s": float(t_multi),
        }
