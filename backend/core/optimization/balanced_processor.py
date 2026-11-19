"""
optimization/balanced_processor.py – Ausgeglichener Audio-Prozessor.
=============================================================

Integrates all 6 optimization priorities into a single processing chain.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from backend.core.optimization.priority1_efficiency import AlgorithmicEfficiencyOptimizer
from backend.core.optimization.priority2_vocals import SelectiveVocalEnhancer
from backend.core.optimization.priority3_oversampling import AdaptiveOversamplingProcessor
from backend.core.optimization.priority4_phase import MultibandPhaseCoherenceEnhancer
from backend.core.optimization.priority5_bass import PhaseCoherentBassProcessor
from backend.core.optimization.priority6_parameters import GenreOptimizedParameters, OptimizedPresets


class BalancedAudioProcessor:
    """Full balanced-optimization pipeline.

    Parameters
    ----------
    sr:
        Sample rate (Hz).
    preset:
        One of ``"gentle"``, ``"balanced"``, ``"aggressive"``.
    n_cores:
        Number of CPU cores for multi-threaded stages.
    """

    def __init__(self, sr: int = 48000, preset: str = "balanced", n_cores: int = 2) -> None:
        self.sr = sr
        self.preset = preset
        self.n_cores = n_cores

        preset_params = OptimizedPresets.get_preset(preset)
        self._preset_params = preset_params

        self.efficiency = AlgorithmicEfficiencyOptimizer(sr=sr, n_cores=n_cores)
        self.vocal_enhancer = SelectiveVocalEnhancer(sr=sr)
        self._oversampler = AdaptiveOversamplingProcessor(sr=sr)
        self._phase_enhancer = MultibandPhaseCoherenceEnhancer(sr=sr)
        self._bass_processor = PhaseCoherentBassProcessor(sr=sr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sr: int, genre: str = "rock") -> np.ndarray:
        """Führt aus: the full 6-stage pipeline on *audio*.

        Returns
        -------
        Processed audio, length within 90–110 % of the input length.
        """
        x = np.asarray(audio, dtype=np.float32)
        if len(x) == 0:
            return x.copy()  # type: ignore[no-any-return]

        # Stage 1: algorithmic efficiency pass
        x = self.efficiency.process(x, sr, use_multicore=False)

        # Stage 2: selective vocal enhancement
        x = self.vocal_enhancer.process(x, sr)

        # Stage 3: adaptive oversampling (transient protection)
        x = self._oversampler.process(x, sr)

        # Stage 4: multiband phase coherence
        x = self._phase_enhancer.process(x, sr)

        # Stage 5: phase-coherent bass processing
        x = self._bass_processor.process(x, sr)

        # Stage 6: genre-specific gain trim
        params = GenreOptimizedParameters.get_parameters(genre)
        bass_boost_db = float(params.get("bass_boost", 1.0))
        gain = 10 ** (bass_boost_db / 40.0)  # gentle
        x = np.clip(x * gain, -1.0, 1.0).astype(np.float32)

        # Ensure output length is within 90–110 % of the input
        target = len(audio)
        if len(x) > int(target * 1.1):
            x = x[: int(target * 1.1)]
        elif len(x) < int(target * 0.9):
            x = np.pad(x, (0, int(target * 0.9) - len(x)))

        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)  # type: ignore[no-any-return]

    def benchmark(self, audio: np.ndarray, sr: int, n_iterations: int = 1) -> dict[str, Any]:
        """Misst real-time factor over *n_iterations*.

        Returns
        -------
        dict with ``"rt_factor"`` and ``"target_achieved"`` (bool, rt < 6).
        """
        audio_f32 = np.asarray(audio, dtype=np.float32)
        duration_s = len(audio_f32) / sr

        total = 0.0
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            self.process(audio_f32, sr)
            total += time.perf_counter() - t0

        rt_factor = (total / n_iterations) / (duration_s + 1e-12)
        return {
            "rt_factor": float(rt_factor),
            "target_achieved": bool(rt_factor < 6.0),
        }
