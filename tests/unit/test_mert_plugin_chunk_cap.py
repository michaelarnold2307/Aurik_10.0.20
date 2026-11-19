from __future__ import annotations

"""Regression guard: MERT analyze() must crop audio to ≤ 10 s before HF inference.

30 s was the old cap that caused >180 s CPU inference (O(n²) attention on 2250
tokens). This test ensures the crop stays at 10 s to prevent future regressions.
"""


import numpy as np
import pytest


class TestMertChunkCap:
    """Verify that MertPlugin.analyze() crops to ≤ 10 s before model inference."""

    def _make_plugin_with_dsp_fallback(self):
        """Return a MertPlugin instance that always falls back to DSP (no ML model loaded)."""
        from plugins.mert_plugin import MertPlugin

        p = MertPlugin.__new__(MertPlugin)
        # Minimal attribute set needed by analyze()
        p._model = None
        p._model_type = "dsp"  # Forces DSP path — no HF forward pass
        p._target_sr = 24_000
        p._processor = None
        p._device = "cpu"
        p._analysis_cache = {}
        p._analysis_cache_lock = __import__("threading").Lock()
        p._analysis_cache_max_entries = 32
        return p

    @pytest.mark.unit
    def test_analyze_crops_long_audio_to_10s(self, monkeypatch):
        """Audio longer than 10 s must be cropped before reaching model inference.

        We monkeypatch _dsp_analyze to capture the length of the audio it receives.
        """
        import plugins.mert_plugin as mp

        received_lengths: list[int] = []

        original_dsp = mp._dsp_analyze

        def spy_dsp(audio, sr):
            received_lengths.append(len(audio))
            return original_dsp(audio, sr)

        monkeypatch.setattr(mp, "_dsp_analyze", spy_dsp)

        plugin = self._make_plugin_with_dsp_fallback()
        sr = 24_000
        # 60-second audio — well above the 10 s cap
        long_audio = np.zeros(60 * sr, dtype=np.float32)

        plugin.analyze(long_audio, sr)

        assert received_lengths, "DSP analyze must be called"
        max_received = max(received_lengths)
        assert max_received <= 10 * sr, (
            f"MERT crop must limit input to ≤ 10 s (≤ {10 * sr} samples), "
            f"but _dsp_analyze received {max_received} samples "
            f"({max_received / sr:.1f} s)"
        )

    @pytest.mark.unit
    def test_analyze_does_not_crop_short_audio(self, monkeypatch):
        """Audio shorter than 10 s must NOT be cropped (no accuracy loss)."""
        import plugins.mert_plugin as mp

        received_lengths: list[int] = []

        original_dsp = mp._dsp_analyze

        def spy_dsp(audio, sr):
            received_lengths.append(len(audio))
            return original_dsp(audio, sr)

        monkeypatch.setattr(mp, "_dsp_analyze", spy_dsp)

        plugin = self._make_plugin_with_dsp_fallback()
        sr = 24_000
        short_audio = np.zeros(5 * sr, dtype=np.float32)  # 5 seconds

        plugin.analyze(short_audio, sr)

        assert received_lengths
        # Must receive the full 5 s (allow ±10 samples for padding)
        assert received_lengths[0] >= 5 * sr - 10, (
            f"5 s audio must not be cropped, but received {received_lengths[0]} samples"
        )
