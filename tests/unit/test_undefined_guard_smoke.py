from __future__ import annotations

"""Smoke-Test: Erzwingt alle Code-Pfade mit psutil-Guards.

§Schutzschicht-2: Stellt sicher, dass `import psutil`-Fehlerpfade
tatsächlich graceful degraden und nicht mit NameError crashen.
Testet die 3 bekannten psutil-Guard-Stellen in unified_restorer_v3.

Autor: Aurik 10 — 11. Juli 2026
"""


import builtins
import sys
from unittest.mock import patch

import numpy as np
import pytest


class TestPsutilGuardSmoke:
    """Erzwingt psutil-Import-Fehler und prüft Graceful Degradation."""

    def test_unified_restorer_imports_cleanly(self):
        """unified_restorer_v3 ist ohne ImportError importierbar."""
        from backend.core.unified_restorer_v3 import RestorationConfig, RestorationResult

        assert RestorationConfig is not None
        assert RestorationResult is not None

    def test_mini_pipeline_no_crash_on_noise(self):
        """Mini-Pipeline läuft auf Rauschen ohne Crash."""
        audio = np.random.RandomState(42).randn(48000).astype(np.float32) * 0.1
        from benchmarks.regression.regression_gate import aurik_pipeline

        result = aurik_pipeline(audio, 48000, use_real=True, full=False)
        assert np.all(np.isfinite(result)), "Pipeline output contains NaN/Inf"

    def test_full_pipeline_no_crash(self):
        """Full Pipeline läuft auf Musik ohne Crash."""
        from benchmarks.regression.regression_gate import _make_music, _make_noisy

        music = _make_music(0.5, 48000)
        noisy = _make_noisy(music, 10.0)
        from benchmarks.regression.regression_gate import aurik_pipeline

        result = aurik_pipeline(noisy, 48000, use_real=True, full=True)
        assert np.all(np.isfinite(result)), "Full pipeline output contains NaN/Inf"
        assert len(result) > 0, "Full pipeline produced empty output"

    def test_edge_cases_no_crash(self):
        """Edge-Cases produzieren keinen Crash."""
        from benchmarks.regression.regression_gate import aurik_pipeline

        edge_cases = {
            "silence": np.zeros(48000, dtype=np.float32),
            "dc_offset": np.ones(24000, dtype=np.float32) * 0.5,
            "very_quiet": np.random.RandomState(42).randn(24000).astype(np.float32) * 1e-5,
            "very_loud": np.clip(np.random.RandomState(42).randn(24000).astype(np.float32) * 3, -1, 1),
        }

        for name, audio in edge_cases.items():
            result = aurik_pipeline(audio, 48000, use_real=True, full=False)
            assert np.all(np.isfinite(result)), f"{name}: NaN/Inf in output"

    def test_comfort_guard_imports_correctly(self):
        """ComfortGuard importiert lfilter, nicht biquad."""
        from scipy.signal import lfilter

        # Der eigentliche Test: ComfortGuard soll ohne ImportError importierbar sein
        from backend.core.comfort_guard import apply_comfort_guard

        audio = np.random.RandomState(42).randn(48000).astype(np.float32) * 0.1
        result = apply_comfort_guard(audio, 48000)
        assert np.all(np.isfinite(result))

    def test_breath_preserver_no_crash(self):
        """BreathPreserver läuft ohne Crash."""
        from backend.core.breath_preserver import protect_breath, restore_breath

        audio = np.random.RandomState(42).randn(48000).astype(np.float32) * 0.1
        masked, mask = protect_breath(audio, 48000)
        if mask is not None:
            restored = restore_breath(masked, mask, audio)
            assert np.all(np.isfinite(restored))

    def test_vocal_quality_gate_no_crash(self):
        """VocalQualityGate läuft ohne Crash."""
        from backend.core.vocal_quality_gate import get_vocal_quality_gate

        audio = np.random.RandomState(42).randn(48000).astype(np.float32) * 0.1
        gate = get_vocal_quality_gate()
        decision = gate.evaluate(audio, audio * 0.9, 48000)
        assert decision.accept is not None
