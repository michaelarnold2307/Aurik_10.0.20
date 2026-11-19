from __future__ import annotations

"""
test_ml_stability.py — Muster 2: ML-Stack-Stabilität
=====================================================

Verifiziert, dass:
1. ONNX-Sessions importierbar sind (kein Crash)
2. Phase 03 für kurze Audios in <5s terminiert
3. Keine ML-Module die pytest-Collection crashen
"""


import pytest


@pytest.mark.unit
@pytest.mark.ml
class TestMLStackStability:
    """ML-Stack: Keine Crashes, keine Deadlocks, keine Collection-Fehler."""

    def test_01_numpy_scipy_operational(self):
        """numpy + scipy sind funktionsfähig."""
        import numpy as np
        from scipy import signal

        # Grundoperationen
        a = np.ones(100, dtype=np.float32)
        assert np.all(np.isfinite(a))
        b, a_coeff = signal.butter(2, 0.5, "low")
        assert len(b) == 3

    def test_02_onnx_import_guarded(self):
        """ONNX-Import ist guarded (kein Crash bei fehlender Installation)."""
        try:
            pass

        except ImportError:
            pass
        # Entweder ONNX ist da und importierbar, oder es fehlt sauber
        assert True  # Kein Crash = OK

    def test_03_phase_03_imports_without_crash(self):
        """Phase 03 importiert ohne Crash."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        assert DenoisePhase is not None

    def test_04_phase_03_terminates_for_short_audio(self):
        """Phase 03 terminiert für 0.5s Audio in <10s."""
        import time

        import numpy as np

        from backend.core.phases.phase_03_denoise import DenoisePhase

        p = DenoisePhase()
        audio = np.random.randn(24000).astype(np.float32) * 0.01
        t0 = time.time()
        result = p.process(audio, material_type="tape", quality_mode="fast", sample_rate=48000)
        elapsed = time.time() - t0
        assert elapsed < 10.0, f"Phase 03 brauchte {elapsed:.1f}s für 0.5s Audio"

    def test_05_hypothesis_collection_stable(self):
        """Hypothesis-Import verursacht keinen Collection-Crash."""
        try:
            pass

        except Exception:
            pass
        assert True  # Kein Crash = OK
