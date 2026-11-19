"""Tests für resolve_jnd_tolerance_db — Frequenzabhängige JND-Toleranz (§V43).

Abdeckung:
  - Breakpoints: Werte an Stützpunkten korrekt
  - Interpolation: Monoton fallend von 200 Hz bis 6000 Hz
  - Extrapolation: Werte unter 200 Hz und über 6000 Hz
  - Wertebereich: immer in [0.5, 3.5]
  - Wichtige Formant-Frequenzen: F1 (500–800 Hz), F2 (1000–2000 Hz), F3 (2000–3500 Hz), F4 (3500–5000 Hz)
"""

import numpy as np
import pytest


@pytest.mark.unit
class TestResolveJNDToleranceDB:
    """resolve_jnd_tolerance_db — Frequenzabhängige JND-Toleranz."""

    def test_imports(self):
        """Import aus dem kanonischen Pfad funktioniert."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        assert callable(resolve_jnd_tolerance_db)

    def test_below_200hz(self):
        """Unter 200 Hz → 3.0 dB."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        assert resolve_jnd_tolerance_db(100.0) == pytest.approx(3.0)
        assert resolve_jnd_tolerance_db(50.0) == pytest.approx(3.0)
        assert resolve_jnd_tolerance_db(20.0) == pytest.approx(3.0)

    def test_at_200hz(self):
        """An 200 Hz → 3.0 dB (links)."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        v = resolve_jnd_tolerance_db(200.0)
        assert v == pytest.approx(3.0, abs=0.1)

    def test_at_500hz(self):
        """An 500 Hz → 2.0 dB."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        v = resolve_jnd_tolerance_db(500.0)
        assert v == pytest.approx(2.0, abs=0.15)

    def test_at_1000hz(self):
        """An 1000 Hz → 1.5 dB."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        v = resolve_jnd_tolerance_db(1000.0)
        assert v == pytest.approx(1.5, abs=0.15)

    def test_at_2000hz(self):
        """An 2000 Hz → 1.0 dB."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        v = resolve_jnd_tolerance_db(2000.0)
        assert v == pytest.approx(1.0, abs=0.15)

    def test_at_4000hz(self):
        """An 4000 Hz → 0.8 dB."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        v = resolve_jnd_tolerance_db(4000.0)
        assert v == pytest.approx(0.8, abs=0.15)

    def test_above_6000hz(self):
        """Über 6000 Hz → 0.6 dB."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        assert resolve_jnd_tolerance_db(8000.0) == pytest.approx(0.6, abs=0.1)
        assert resolve_jnd_tolerance_db(16000.0) == pytest.approx(0.6, abs=0.1)

    def test_monotonically_decreasing(self):
        """Toleranz muss mit steigender Frequenz monoton fallen."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        freqs = [100, 200, 350, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
        values = [resolve_jnd_tolerance_db(f) for f in freqs]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-4, (
                f"Nicht-monoton: freq={freqs[i]}/{freqs[i + 1]} val={values[i]:.3f}/{values[i + 1]:.3f}"
            )

    def test_range_always_valid(self):
        """Toleranz immer in [0.5, 3.5]."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        for freq in np.linspace(20, 20000, 200):
            v = resolve_jnd_tolerance_db(float(freq))
            assert 0.5 <= v <= 3.5, f"Toleranz außerhalb Bereich: f={freq:.0f} Hz → {v:.3f} dB"

    def test_formant_f1_region(self):
        """F1 (~500–800 Hz) → Toleranz in [1.5, 2.5]."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        for freq in [500, 600, 700, 800]:
            v = resolve_jnd_tolerance_db(float(freq))
            assert 1.3 <= v <= 2.5, f"F1-Bereich {freq} Hz: {v:.3f} dB außerhalb [1.3, 2.5]"

    def test_formant_f2_region(self):
        """F2 (~1000–2000 Hz) → Toleranz in [1.0, 1.6]."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        for freq in [1000, 1200, 1500, 1800, 2000]:
            v = resolve_jnd_tolerance_db(float(freq))
            assert 0.9 <= v <= 1.6, f"F2-Bereich {freq} Hz: {v:.3f} dB außerhalb [0.9, 1.6]"

    def test_formant_f3_f4_region(self):
        """F3/F4 (~2000–5000 Hz) → Toleranz in [0.6, 1.1]."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        for freq in [2000, 3000, 4000, 5000]:
            v = resolve_jnd_tolerance_db(float(freq))
            assert 0.6 <= v <= 1.1, f"F3/F4-Bereich {freq} Hz: {v:.3f} dB außerhalb [0.6, 1.1]"

    def test_no_nan(self):
        """Kein NaN in Output."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        for freq in [20, 100, 200, 500, 1000, 2000, 4000, 6000, 10000, 20000]:
            v = resolve_jnd_tolerance_db(float(freq))
            assert not np.isnan(v), f"NaN für freq={freq} Hz"

    def test_f1_stricter_than_f3(self):
        """F1-Toleranz > F3-Toleranz (F1 wahrnehmungsunkritischer als F3 bei Energie)."""
        from backend.core.dsp.lpc_formant_tracker import resolve_jnd_tolerance_db

        # F1 ~600 Hz → ~1.8 dB; F3 ~3000 Hz → ~0.8 dB
        v_f1 = resolve_jnd_tolerance_db(600.0)
        v_f3 = resolve_jnd_tolerance_db(3000.0)
        assert v_f1 > v_f3, f"F1 ({v_f1:.2f} dB) soll toleranter als F3 ({v_f3:.2f} dB) sein"
