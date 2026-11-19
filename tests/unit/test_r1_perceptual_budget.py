"""§SOTA-R1 (Wahrnehmungs-Budget-Bilanz) — Tests des JND-Budget-Moduls.

Deckt ``backend/core/dsp/perceptual_budget.py``:
- identische Signale ⇒ 0 JND-Einheiten, keine hörbaren Änderungen
- Pegel-/Lautheits-/IACC-/Centroid-Deltas in JND-Einheiten
- Ketten-Summenbericht (Wahrnehmungs-Wasserzeichen)
- Determinismus, NaN-Schutz

Autor: Aurik Testing Team
"""

import numpy as np

from backend.core.dsp.perceptual_budget import (
    measure_perceptual_budget,
    summarize_budget,
)


def _mono(n: int = 48000, f: float = 440.0, amp: float = 0.2) -> np.ndarray:
    t = np.linspace(0, n / 48000, n, endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _stereo(n: int = 48000) -> np.ndarray:
    left = _mono(n)
    right = np.roll(left, 3) * 0.9
    return np.column_stack([left, right]).astype(np.float32)


class TestMeasurePerceptualBudget:
    def test_identical_is_zero_budget(self):
        x = _mono()
        rep = measure_perceptual_budget(x, x.copy(), 48000, phase="identisch")
        assert rep.total_jnd_units == 0.0
        assert not any(rep.audible_changes.values())

    def test_level_delta_in_jnd_units(self):
        x = _mono(amp=0.2)
        y = _mono(amp=0.4)  # +6 dB
        rep = measure_perceptual_budget(x, y, 48000, phase="gain")
        assert np.isclose(rep.jnd_units["level_broadband"], 6.0, atol=0.05)
        assert rep.jnd_units["level_broadband"] > 1.0
        assert rep.audible_changes["level_broadband"] is True

    def test_loudness_delta_reported(self):
        x = _mono(amp=0.1)
        y = _mono(amp=0.4)
        rep = measure_perceptual_budget(x, y, 48000)
        assert "loudness_ratio" in rep.jnd_units
        assert rep.jnd_units["loudness_ratio"] > 1.0

    def test_stereo_collapse_raises_iacc_budget(self):
        stereo = _stereo()
        mono_collapsed = np.column_stack([_mono(), _mono()]).astype(np.float32)  # L==R ⇒ IACC=1
        rep = measure_perceptual_budget(stereo, mono_collapsed, 48000)
        assert "iacc" in rep.jnd_units
        assert rep.jnd_units["iacc"] > 0.0

    def test_deterministic(self):
        x = _mono(24000)
        y = _mono(24000, f=880.0, amp=0.3)
        r1 = measure_perceptual_budget(x, y, 48000)
        r2 = measure_perceptual_budget(x, y, 48000)
        assert r1.as_dict() == r2.as_dict()

    def test_nan_input_guarded(self):
        x = _mono(8000)
        y = _mono(8000).copy()
        y[100] = np.nan
        rep = measure_perceptual_budget(x, y, 48000)
        assert np.isfinite(rep.total_jnd_units)


class TestSummarizeBudget:
    def test_summary_watermark(self):
        x = _mono(16000)
        reports = [
            measure_perceptual_budget(x, x * 1.2, 48000, phase="phase_a"),
            measure_perceptual_budget(x, x * 2.0, 48000, phase="phase_b"),
        ]
        summary = summarize_budget(reports)
        assert summary["n_reports"] == 2
        assert summary["total_jnd_units"] > 0.0
        assert summary["per_kind_jnd_units"]["level_broadband"] > 1.0
        assert summary["top_consumers"]["level_broadband"]["phase"] == "phase_b"

    def test_empty_reports(self):
        summary = summarize_budget([])
        assert summary["n_reports"] == 0
        assert summary["total_jnd_units"] == 0.0
