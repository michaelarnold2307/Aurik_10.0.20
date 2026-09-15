"""§SOTA-P1 (2026-09-15) — af-Never-worsen-Guard: Tests.

Deckt ``backend/core/dsp/artifact_freedom_guard.py``:
- identisches Signal ⇒ Passthrough (af_guard_applied=False)
- künstlicher Click im Output ⇒ proportionaler Rückblend (wet < 1)
- enabled=False / Kurzsignal ⇒ Passthrough
- Determinismus, NaN-Schutz
- Phasen-Verdrahtung: 07/17/19/38 melden ``af_guard``-Metadatum
- Diagnose-Skript: ``compute_fail_delta_violations`` (CI-Gate)

Autor: Aurik Testing Team
"""

import numpy as np

from backend.core.dsp.artifact_freedom_guard import af_fast_never_worsen, fast_af_proxy
from scripts.artifact_freedom_diagnosis import compute_fail_delta_violations


def _audio(n: int = 48000) -> np.ndarray:
    t = np.linspace(0, n / 48000, n, endpoint=False, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _click(x: np.ndarray, at: int = 20000, amp: float = 0.6) -> np.ndarray:
    out = x.copy()
    out[at] = amp
    return out.astype(np.float32)


class TestAfFastNeverWorsen:
    def test_identical_is_passthrough(self):
        x = _audio()
        out, meta = af_fast_never_worsen(x, x.copy(), 48000)
        assert np.array_equal(out, x)
        assert meta["af_guard_applied"] is False

    def test_click_degradation_blends_back(self):
        x = _audio()
        y = _click(x)
        out, meta = af_fast_never_worsen(x, y, 48000, tolerance=0.02)
        assert meta["af_guard_applied"] is True
        assert 0.0 <= meta["af_guard_wet"] < 1.0
        # Rücknahme Richtung Input: |out − x| < |y − x|
        d_out = float(np.abs(out - x).mean())
        d_y = float(np.abs(y - x).mean())
        assert d_out < d_y
        assert np.isfinite(out).all()

    def test_enabled_false_passthrough(self):
        x = _audio(8000)
        y = _click(x, at=4000)
        out, meta = af_fast_never_worsen(x, y, 48000, enabled=False)
        assert np.array_equal(out, y)
        assert meta["af_guard_applied"] is False

    def test_short_signal_passthrough(self):
        x = _audio(256)  # < 512 Samples: Detektoren nicht anwendbar
        y = _click(x, at=128)
        out, meta = af_fast_never_worsen(x, y, 48000)
        assert np.array_equal(out, y)
        assert meta["af_guard_applied"] is False

    def test_deterministic(self):
        x = _audio(24000)
        y = _click(x, at=12000)
        o1, _ = af_fast_never_worsen(x, y, 48000)
        o2, _ = af_fast_never_worsen(x, y, 48000)
        assert np.array_equal(o1, o2)

    def test_nan_input_guarded(self):
        x = _audio(16000)
        y = _click(x, at=8000)
        y[100] = np.nan
        out, _ = af_fast_never_worsen(x, y, 48000)
        assert np.isfinite(out).all()

    def test_proxy_bounded(self):
        x = _audio(16000)
        assert 0.0 <= fast_af_proxy(x, 48000) <= 1.0


class TestPhaseWiring:
    def test_phases_report_af_guard_metadata(self):
        from backend.core.defect_scanner import MaterialType

        x = _audio()
        cases = [
            ("phase_07_harmonic_restoration", "HarmonicRestorationPhase", False),
            ("phase_17_mastering_polish", "MasteringPolishPhase", True),
            ("phase_19_de_esser", "DeEsserPhase", False),
            ("phase_38_presence_boost", "PresenceBoost", False),
        ]
        import importlib

        for mod, cls, material_positional in cases:
            m = importlib.import_module(f"backend.core.phases.{mod}")
            ph = getattr(m, cls)()
            if material_positional:
                res = ph.process(x, 48000, MaterialType.VINYL)
            else:
                res = ph.process(x, sample_rate=48000, material_type=MaterialType.VINYL)
            if mod == "phase_19_de_esser" and res.metadata.get("de_essing_applied") is False:
                # Early-Exit ohne Bearbeitung (kein Sibilant im Testsignal) —
                # Guard ist dort bewusst unnötig (unverändertes Audio).
                continue
            af_meta = res.metadata.get("af_guard")
            assert af_meta is not None, f"{mod} meldet kein af_guard-Metadatum"
            assert isinstance(af_meta.get("af_guard_applied"), bool)
            assert np.isfinite(np.asarray(res.audio)).all()
            assert len(np.asarray(res.audio)) == len(x)


class TestFailDeltaGate:
    def test_no_violations(self):
        report = {"phases": [{"phase": "a", "af_delta": -0.01}, {"phase": "b", "af_delta": 0.02}]}
        assert compute_fail_delta_violations(report, 0.02) == []

    def test_violation_detected(self):
        report = {"phases": [{"phase": "a", "af_delta": -0.05}, {"phase": "b", "af_delta": None}]}
        v = compute_fail_delta_violations(report, 0.02)
        assert v == [{"phase": "a", "af_delta": -0.05}]

    def test_exact_threshold_is_ok(self):
        report = {"phases": [{"phase": "a", "af_delta": -0.02}]}
        assert compute_fail_delta_violations(report, 0.02) == []
