"""§SOTA-P1 (Residual-Artefakt-Diagnose) — Tests für das Mess-Werkzeug.

Deckt ``scripts/artifact_freedom_diagnosis.py``:
- Messpfad-Struktur: af je Phase, Delta, below_floor-Flags, Kandidatenliste
- Deterministisch; Fehler einer Phase brechen die Kette nicht (§V6-Muster)
- Real-Smoke auf einer kurzen echten Phase

Autor: Aurik Testing Team
"""

import importlib.util
import os

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "scripts.artifact_freedom_diagnosis",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts",
        "artifact_freedom_diagnosis.py",
    ),
)
assert _SPEC is not None and _SPEC.loader is not None
_diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_diag)

SR = _diag.SR


def _make_audio(n: int = SR) -> np.ndarray:
    rng = np.random.default_rng(5)
    t = np.linspace(0, n / SR, n, endpoint=False, dtype=np.float32)
    x = 0.2 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 0.005, n).astype(np.float32)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.5).astype(np.float32)


class TestDiagnosisLogic:
    def test_report_structure_with_one_real_phase(self, monkeypatch):
        # Messpfad mit EINER echten Phase (schnell) statt der vollen Kette.
        monkeypatch.setattr(
            _diag,
            "PHASES",
            [("phase_16_final_eq", "phase_16_final_eq", "FinalEQ")],
        )
        report = _diag.run_diagnosis(_make_audio())
        assert report["af_floor"] == 0.95
        assert report["input_af"] is not None
        assert len(report["phases"]) == 1
        entry = report["phases"][0]
        assert entry["phase"] == "phase_16_final_eq"
        assert 0.0 <= entry["af_after"] <= 1.0
        assert entry["af_delta"] is not None
        assert entry["wall_ms"] >= 0.0
        assert entry["below_floor"] is (entry["af_after"] < 0.95)
        assert report["min_af"] == entry["af_after"]
        assert report["worst_phase"] == "phase_16_final_eq"

    def test_phase_error_does_not_break_chain(self, monkeypatch):
        monkeypatch.setattr(
            _diag,
            "PHASES",
            [
                ("phase_broken", "definitely_missing_module", "Nope"),
                ("phase_16_final_eq", "phase_16_final_eq", "FinalEQ"),
            ],
        )
        report = _diag.run_diagnosis(_make_audio())
        assert report["phases"][0]["error"] is not None
        assert report["phases"][0]["af_after"] is None
        assert report["phases"][1]["af_after"] is not None
        assert report["min_af"] == report["phases"][1]["af_after"]

    def test_deterministic(self, monkeypatch):
        monkeypatch.setattr(
            _diag,
            "PHASES",
            [("phase_16_final_eq", "phase_16_final_eq", "FinalEQ")],
        )
        audio = _make_audio()
        r1 = _diag.run_diagnosis(audio)
        r2 = _diag.run_diagnosis(audio)
        assert r1["input_af"] == r2["input_af"]
        assert r1["phases"][0]["af_after"] == r2["phases"][0]["af_after"]

    def test_measure_af_range(self):
        af = _diag.measure_af(_make_audio(SR // 2))
        assert 0.0 <= af <= 1.0
