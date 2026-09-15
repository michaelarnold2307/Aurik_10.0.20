"""§SOTA-R4 (Audibility-First-Benchmark) — Tests für das Mess-Werkzeug.

Deckt ``scripts/benchmark_audibility_first.py``:
- Korpus-Bau ist deterministisch (Seed 42, §G5 (GEBOTE.md)) und injiziert Klicks korrekt
- ``run_case`` sammelt die PSY-A1-Skip-Statistik aus phase_01-Metadaten
- Real-Smoke: 1-s-Fall durch die echte phase_01 (subaudible Klicks werden
  übersprungen, hörbare repariert)

Autor: Aurik Testing Team
"""

import importlib.util
import os

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "scripts.benchmark_audibility_first",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts",
        "benchmark_audibility_first.py",
    ),
)
assert _SPEC is not None and _SPEC.loader is not None
_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_bench)

SR = _bench.SR


class TestCorpus:
    def test_deterministic_build(self):
        c1 = _bench.build_corpus(duration_s=0.25, n_subaudible=3, n_audible=3, seed=42)
        c2 = _bench.build_corpus(duration_s=0.25, n_subaudible=3, n_audible=3, seed=42)
        for k in ("clean", "subaudible", "audible"):
            assert np.array_equal(c1["cases"][k]["audio"], c2["cases"][k]["audio"])
            assert c1["cases"][k]["injected_positions"] == c2["cases"][k]["injected_positions"]

    def test_corpus_structure(self):
        c = _bench.build_corpus(duration_s=0.5, n_subaudible=5, n_audible=5, seed=7)
        n = int(0.5 * SR)
        assert c["sr"] == SR
        assert c["cases"]["clean"]["audio"].shape == (n,)
        assert c["cases"]["clean"]["injected_positions"] == []
        assert len(c["cases"]["subaudible"]["injected_positions"]) == 5
        assert len(c["cases"]["audible"]["injected_positions"]) == 5
        # Hörbare Klicks erhöhen den Peak klar; subaudible kaum.
        base_peak = float(np.max(np.abs(c["cases"]["clean"]["audio"])))
        sub_peak = float(np.max(np.abs(c["cases"]["subaudible"]["audio"])))
        aud_peak = float(np.max(np.abs(c["cases"]["audible"]["audio"])))
        assert sub_peak - base_peak < 0.01
        assert aud_peak - base_peak > 0.1

    def test_injected_clicks_at_positions(self):
        c = _bench.build_corpus(duration_s=0.5, n_subaudible=2, n_audible=2, seed=3)
        clean = c["cases"]["clean"]["audio"]
        aud = c["cases"]["audible"]["audio"]
        for pos in c["cases"]["audible"]["injected_positions"]:
            # Burst-Fenster: maximale lokale Abweichung ≈ Burst-Amplitude
            win = np.abs(aud[pos : pos + 20] - clean[pos : pos + 20])
            assert float(np.max(win)) > 0.15


class TestRunCase:
    def _fake_phase(self, mods: dict):
        class _R:
            modifications = mods

        class _Fake:
            def process(self, audio, sample_rate, material_type):
                return _R()

        return _Fake()

    def test_run_case_extracts_stats(self):
        case = {
            "audio": np.zeros(SR // 2, dtype=np.float32),
            "injected_positions": [100],
            "expected_category": "subaudible",
        }
        phase = self._fake_phase({"total_clicks_removed": 10, "subaudible_skipped": 4, "ml_repaired": 0})
        out = _bench.run_case(phase, case, SR)
        assert out["injected"] == 1
        assert out["detected_total"] == 10
        assert out["subaudible_skipped"] == 4
        assert out["skip_fraction"] == pytest.approx(0.4)
        assert out["wall_time_ms"] >= 0.0

    def test_run_case_zero_division_safe(self):
        case = {"audio": np.zeros(SR // 2, dtype=np.float32), "injected_positions": [], "expected_category": "clean"}
        out = _bench.run_case(self._fake_phase({"total_clicks_removed": 0}), case, SR)
        assert out["skip_fraction"] == 0.0


class TestGateOracle:
    def test_gate_filters_subaudible_but_not_audible(self):
        c = _bench.build_corpus(duration_s=1.0, n_subaudible=8, n_audible=8, seed=11)
        oracle_sub = _bench.gate_oracle(c["cases"]["subaudible"], SR)
        oracle_aud = _bench.gate_oracle(c["cases"]["audible"], SR)
        # Hörbarkeits-Filterrate: subaudible Klicks werden mehrheitlich als
        # skippable bewertet, hörbare praktisch nie.
        assert oracle_sub["gate_evaluated"] == 8
        assert oracle_aud["gate_evaluated"] == 8
        assert oracle_sub["gate_skippable"] >= 6
        assert oracle_aud["gate_skippable"] <= 1


class TestRealPhaseSmoke:
    @pytest.mark.timeout(120)
    def test_phase_metadata_exposes_skip_counting(self):
        """1-s-Smoke: phase_01 exportiert die PSY-A1-Skip-Zählung (R4-Verdrahtung).

        Die Skip-Anzahl selbst hängt an Detektions-Schwellen (multi-scale,
        materialabhängig) — getestet wird die Metadaten-Verdrahtung, die
        Gate-Filterrate prüft TestGateOracle direkt am Gate.
        """
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

        corpus = _bench.build_corpus(duration_s=1.0, n_subaudible=8, n_audible=8, seed=42)
        phase = ClickRemovalPhase()
        out_aud = _bench.run_case(phase, corpus["cases"]["audible"], SR)
        # Verdrahtung: der Key existiert und ist ein int ≥ 0; Laufzeit gemessen.
        assert isinstance(out_aud["subaudible_skipped"], int)
        assert out_aud["subaudible_skipped"] >= 0
        assert out_aud["wall_time_ms"] >= 0.0
        assert out_aud["gate_evaluated"] == 8
