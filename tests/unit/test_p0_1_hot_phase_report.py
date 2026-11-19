"""§SOTA-P0-1 (2026-09-15) — Hot-Phase-Analyse: Tests.

Deckt ``compute_hot_phases`` (scripts/artifact_freedom_diagnosis.py):
- rt_factor = wall_ms / 1000 / input_seconds je Phase
- Hot-Phasen ab Schwelle, absteigend sortiert
- Fehler-Phasen (wall_ms None) werden übersprungen

Autor: Aurik Testing Team
"""

from scripts.artifact_freedom_diagnosis import compute_hot_phases


class TestComputeHotPhases:
    def test_rt_factor_computed_and_sorted(self):
        report = {
            "phases": [
                {"phase": "fast", "wall_ms": 200.0, "af_after": 0.9},
                {"phase": "slow", "wall_ms": 40000.0, "af_after": 0.8},
                {"phase": "mid", "wall_ms": 12000.0, "af_after": 0.85},
            ]
        }
        hot = compute_hot_phases(report, input_seconds=20.0)
        assert [h["phase"] for h in hot] == ["slow", "mid"]  # fast: 0,01× RT < 0,5
        assert hot[0]["rt_factor"] == 2.0
        assert report["phases"][0]["rt_factor"] == 0.01

    def test_error_phases_skipped(self):
        report = {"phases": [{"phase": "err", "wall_ms": None, "af_after": None}]}
        assert compute_hot_phases(report, 20.0) == []

    def test_threshold_respected(self):
        report = {"phases": [{"phase": "a", "wall_ms": 11000.0, "af_after": 0.9}]}
        assert compute_hot_phases(report, 20.0, hot_rt_threshold=0.5) == [
            {"phase": "a", "rt_factor": 0.55, "wall_ms": 11000.0}
        ]
        assert compute_hot_phases(report, 20.0, hot_rt_threshold=1.0) == []
