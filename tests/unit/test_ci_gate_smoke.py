"""CI-Gate Smoke Test — End-to-End Pipeline (§Schutzschicht-3).

Hinweis (Befund 2026-10-05): Diese drei Smoke-Tests fahren echte End-to-End-Pfade
(Regression-Gate-Baseline, `aurik_pipeline(use_real=True)`, Competitive-Benchmark)
und brauchen ~60 s je Test — das CLI-Timeout des Chunk-Smokes (20–25 s) ist damit
IMMER zu klein. Sichtbar wurde das erst im vollständigen Scan (alle 65 Chunks); der
Commit-Hook prüft nur Chunk 1 und erreicht diese Datei nie. Der Modul-Marker hebt
das CLI-Default deterministisch an (Tests dürfen nicht last-/zeitabhängig kippen,
§G5 (GEBOTE.md) / tests.instructions.md).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.timeout(600)


def test_ci_gate_regression_baseline_runs():
    """Regression Gate läuft und produziert Baseline."""
    from benchmarks.regression.regression_gate import generate_baseline

    bl = generate_baseline(0.3)
    for name, r in bl.scenarios.items():
        assert r["pqs"] > 0, f"{name}: PQS <= 0"
    assert len(bl.scenarios) >= 4


def test_ci_gate_mini_pipeline_no_nan():
    """Mini-Pipeline produziert kein NaN."""
    from benchmarks.regression.regression_gate import _make_music, _make_noisy, aurik_pipeline

    music = _make_music(0.3)
    noisy = _make_noisy(music, 15.0)
    result = aurik_pipeline(noisy, 48000, use_real=True, full=False)
    assert np.all(np.isfinite(result))


def test_ci_gate_open_source_benchmark_runs():
    """Competitive Benchmark importiert und läuft."""
    from benchmarks.competitive.open_source_benchmark import run

    results, summary = run(["scipy_wiener"], dur=0.3)
    assert summary["ok"] > 0
