"""§v10.40c (2026-09-09): GPU-Modell-Registry — Policy-Unit-Tests.

Testet verdict_for_model()/apply_gpu_policy() aus backend/core/gpu_model_registry.py
mit einer temporären JSON-Registry (kein echtes Modell nötig).
Regeln: GPU nur bei gemessenem Verdict; "cpu" erzwingt CPU auch bei GPU-Request;
CPU-only-Aufrufer (AURIK_FORCE_CPU) werden nie auf GPU gehoben.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.core.gpu_model_registry import apply_gpu_policy, load_registry, verdict_for_model

_TEST_REG = Path("/tmp/aurik_test_gpu_registry.json")


def _setup_registry(monkeypatch, entries: dict) -> None:
    _TEST_REG.write_text(json.dumps(entries), encoding="utf-8")
    import backend.core.gpu_model_registry as _mod

    monkeypatch.setattr(_mod, "_REGISTRY_PATH", _TEST_REG)
    _mod._cache = None  # Cache invalidierten


def test_verdict_unknown_when_not_scanned(monkeypatch) -> None:
    _setup_registry(monkeypatch, {"models/x/foo.onnx": {"verdict": "migraphx"}})
    assert verdict_for_model("models/y/bar.onnx") == "unknown"


def test_verdict_matched_by_filename_suffix(monkeypatch) -> None:
    _setup_registry(monkeypatch, {"models/x/foo.onnx": {"verdict": "cpu"}})
    # anderer Pfad, gleicher Dateiname → Suffix-Match
    assert verdict_for_model("/anders/verschoben/foo.onnx") == "cpu"


def test_cpu_verdict_forces_cpu_even_on_gpu_request(monkeypatch) -> None:
    _setup_registry(monkeypatch, {"models/x/foo.onnx": {"verdict": "cpu"}})
    out = apply_gpu_policy(["ROCMExecutionProvider", "CPUExecutionProvider"], "models/x/foo.onnx")
    assert out == ["CPUExecutionProvider"]


def test_migraphx_verdict_sets_migraphx_first(monkeypatch) -> None:
    _setup_registry(monkeypatch, {"models/x/foo.onnx": {"verdict": "migraphx"}})
    out = apply_gpu_policy(["ROCMExecutionProvider", "CPUExecutionProvider"], "models/x/foo.onnx")
    assert out == ["MIGraphXExecutionProvider", "CPUExecutionProvider"]


def test_rocm_verdict_downgrades_migraphx_request(monkeypatch) -> None:
    _setup_registry(monkeypatch, {"models/x/foo.onnx": {"verdict": "rocm"}})
    out = apply_gpu_policy(["MIGraphXExecutionProvider", "CPUExecutionProvider"], "models/x/foo.onnx")
    assert out == ["ROCMExecutionProvider", "CPUExecutionProvider"]


def test_cpu_only_caller_never_gets_gpu(monkeypatch) -> None:
    _setup_registry(monkeypatch, {"models/x/foo.onnx": {"verdict": "migraphx"}})
    out = apply_gpu_policy(["CPUExecutionProvider"], "models/x/foo.onnx")
    assert out == ["CPUExecutionProvider"], "AURIK_FORCE_CPU-Aufrufer dürfen nicht auf GPU gehoben werden"


def test_load_registry_missing_file_returns_empty(monkeypatch) -> None:
    import backend.core.gpu_model_registry as _mod

    monkeypatch.setattr(_mod, "_REGISTRY_PATH", Path("/tmp/does_not_exist_reg.json"))
    _mod._cache = None
    assert load_registry(force=True) == {}
