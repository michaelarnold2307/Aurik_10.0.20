"""Tests fuer den ORT-Inference-Session-Cache (TODO-P0-2, backend/core/onnx/runtime.py).

Normativ (per Docstring-Referenz, §G9 (GEBOTE.md)): Die ORT-Graph-Optimierung muß VOR
`InferenceSession()` dauerhaft nach `output/onnx_session_cache/<stem>.<key>.ort`
serialisiert werden — der Compile-Cache ist ueber Inkarnationen, Prozesse und
Terminate-Populationen hinweg persistent (genau der T1-DestroyedSession-
Rollback-Fail-Modus, der den harten Performance-Gap in §G5.txt schloss).
Determinismus §G5 (GEBOTE.md): gleicher Input + Version => bit-identischer Output; die
Serialisierung darf daran nichts aendern. Paritaet wird hier EVIDENZIERT
(bit-identisch ueber uncached/warm/reload), inkl. Cache-Key-Invaldierung bei
Modellaenderung und Kill-Switch `AURIK_ORT_CACHE=0` (§V6 (copilot-instructions.md): unveraendertes
bisheriges Verhalten).

Beweisschluss: `pytest tests/unit/test_ort_session_cache.py`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")

from backend.core.onnx.runtime import (
    _session_cache_key,
    apply_session_cache,
    create_inference_session,
)

_CPU = ["CPUExecutionProvider"]


def _export_tiny_linear(path: Path) -> None:
    """Winziges deterministisches ONNX-Modell (Linear 4->3) als CPU-Testanker."""
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight[:] = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        model.bias[:] = torch.tensor([0.5, -0.5, 1.0])
    torch.onnx.export(
        model,
        torch.zeros(1, 4),
        str(path),
        opset_version=17,
        input_names=["in"],
        output_names=["out"],
        dynamo=False,
    )


def test_cache_key_determinism_and_invalidation(tmp_path: Path) -> None:
    m = tmp_path / "m.onnx"
    _export_tiny_linear(m)
    key1 = _session_cache_key(m, _CPU)
    key2 = _session_cache_key(m, _CPU)
    assert key1 == key2 and len(key1) == 24  # deterministisch (§G5 (GEBOTE.md))
    time.sleep(0.02)
    m.write_bytes(m.read_bytes() + b"\x00")  # size+mtime-Aenderung -> neuer Key
    assert _session_cache_key(m, _CPU) != key1  # invalidation on change
    assert _session_cache_key(m, ["CUDAExecutionProvider"]) != key2  # provider-faehig


def test_apply_cache_path_and_killswitch(tmp_path: Path, monkeypatch) -> None:
    m = tmp_path / "m.onnx"
    _export_tiny_linear(m)
    monkeypatch.chdir(tmp_path)  # isoliertes output/onnx_session_cache
    opts = ort.SessionOptions()
    assert apply_session_cache(opts, m, _CPU) is True
    cached = Path(opts.optimized_model_filepath)
    assert cached.parent == Path("output") / "onnx_session_cache"
    assert cached.name.startswith("m.") and cached.suffix == ".ort"
    monkeypatch.setenv("AURIK_ORT_CACHE", "0")  # Kill-Switch
    opts2 = ort.SessionOptions()
    assert apply_session_cache(opts2, m, _CPU) is False
    assert opts2.optimized_model_filepath == ""  # unberuehrt (§V6 (copilot-instructions.md))


def test_persisted_ort_cache_bit_identical(tmp_path: Path, monkeypatch) -> None:
    """§G5-Kern: Serialisierung ist ueber Instanzen hinweg persistent + par."""
    m = tmp_path / "m.onnx"
    _export_tiny_linear(m)
    monkeypatch.chdir(tmp_path)
    x = np.array([[1.0, -2.0, 3.5, 0.25]], dtype=np.float32)

    o_nocache = ort.SessionOptions()
    o_nocache.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s0 = ort.InferenceSession(str(m), o_nocache, providers=_CPU)
    y0 = s0.run(None, {"in": x})[0]
    del s0

    o_warm = ort.SessionOptions()
    o_warm.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert apply_session_cache(o_warm, m, _CPU) is True
    s1 = ort.InferenceSession(str(m), o_warm, providers=_CPU)
    y1 = s1.run(None, {"in": x})[0]
    del s1  # Rollback-Pfad wie T1: Session darf sterben, Cache bleibt

    cached = Path(o_warm.optimized_model_filepath)
    assert cached.is_file() and cached.stat().st_size > 0  # serialisiert + persistent
    o_reload = ort.SessionOptions()
    o_reload.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert apply_session_cache(o_reload, m, _CPU) is True
    assert Path(o_reload.optimized_model_filepath) == cached  # gleicher Persist
    s2 = ort.InferenceSession(str(m), o_reload, providers=_CPU)
    y2 = s2.run(None, {"in": x})[0]
    del s2

    assert np.array_equal(y0, y1) and np.array_equal(y1, y2)  # bit-identisch (§G5 (GEBOTE.md))


def test_create_inference_session_with_cache(tmp_path: Path, monkeypatch) -> None:
    """Zentraler Session-Builder (create_inference_session) inkl. Cache."""
    m = tmp_path / "m.onnx"
    _export_tiny_linear(m)
    monkeypatch.chdir(tmp_path)
    s = create_inference_session(m, providers=_CPU)
    y = s.run(None, {"in": np.zeros((1, 4), dtype=np.float32)})[0]
    assert y.shape == (1, 3)
    del s
