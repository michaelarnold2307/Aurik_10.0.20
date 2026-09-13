"""§SOTA-BSR-GPU-S1 — ONNX-Softmax-Rewrite: numerische Äquivalenz (CPU-Referenz).

Der Rewrite ersetzt Softmax durch ReduceMax/Sub/Exp/ReduceSum/Div — numerisch
gleichwertig, aber ohne den defekten ROCm-Softmax-Kernel.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from backend.core.dsp.onnx_softmax_rewrite import rewrite_softmax_to_primitives


def _build_softmax_model(tmp_path, opset: int, axis_attr: bool):
    nodes = [helper.make_node("Softmax", ["X"], ["Y"], **({"axis": 1} if axis_attr else {}))]
    graph = helper.make_graph(
        nodes,
        "softmax_test",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    path = tmp_path / "softmax_model.onnx"
    onnx.save(model, str(path))
    return path


@pytest.mark.parametrize("opset,axis_attr", [(13, False), (13, True), (11, False)])
def test_rewrite_numerically_equivalent(tmp_path, opset, axis_attr):
    src = _build_softmax_model(tmp_path, opset, axis_attr)
    rewritten = rewrite_softmax_to_primitives(src)

    x = np.array([[1.0, 2.0, 3.0], [-10.0, 0.0, 10.0]], dtype=np.float32)
    sess_orig = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(str(rewritten), providers=["CPUExecutionProvider"])
    y_orig = sess_orig.run(None, {"X": x})[0]
    y_new = sess_new.run(None, {"X": x})[0]
    assert np.max(np.abs(y_orig - y_new)) < 1e-5
    # Softmax-Summen-Eigenschaft bleibt erhalten
    assert np.allclose(y_new.sum(axis=1), 1.0, atol=1e-5)


def test_rewrite_removes_softmax_nodes(tmp_path):
    src = _build_softmax_model(tmp_path, 13, False)
    rewritten = rewrite_softmax_to_primitives(src)
    model = onnx.load(str(rewritten))
    assert not any(n.op_type == "Softmax" for n in model.graph.node)
    ops = {n.op_type for n in model.graph.node}
    assert {"ReduceMax", "Sub", "Exp", "ReduceSum", "Div"} <= ops


def test_no_softmax_returns_source(tmp_path):
    # Modell ohne Softmax → Rewrite gibt den Quellpfad unverändert zurück
    nodes = [helper.make_node("Add", ["X", "X"], ["Y"])]
    graph = helper.make_graph(
        nodes,
        "add_test",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    src = tmp_path / "add_model.onnx"
    onnx.save(model, str(src))
    assert rewrite_softmax_to_primitives(src) == src


def test_cache_reused(tmp_path):
    src = _build_softmax_model(tmp_path, 13, False)
    r1 = rewrite_softmax_to_primitives(src)
    r2 = rewrite_softmax_to_primitives(src)
    assert r1 == r2
    assert r1.exists()
