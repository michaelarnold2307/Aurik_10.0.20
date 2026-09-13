"""§SOTA-BSR-GPU-S1 — ONNX-Softmax-Rewrite für ROCm (Numerik-Fix).

Befund 2026-09-13 (scripts/diagnose_bsr_rocm_numerics.py): Der
ROCmExecutionProvider rechnet Softmax im MelBandRoformer falsch (rel ≈ 3,5 an
der ersten Attention — alles danach erbt den Fehler; Gesamtmodell rel=6,1).
Dieses Modul ersetzt jeden Softmax-Knoten durch numerisch äquivalente
Primitive, deren ROCm-Kernels korrekt sind:

    m = ReduceMax(x, axis, keepdims=1)
    y = Sub(x, m)
    e = Exp(y)
    s = ReduceSum(e, axis, keepdims=1)
    out = Div(e, s)

Ergebnis wird neben dem Original als `<name>.rocm_safe.onnx` gecacht
(deterministisch; unverändert, solange die Quelle unverändert ist).
Fail-closed: Jeder Fehler wirft — der Aufrufer entscheidet über den Fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import helper

logger = logging.getLogger(__name__)

_CACHE_SUFFIX = ".rocm_safe.onnx"


def _softmax_axis(node, default_axis: int) -> int:
    for attr in node.attribute:
        if attr.name == "axis":
            return int(attr.i)
    return default_axis


def _rewrite_graph(graph, default_axis: int, opset_version: int) -> int:
    """Ersetzt Softmax-Knoten in-place durch Primitive. Gibt die Anzahl zurück."""
    new_nodes = []
    new_inits = []
    count = 0
    for node in graph.node:
        if node.op_type != "Softmax":
            new_nodes.append(node)
            continue
        count += 1
        x = node.input[0]
        out = node.output[0]
        axis = _softmax_axis(node, default_axis)
        stem = f"{out}__rocm_safe"
        # axes: ReduceSum seit Opset 13 als INPUT (kein Attribut mehr);
        # ReduceMax-Attribut gilt bis Opset 17. Ab 18: beide als Input.
        if opset_version >= 18:
            axes_name = f"{stem}_axes"
            new_inits.append(onnx.numpy_helper.from_array(np.array([axis], dtype=np.int64), name=axes_name))
            m = helper.make_node("ReduceMax", [x, axes_name], [f"{stem}_max"], keepdims=1)
            s = helper.make_node("ReduceSum", [f"{stem}_exp", axes_name], [f"{stem}_sum"], keepdims=1)
        elif opset_version >= 13:
            axes_name = f"{stem}_axes"
            new_inits.append(onnx.numpy_helper.from_array(np.array([axis], dtype=np.int64), name=axes_name))
            m = helper.make_node("ReduceMax", [x], [f"{stem}_max"], axes=[axis], keepdims=1)
            s = helper.make_node("ReduceSum", [f"{stem}_exp", axes_name], [f"{stem}_sum"], keepdims=1)
        else:
            m = helper.make_node("ReduceMax", [x], [f"{stem}_max"], axes=[axis], keepdims=1)
            s = helper.make_node("ReduceSum", [f"{stem}_exp"], [f"{stem}_sum"], axes=[axis], keepdims=1)
        y = helper.make_node("Sub", [x, f"{stem}_max"], [f"{stem}_sub"])
        e = helper.make_node("Exp", [f"{stem}_sub"], [f"{stem}_exp"])
        d = helper.make_node("Div", [f"{stem}_exp", f"{stem}_sum"], [out])
        new_nodes.extend([m, y, e, s, d])
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_inits)
    return count


def rewrite_softmax_to_primitives(model_path: Path) -> Path:
    """Schreibt die ROCm-sichere Variante (gecacht) und gibt ihren Pfad zurück."""
    src = Path(model_path)
    cache = src.with_name(src.name + _CACHE_SUFFIX)
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return cache
    model = onnx.load(str(src))
    _default_axis = -1  # Opset ≥ 13
    _opset = 13
    for _ops in model.opset_import:
        if _ops.domain in ("", "ai.onnx"):
            _opset = _ops.version
            if _opset < 13:
                _default_axis = 1
    count = _rewrite_graph(model.graph, _default_axis, _opset)
    if count == 0:
        return src
    onnx.save(model, str(cache))
    logger.info("§BSR-GPU-S1 Softmax-Rewrite: %d Knoten ersetzt → %s", count, cache)
    return cache
