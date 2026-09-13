#!/usr/bin/env python3
"""§SOTA-BSR-GPU-S1 — ROCm-Numerik-Bisektion (Diagnose, keine Verarbeitung).

Lädt ein ONNX-Modell EINMAL und hängt N gleichmäßig verteilte Zwischenknoten
als zusätzliche Ausgänge an. Ein Forward-Lauf auf CPU und ROCm liefert dann
den relativen Fehler je Probe → der Knoten, an dem die Abweichung explodiert,
ist der numerisch defekte EP-Kernel. Nur Diagnose; schreibt nichts ins Repo.

Usage:
  .venv_aurik/bin/python scripts/diagnose_bsr_rocm_numerics.py \
      --model models/melbandroformer/melbandroformer_optimized.onnx \
      --probes 10 [--round2 <start_node> <end_node>]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import onnx
import onnxruntime as ort


def _dummy_inputs(sess) -> dict:
    rng = np.random.default_rng(0)
    d: dict = {}
    for inp in sess.get_inputs():
        shape = [max(1, int(x) if isinstance(x, int) else 8) for x in inp.shape]
        d[inp.name] = rng.uniform(-1.0, 1.0, tuple(shape)).astype(np.float32)
    return d


def _probe_nodes(nodes, start: int, end: int, n_probes: int) -> list[tuple[int, str, str]]:
    idxs = sorted({int(start + i * (end - start) / max(1, n_probes - 1)) for i in range(n_probes)})
    probes = []
    for i in idxs:
        n = nodes[i]
        if n.output and n.op_type != "Constant":
            probes.append((i, n.op_type, n.output[0]))
    return probes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    m = onnx.load(args.model)
    # Shape-Inferenz liefert die Element-Typen der Zwischentensoren (z. B. int64 bei Shape).
    try:
        m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Shape-Inferenz fehlgeschlagen (Typen werden als float angenommen): {type(exc).__name__}")
    _type_map: dict[str, object] = {}
    for _vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        _type_map[_vi.name] = _vi.type
    nodes = list(m.graph.node)
    end = args.end if args.end >= 0 else len(nodes) - 1
    end = min(end, len(nodes) - 1)
    n_orig_out = len(m.graph.output)
    probes = _probe_nodes(nodes, args.start, end, args.probes)
    if not probes:
        print("Keine Probe-Knoten gefunden.")
        return 2

    for _i, _op, name in probes:
        _t = _type_map.get(name)
        _elem = _t.tensor_type.elem_type if _t is not None and _t.HasField("tensor_type") else onnx.TensorProto.FLOAT
        m.graph.output.append(onnx.helper.make_tensor_value_info(name, _elem, None))
    tmp = args.model + ".probe.onnx"
    onnx.save(m, tmp)

    sess_cpu = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    sess_roc = ort.InferenceSession(tmp, providers=["ROCMExecutionProvider", "CPUExecutionProvider"])
    print(f"ROCm-EP platziert: {sess_roc.get_providers()[0]}")
    inputs = _dummy_inputs(sess_cpu)
    outs_cpu = sess_cpu.run(None, inputs)
    outs_roc = sess_roc.run(None, inputs)

    results = []
    for k, (_i, _op, _name) in enumerate(probes):
        a = np.asarray(outs_cpu[n_orig_out + k])
        b = np.asarray(outs_roc[n_orig_out + k])
        if a.dtype != np.float32:
            continue
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        scale = max(float(np.max(np.abs(a))), 1e-9)
        rel = float(np.max(np.abs(a - b))) / scale
        results.append({"node_idx": _i, "op_type": _op, "rel_err": rel, "shape": list(a.shape)})

    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    worst = max(results, key=lambda r: r["rel_err"])
    print(f"WORST: node {worst['node_idx']} ({worst['op_type']}) rel={worst['rel_err']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
