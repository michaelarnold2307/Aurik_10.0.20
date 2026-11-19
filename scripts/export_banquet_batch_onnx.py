"""
BANQUET-Batch-Re-Export (SOTA-ML-V5) — batch-generische Variante von banquet_vinyl_final.onnx.

Kontext
-------
`models/banquet/banquet_vinyl_final.onnx` ist per Konstruktion batch-1-spezifisch
(Beleg: TODOS_SOTA_ROADMAP.md, TODO SOTA-ML-V5). Die Batch-Dim ist statisch
``[1,128,128,128]`` und wird an den 24 ``seqband``-Modulgrenzen mit der Band-Dim
verschmolzen (``1 x 128 = 128``); drei geteilte Shape-Konstanten und 48
h0/c0-Nullkonstanten tragen feste Werte. Ein Re-Export aus dem PyTorch-Training
ist nicht möglich (Trainings-Code der 2023er-Vinyl-Variante extern gebunden,
siehe Roadmap), daher wird der Graph hier chirurgisch batch-generisch gemacht.

Warum das bit-erhaltend ist
---------------------------
Der Graph besteht aus 24 identischen ``seqband``-Modulen:

    LayerNorm(feat) -> [B,128,128,128]
    Reshape (B*128, 128, 128)   # Batch mit Band-Dim verschmelzen (val_6)
    Transpose (1,0,2)           # Frames als Sequenz, B*128 Baender als Batch
    LSTM(bidir, 256, layout=0)  # [128, B*128, 128] -> [128, 2, B*128, 256]
    Transpose (0,2,1,3) -> Reshape (128, B*128, 512)   # (val_133)
    Transpose (1,0,2) -> Reshape (B, 128, 128, 512)    # (val_140)
    MatMul fc (512->128) + bias + Residual-Add

Die LSTM-Batch-Achse ist semantisch frei: jedes (Fenster, Band)-Paar ist eine
unabhängige Sequenz über die 128 Frames. B*128 Batch-Elemente sind exakt
dieselbe Rechnung wie B Einzelläufe — kein Cross-Window-State (LSTM-Batch-
Elemente sind per ONNX-Spec unabhängig). Alle Slices im Graphen betreffen nur
Gewichte/Nullen (kein Slice auf Daten-Tensoren, statisch verifiziert).

Aenderungen (nur Shape-Ops):
1. Input-/Output-Dim 0 wird symbolisch (``batch``).
2. val_6   (128,128,128)   -> [B*128, 128, 128]   (berechnet)
3. val_133 (128,128,512)   -> [128, B*128, 512]   (berechnet)
4. val_140 (1,128,128,512) -> [B, 128, 128, 512]  (berechnet)
5. h0/c0 je LSTM: ConstantOfShape([2, B*128, 256], 0) statt konstanter Nullen.

Verifikation (im Skript, §G5 (copilot-instructions.md)):
- B=1: bit-identisch zum Original-Modell (max|Δ| == 0).
- B=2/3: out[i] == B=1-Lauf des jeweiligen Fensters (Unabhängigkeitsbeweis,
  Toleranz 1e-6 relativ wegen GEMM-Blocking).

Usage:
    .venv_aurik/bin/python scripts/export_banquet_batch_onnx.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SRC = Path("models/banquet/banquet_vinyl_final.onnx")
DST = Path("models/banquet/banquet_vinyl_batch.onnx")

_CONST_MERGE = "val_6"  # (128,128,128) — Batch×Bänder verschmelzen
_CONST_DIRM = "val_133"  # (128,128,512) — Richtungs-Merge
_CONST_UNM = "val_140"  # (1,128,128,512) — Batch×Bänder trennen

_NEW_NODE = "batchgen_"


def _make_int64(name: str, values: list[int]) -> TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def build_batch_generic(src: Path, dst: Path) -> None:
    m = onnx.load(str(src))
    g = m.graph

    # 1) Symbolische Batch-Dim an Ein- und Ausgang.
    for vi in list(g.input) + list(g.output):
        d0 = vi.type.tensor_type.shape.dim[0]
        d0.dim_param = "batch"
        d0.ClearField("dim_value")
    # Stale value_info entfernen (harte batch-1-Hints wuerden ORT verwirren).
    del g.value_info[:]

    # 2-4) Berechnete Shape-Tensoren (int64).
    #      batch   = Shape(input)[0]
    #      b128    = batch * 128
    #      Die neuen Knoten werden VORNE eingefuegt (topologisch vor den
    #      Reshape/LSTM-Konsumenten), damit der Graph sortiert bleibt.
    _pre_nodes: list = []
    shp_in = _NEW_NODE + "shape_in"
    _pre_nodes.append(helper.make_node("Shape", ["input_fixed"], [shp_in], name=_NEW_NODE + "shape"))
    const0 = _NEW_NODE + "c0"
    g.initializer.append(_make_int64(const0, [0]))
    b_dim = _NEW_NODE + "b_dim"
    _pre_nodes.append(helper.make_node("Gather", [shp_in, const0], [b_dim], axis=0, name=_NEW_NODE + "gather"))
    const128 = _NEW_NODE + "c128"
    g.initializer.append(_make_int64(const128, [128]))
    b128_dim = _NEW_NODE + "b128"
    _pre_nodes.append(helper.make_node("Mul", [b_dim, const128], [b128_dim], name=_NEW_NODE + "mul"))

    def _concat_shape(out: str, parts: list[object]) -> str:
        consts = []
        in_names = []
        for i, p in enumerate(parts):
            if isinstance(p, str):  # Symbolname
                in_names.append(p)
            else:
                cname = f"{out}_c{i}"
                g.initializer.append(_make_int64(cname, [int(p)]))
                consts.append(cname)
                in_names.append(cname)
        _pre_nodes.append(helper.make_node("Concat", in_names, [out], axis=0, name=out))
        return out

    t_merge = _concat_shape(_NEW_NODE + "t_merge", [b128_dim, 128, 128])
    t_dirm = _concat_shape(_NEW_NODE + "t_dirm", [128, b128_dim, 512])
    t_unm = _concat_shape(_NEW_NODE + "t_unm", [b_dim, 128, 128, 512])

    # 5) Dynamische h0/c0-Nullen [2, B*128, 256].
    h_shape = _concat_shape(_NEW_NODE + "h_shape", [2, b128_dim, 256])
    h0_dyn = _NEW_NODE + "h0_state"
    c0_dyn = _NEW_NODE + "c0_state"
    for out in (h0_dyn, c0_dyn):
        _pre_nodes.append(
            helper.make_node(
                "ConstantOfShape",
                [h_shape],
                [out],
                value=helper.make_tensor("fill", TensorProto.FLOAT, [1], [0.0]),
                name=out,
            )
        )
    # Neue Knoten vorne einfügen (RepeatedCompositeContainer unterstützt
    # kein Slice-Assignment → clear + extend in Zielreihenfolge).
    _ordered = _pre_nodes + list(g.node)
    del g.node[:]
    g.node.extend(_ordered)

    # Shape-Ziele umhaengen.
    n_reshape_fixed = 0
    for n in g.node:
        if n.op_type == "Reshape" and len(n.input) >= 2:
            if n.input[1] == _CONST_MERGE:
                n.input[1] = t_merge
                n_reshape_fixed += 1
            elif n.input[1] == _CONST_DIRM:
                n.input[1] = t_dirm
                n_reshape_fixed += 1
            elif n.input[1] == _CONST_UNM:
                n.input[1] = t_unm
                n_reshape_fixed += 1

    # h0/c0 der LSTMs umhaengen; tote Slice-Knoten entfernen.
    old_h_names = set()
    for n in g.node:
        if n.op_type == "LSTM":
            old_h_names.add(n.input[5])
            old_h_names.add(n.input[6])
            n.input[5] = h0_dyn
            n.input[6] = c0_dyn
    dead = [n for n in g.node if n.op_type == "Slice" and all(o in old_h_names for o in n.output)]
    for n in dead:
        g.node.remove(n)

    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    logger.info(
        "Batch-generischer Graph geschrieben: %s (%d Reshape-Ziele umgehaengt, %d tote h0/c0-Slices entfernt, %d LSTM-Knoten)",
        dst,
        n_reshape_fixed,
        len(dead),
        sum(1 for n in g.node if n.op_type == "LSTM"),
    )


def _run(model_path: Path, feeds: dict[str, np.ndarray], opt_level: str = "default") -> np.ndarray:
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    if opt_level == "off":
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_opts, providers=["CPUExecutionProvider"])
    out_name = sess.get_outputs()[0].name
    return sess.run([out_name], feeds)[0]


def verify(src: Path, dst: Path) -> None:
    """B=1 bit-identisch (Optimizer aus); B=2/3 Batch-Unabhängigkeit (rel <= 1e-5)."""
    rng = np.random.default_rng(42)
    shape = (1, 128, 128, 128)

    # B=1 Parität: mit deaktivierten ORT-Optimierungen muss die Rechnung
    # bit-exakt sein (gleicher Graph, gleiche Kernels) — das beweist die
    # semantische Identität des Rewrites. Mit Default-Optimierern darf die
    # dynamische Batch-Dim andere GEMM-Kernels wählen (Toleranz 1e-5, weit
    # unter dem §G5 (copilot-instructions.md)-Paritäts-Gate rel <= 1e-3).
    for k in range(2):
        x = rng.standard_normal(shape).astype(np.float32)
        a = _run(src, {"input_fixed": x}, opt_level="off")
        b = _run(dst, {"input_fixed": x}, opt_level="off")
        delta_off = float(np.abs(a - b).max())
        status_off = "OK" if delta_off == 0.0 else "FAIL"
        logger.info("B=1 Parity (Opt aus) Eingabe %d: max|Δ|=%.3e %s", k, delta_off, status_off)
        if delta_off != 0.0:
            raise SystemExit(1)
        a_d = _run(src, {"input_fixed": x})
        b_d = _run(dst, {"input_fixed": x})
        delta_def = float(np.abs(a_d - b_d).max())
        status_def = "OK" if delta_def <= 1e-5 else "FAIL"
        logger.info("B=1 Parity (Opt an)  Eingabe %d: max|Δ|=%.3e %s", k, delta_def, status_def)
        if delta_def > 1e-5:
            raise SystemExit(1)

    # Batch-Unabhängigkeit: gestapelte Eingaben vs. Einzelläufe.
    for batch in (2, 3):
        xs = [rng.standard_normal(shape).astype(np.float32) for _ in range(batch)]
        stacked = np.concatenate(xs, axis=0)
        out_multi = _run(dst, {"input_fixed": stacked})
        worst = 0.0
        for i in range(batch):
            single = _run(dst, {"input_fixed": xs[i]})
            worst = max(worst, float(np.abs(out_multi[i] - single).max()))
        rel = worst / max(float(np.abs(out_multi).max()), 1e-12)
        status = "OK" if rel <= 1e-5 else "FAIL"
        logger.info("B=%d Unabhängigkeit: max|Δ|=%.3e rel=%.3e %s", batch, worst, rel, status)
        if rel > 1e-5:
            raise SystemExit(1)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Quelle fehlt: {SRC}")
    build_batch_generic(SRC, DST)
    verify(SRC, DST)
    logger.info("ERFOLG: %s ist batch-generisch und paritätsverifiziert.", DST)


if __name__ == "__main__":
    main()
