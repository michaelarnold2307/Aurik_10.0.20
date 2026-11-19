"""musik_mos_v2 — lernfähiger Nachfolger des zertifizierten MusikMOS-Zeugen.

Schritt 1 des „musik_mos als ONNX“-Plans (2026-11-09): Ein echtes Modell
mit Gewichten statt der handkodierten v1-Konstanten. Der v2-Kopf hat
**fünf Feature-Inputs** (die v1-Komponenten q_rausch, q_tonal, q_clip,
q_hf, q_dyn), eine lineare Schicht + Sigmoid — und startet mit den
v1-Gewichten als Prior: **bit-identisch zu v1 bei Prior-Gewichten**
(Test-gepinnt), aber lernfähig aus A/B-Urteilen.

Lernen: Bradley-Terry über die 5-Feature-DIFFERENZEN von Urteilspaaren mit
Pseudo-Count-Mischung (gleiche Logik wie der Ohr-Prior):
  w_eff = (n·w_gelernt + pseudo_count·w_prior) / (n + pseudo_count)
Bei n=0 gilt v1; mit Urteilen trägt sich das gelernte Ohr selbst.

ONNX: `export_onnx` schreibt einen Graphen q[5] → Gewichtete-Summe → Clip →
Sigmoid → MOS[1,5] (nur Standard-Ops — numerisch unkritisch). Der
§III.9-Paritätsnachweis gegen die Python-Referenz ist im Vertragstest
gepinnt (rel ≤ 1e-3, ONNX-CPU).

Einsatz-Regel: v2 ersetzt v1 als ZEUGEN erst nach erneutem
Richtungs-Zertifikat (`quality_metric_direction_gate`) + B3-Never-worsen —
bis dahin bleibt v1 der aktive Zeuge (§V7 (copilot-instructions.md)).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

MUSIK_MOS_V2_FEATURES = ["rausch", "tonal", "clip", "hf", "dyn"]
# v1-Gewichte als Prior — v2(Prior) ist bit-identisch zu v1 (Test-gepinnt).
MUSIK_MOS_V2_PRIOR_WEIGHTS = np.array([0.28, 0.24, 0.14, 0.14, 0.20], dtype=np.float64)
V2_PSEUDO_COUNT_DEFAULT = 10
V2_FIT_SEED = 42


def feature_vector_v2(audio: np.ndarray, sr: int) -> np.ndarray:
    """Die fünf v1-Komponenten als Vektor (q_rausch, q_tonal, q_clip, q_hf, q_dyn).

    Berechnung identisch zu plugins/musik_mos_plugin.py (Bit-Identitätstest
    gegen v1-details gepinnt) — bewusst dupliziert statt refaktoriert, weil
    v1 der zertifizierte Zeuge ist und nicht angefasst werden soll.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[0] <= 8 and x.shape[-1] > 8:
            x = x.T
        x = x.mean(axis=1)
    ziel_sr = 48000
    if sr != ziel_sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(ziel_sr, int(sr))
        x = resample_poly(x, ziel_sr // g, int(sr) // g).astype(np.float32)
        sr = ziel_sr
    seg = x[: min(len(x), 2048 * 16)]
    if seg.size < 2048:
        raise ValueError("Signal zu kurz für musik_mos_v2 (mindestens 2048 Samples)")

    spec = np.abs(np.fft.rfft(seg, n=2048)) ** 2 + 1e-18
    flatness = float(np.clip(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-12), 0.0, 1.0))

    harmonie: list[float] = []
    rahmen = 2048
    for start in range(0, seg.size - rahmen + 1, rahmen):
        r = seg[start : start + rahmen]
        acf = np.correlate(r, r, "full")
        acf = acf[len(acf) // 2 :]
        peak = int(np.argmax(acf[20:])) + 20
        harmonie.append(float(np.clip(acf[peak] / (acf[0] + 1e-12), 0.0, 1.0)))
    q_tonal = float(np.median(harmonie)) if harmonie else 0.5

    clip_ratio = float(np.mean(np.abs(seg) > 0.98))
    freqs = np.fft.rfftfreq(2048, d=1.0 / sr)
    hf = float(np.sum(spec[freqs > 8000]) / (np.sum(spec) + 1e-12))
    rms = float(np.sqrt(np.mean(seg**2)) + 1e-12)
    crest_db = 20.0 * math.log10((float(np.max(np.abs(seg))) + 1e-12) / rms)

    q_rausch = 1.0 - min(1.0, 2.0 * flatness)
    q_clip = 1.0 - min(1.0, 10.0 * clip_ratio)
    q_hf = min(1.0, hf / 0.02) * (1.0 - flatness)
    q_dyn = float(math.exp(-0.5 * ((crest_db - 14.0) / 6.0) ** 2))
    return cast(np.ndarray, np.array([q_rausch, q_tonal, q_clip, q_hf, q_dyn], dtype=np.float64))


def mos_from_z(z: float) -> float:
    """MOS-Abbildung identisch zu v1: 1 + 4·σ(10·(z − 0.5)), geclippt [1, 5]."""
    z = float(np.clip(z, 0.0, 1.0))
    mos = 1.0 + 4.0 / (1.0 + math.exp(-10.0 * (z - 0.5)))
    return float(np.clip(mos, 1.0, 5.0))


def estimate_v2(audio: np.ndarray, sr: int, weights: np.ndarray | None = None) -> float:
    """musik_mos_v2-Schätzung; weights=None ⇒ Prior (bit-identisch zu v1)."""
    q = feature_vector_v2(audio, sr)
    w = MUSIK_MOS_V2_PRIOR_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64)
    z = float(np.clip(np.dot(w, q), 0.0, 1.0))
    return round(mos_from_z(z), 4)


def fit_v2_weights(
    pairs: list[tuple[np.ndarray, np.ndarray, int]],
    sr: int,
    pseudo_count: int = V2_PSEUDO_COUNT_DEFAULT,
    seed: int = V2_FIT_SEED,
) -> dict[str, float | int | list[float]]:
    """Bradley-Terry über 5-Feature-Differenzen mit Pseudo-Count-Prior (v1-Gewichte).

    pairs: (audio_a, audio_b, label), label=1 wenn A gewinnt.
    Deterministisch (feste Schritte, kein Sampling) — §G5 (copilot-instructions.md).
    """
    n = len(pairs)
    if n == 0:
        return {
            "weights": [float(w) for w in MUSIK_MOS_V2_PRIOR_WEIGHTS],
            "weights_prior": [float(w) for w in MUSIK_MOS_V2_PRIOR_WEIGHTS],
            "n_labels": 0,
            "pseudo_count": pseudo_count,
            "log_loss": 0.0,
        }
    xs = np.stack([feature_vector_v2(a, sr) - feature_vector_v2(b, sr) for a, b, _ in pairs])
    ys = np.asarray([float(label) for _, _, label in pairs], dtype=np.float64)
    w = MUSIK_MOS_V2_PRIOR_WEIGHTS.copy()
    lam = float(pseudo_count) / float(n)
    lr = 0.05
    for _ in range(3000):
        p = 1.0 / (1.0 + np.exp(-np.clip(xs @ w, -40.0, 40.0)))
        grad = xs.T @ (p - ys) / n + lam * (w - MUSIK_MOS_V2_PRIOR_WEIGHTS)
        w = np.clip(w - lr * grad, -5.0, 5.0)
    p = 1.0 / (1.0 + np.exp(-np.clip(xs @ w, -40.0, 40.0)))
    eps = 1e-9
    log_loss = float(-np.mean(ys * np.log(p + eps) + (1.0 - ys) * np.log(1.0 - p + eps)))
    w_eff = (n * w + pseudo_count * MUSIK_MOS_V2_PRIOR_WEIGHTS) / (n + pseudo_count)
    return {
        "weights": [float(v) for v in w_eff],
        "weights_prior": [float(v) for v in MUSIK_MOS_V2_PRIOR_WEIGHTS],
        "weights_learned": [float(v) for v in w],
        "n_labels": n,
        "pseudo_count": pseudo_count,
        "log_loss": round(log_loss, 6),
        "seed": seed,
    }


def export_onnx(weights: np.ndarray | None, path: str | Path, opset: int = 13) -> None:
    """Exportiert den v2-Kopf q[5] → Gemm/ReduceSum → Clip → Sigmoid → MOS.

    Nur Standard-Ops (Mul, ReduceSum, Add, Clip, Sub, Mul, Sigmoid, Clip) —
    numerisch unkritisch; der §III.9-Paritätsnachweis gegen die
    Python-Referenz ist im Vertragstest gepinnt.
    """
    import onnx
    from onnx import TensorProto, helper

    w = MUSIK_MOS_V2_PRIOR_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64)
    w_f = w.astype(np.float32)

    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, [5])
    mos = helper.make_tensor_value_info("mos", TensorProto.FLOAT, [1])
    nodes = [
        helper.make_node("Mul", ["q", "W"], ["weighted"]),
        helper.make_node("ReduceSum", ["weighted"], ["z_raw"]),
        helper.make_node("Clip", ["z_raw", "z_min", "z_max"], ["z"]),
        helper.make_node("Sub", ["z", "half"], ["z_center"]),
        helper.make_node("Mul", ["z_center", "ten"], ["z_scaled"]),
        helper.make_node("Sigmoid", ["z_scaled"], ["s"]),
        helper.make_node("Mul", ["s", "four"], ["s4"]),
        helper.make_node("Add", ["s4", "one"], ["mos_raw"]),
        helper.make_node("Clip", ["mos_raw", "one", "five"], ["mos"]),
    ]
    inits = [
        helper.make_tensor("W", TensorProto.FLOAT, [5], w_f.tolist()),
        helper.make_tensor("z_min", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("z_max", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("half", TensorProto.FLOAT, [], [0.5]),
        helper.make_tensor("ten", TensorProto.FLOAT, [], [10.0]),
        helper.make_tensor("four", TensorProto.FLOAT, [], [4.0]),
        helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("five", TensorProto.FLOAT, [], [5.0]),
    ]
    graph = helper.make_graph(nodes, "musik_mos_v2_kopf", [q], [mos], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


__all__ = [
    "MUSIK_MOS_V2_FEATURES",
    "MUSIK_MOS_V2_PRIOR_WEIGHTS",
    "feature_vector_v2",
    "mos_from_z",
    "estimate_v2",
    "fit_v2_weights",
    "export_onnx",
]
