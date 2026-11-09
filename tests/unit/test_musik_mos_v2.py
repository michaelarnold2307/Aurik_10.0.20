"""Vertragstests musik_mos_v2 (2026-11-09) — lernfähiger v1-Klon + ONNX-Export.

Kern-Garantien: (a) v2 mit Prior-Gewichten ist BIT-IDENTISCH zu v1,
(b) der ONNX-Export ist §III.9-paritätsverifiziert (rel ≤ 1e-3) gegen die
Python-Referenz, (c) der Fit lernt aus A/B-Urteilen, (d) Determinismus.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.musik_mos_v2 import (
    MUSIK_MOS_V2_PRIOR_WEIGHTS,
    estimate_v2,
    export_onnx,
    feature_vector_v2,
    fit_v2_weights,
)

SR = 48000


def _sig(kind: str, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(SR * 3) / SR
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    if kind == "sinus":
        return tone.astype(np.float32)
    if kind == "rauschen":
        return (rng.standard_normal(len(t)) * 0.1).astype(np.float32)
    return (tone + rng.standard_normal(len(t)) * 0.02).astype(np.float32)


def test_v2_prior_bit_identisch_zu_v1() -> None:
    """v2 mit Prior-Gewichten == v1 estimate_mos (der zertifizierte Zeuge)."""
    from plugins.musik_mos_plugin import get_musik_mos

    for kind in ("sinus", "rauschen", "mix"):
        sig = _sig(kind)
        m1 = get_musik_mos().estimate_mos(sig, SR).mos
        m2 = estimate_v2(sig, SR)
        assert m1 == pytest.approx(m2, abs=1e-6), f"{kind}: v1={m1} v2={m2}"


def test_onnx_paritaet_rel_1e3() -> None:
    """§III.9: ONNX-Inferenz gegen die Python-Referenz, rel ≤ 1e-3."""
    ort = pytest.importorskip("onnxruntime")
    export_onnx(None, "/tmp/musik_mos_v2_test.onnx")
    sess = ort.InferenceSession("/tmp/musik_mos_v2_test.onnx", providers=["CPUExecutionProvider"])
    for kind in ("sinus", "rauschen", "mix"):
        sig = _sig(kind, seed=7)
        f = feature_vector_v2(sig, SR).astype(np.float32)
        py_mos = estimate_v2(sig, SR)
        onx_mos = float(np.asarray(sess.run(None, {"q": f})[0]).ravel()[0])
        rel = abs(py_mos - onx_mos) / max(py_mos, 1e-9)
        assert rel <= 1e-3, f"{kind}: rel={rel:.2e}"


def test_feature_vector_endlich_und_bereich() -> None:
    f = feature_vector_v2(_sig("mix"), SR)
    assert f.shape == (5,)
    assert np.all(np.isfinite(f))
    assert np.all(f >= 0.0) and np.all(f <= 1.0)


def test_fit_lernt_aus_urteilen() -> None:
    """Sauber gewinnt immer gegen Rauschen — das Gewicht wächst im Betrag."""
    sauber = _sig("sinus")
    rauschig = _sig("rauschen")
    pairs = [(sauber, rauschig, 1)] * 150
    fit = fit_v2_weights(pairs, SR, pseudo_count=10, seed=42)
    assert fit["n_labels"] == 150
    w = np.asarray(fit["weights"])
    prior = MUSIK_MOS_V2_PRIOR_WEIGHTS
    assert np.max(np.abs(w - prior)) > 0.01, f"Fit hat nicht gelernt: {w} vs {prior}"


def test_fit_determinismus_und_json() -> None:
    pairs = [(_sig("sinus", 1), _sig("rauschen", 2), 1)] * 30
    a = fit_v2_weights(pairs, SR, pseudo_count=10, seed=42)
    b = fit_v2_weights(pairs, SR, pseudo_count=10, seed=42)
    assert np.allclose(a["weights"], b["weights"], atol=1e-12)
    assert a["log_loss"] == b["log_loss"]


def test_export_mit_gelernten_gewichten() -> None:
    """Export mit gelernten Gewichten liefert dieselben Werte wie estimate_v2."""
    ort = pytest.importorskip("onnxruntime")
    pairs = [(_sig("sinus", 1), _sig("rauschen", 2), 1)] * 40
    fit = fit_v2_weights(pairs, SR, pseudo_count=10, seed=42)
    w = np.asarray(fit["weights"], dtype=np.float32)
    export_onnx(w, "/tmp/musik_mos_v2_learned.onnx")
    sess = ort.InferenceSession("/tmp/musik_mos_v2_learned.onnx", providers=["CPUExecutionProvider"])
    sig = _sig("mix", 3)
    f = feature_vector_v2(sig, SR).astype(np.float32)
    py_mos = estimate_v2(sig, SR, weights=w)
    onx_mos = float(np.asarray(sess.run(None, {"q": f})[0]).ravel()[0])
    assert abs(py_mos - onx_mos) / max(py_mos, 1e-9) <= 1e-3
