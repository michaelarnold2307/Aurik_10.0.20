"""§P1-6 — DeepFilterNet dec.onnx Alpha-Head (ML→DSP-Befund + Fix).

Befund 2026-09-08: IndexError in _infer_spectral_chunk (alpha = dec_out[1])
→ stiller OMLSA-Fallback. Root-Cause: der aktive finetuned dec.onnx war ein
veralteter Export ohne Alpha-Head (df_fc_a). Fix 2026-09-08: alle drei
ONNX-Modelle aus dfn_musik_best.pt neu exportiert (scripts/
export_dfn_finetuned_onnx.py), dec MIT Alpha-Ausgang; alpha optional
behandeln (fehlendes alpha = kein IndexError).

§v10-DFN-Feature-Fix (2026-09-10): Der trainierte DFN3-Forward nutzt df_fc_a
(alpha) NICHT — der Enhance-Pfad wendet pure df_op an (5 komplexe Taps),
kein Alpha-Blend. Diese Tests prüfen genau diese Semantik.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepfilternet_v3_ii"


class _FakeSession:
    """ONNX-Session-Stub mit festen Outputs (nur run())."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outs = outputs

    def run(self, *_args, **_kwargs) -> list[np.ndarray]:
        return list(self._outs)


_S = 12  # Frames (realistisch >= DF_ORDER; Produktion: T=100)


def _make_plugin(dec_outputs: list[np.ndarray]) -> DeepFilterNetV3Plugin:
    p = DeepFilterNetV3Plugin.__new__(DeepFilterNetV3Plugin)  # ohne Modell-Load
    p._df_state = None  # §v10-DFN-Feature-Fix: numpy-Kette im Test (libdf-frei)
    p._erb_widths = None
    p._current_energy_bias_db = 0.0
    p._enc = _FakeSession(
        [
            np.zeros((1, 16, _S, 32)),
            np.zeros((1, 16, _S, 16)),
            np.zeros((1, 16, _S, 8)),
            np.zeros((1, 16, _S, 8)),
            np.zeros((1, _S, 256)),
            np.zeros((1, 16, _S, 96)),
            np.zeros((1, _S, 1)),
        ]
    )
    p._erb_dec = _FakeSession([np.zeros((1, 1, _S, 32))])
    p._dec = _FakeSession(dec_outputs)
    return p


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_erb = np.zeros((1, 1, _S, 32), dtype=np.float32)
    feat_spec = np.zeros((1, 2, _S, 96), dtype=np.float32)
    spec_cx = np.zeros((481, _S), dtype=np.complex64)  # volle FFT-Bins
    return feat_erb, feat_spec, spec_cx


def test_infer_spectral_chunk_single_output_no_crash() -> None:
    """§P1-6: dec mit nur coefs → kein IndexError, pure DF."""
    p = _make_plugin([np.zeros((1, _S, 96, 10), dtype=np.float32)])
    out = p._infer_spectral_chunk(*_inputs())
    assert out.shape == (481, _S)


def test_infer_spectral_chunk_two_outputs_alpha_ignored() -> None:
    """dec mit coefs + alpha → kein Crash; alpha geht nicht in den Enhance-Pfad
    ein (trainierter DFN3-Forward nutzt df_fc_a nicht, §v10-DFN-Feature-Fix)."""
    alpha = np.full((1, _S, 1), 0.5, dtype=np.float32)
    p = _make_plugin([np.zeros((1, _S, 96, 10), dtype=np.float32), alpha])
    out = p._infer_spectral_chunk(*_inputs())
    assert out.shape == (481, _S)


def test_apply_df_filter_pure_df_zero_tap() -> None:
    """pure df_op: nur der 0. Tap aktiv (0.25 reell) → acc = 0.25·spec
    auf den ersten 96 Bins; höhere Bins bleiben original."""
    p = DeepFilterNetV3Plugin.__new__(DeepFilterNetV3Plugin)
    spec = np.ones((481, _S), dtype=np.complex64)
    coefs = np.zeros((_S, 96, 10), dtype=np.float32)
    coefs[:, :, 0] = 0.25  # re des 0. Taps
    out = p._apply_df_filter(spec, coefs)
    expected = np.ones((481, _S), dtype=np.complex64)
    expected[:96, :] = 0.25
    assert np.allclose(out, expected, atol=1e-6)


def test_apply_df_filter_tap1_shift_semantics() -> None:
    """df_op mit Tap 1 aktiv (1.0 reell) → enh[t] = spec[t-1] (Clamp t=0)."""
    p = DeepFilterNetV3Plugin.__new__(DeepFilterNetV3Plugin)
    rng = np.random.default_rng(7)
    spec = (rng.standard_normal((481, _S)) + 1j * rng.standard_normal((481, _S))).astype(np.complex64)
    coefs = np.zeros((_S, 96, 10), dtype=np.float32)
    coefs[:, :, 2] = 1.0  # re des 1. Taps (Paar-Layout)
    out = p._apply_df_filter(spec, coefs)
    expected = spec.copy()
    shifted = spec[:96, :].copy()
    _s = np.empty_like(shifted)
    _s[:, 1:] = shifted[:, :-1]
    _s[:, 0] = shifted[:, 0]  # clamp max(0, t-1)
    expected[:96, :] = _s
    assert np.allclose(out, expected, atol=1e-6)


def test_active_finetuned_dec_has_alpha_head() -> None:
    """§Fix 2026-09-08: aktiver finetuned dec.onnx MUSS den Alpha-Head liefern.

    Regression gegen veraltete Exporte ohne df_fc_a (→ pure-DF-Degradation).
    Modelle sind gitignored → skip, wenn nicht vorhanden.
    """
    dec = _MODEL_DIR / "finetuned" / "dec.onnx"
    if not dec.is_file():
        pytest.skip("models/ nicht vorhanden (gitignored)")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(dec), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    assert "coefs" in names
    assert len(names) >= 2, names
    assert any("alpha" in n.lower() or "sigmoid" in n.lower() for n in names), names


def test_plugin_loads_alpha_from_finetuned_model() -> None:
    """Plugin lädt das aktive finetuned-Set und erkennt den Alpha-Head."""
    dec = _MODEL_DIR / "finetuned" / "dec.onnx"
    enc = _MODEL_DIR / "finetuned" / "enc.onnx"
    erb = _MODEL_DIR / "finetuned" / "erb_dec.onnx"
    if not (dec.is_file() and enc.is_file() and erb.is_file()):
        pytest.skip("models/ nicht vorhanden (gitignored)")
    p = DeepFilterNetV3Plugin()
    assert p._dec_has_alpha is True
    assert p._dec is not None and p._enc is not None and p._erb_dec is not None
