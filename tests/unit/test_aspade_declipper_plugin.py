"""Unit-Tests für das A-SPADE-Declipper-Plugin (DECLIPPER_SOTA_PLAN.md Slice B).

Prüft: Fallback ohne Modell (§V6 (copilot-instructions.md)), ONNX-Inferenz
über ein synthetisches Identity-Modell (dynamische Zeitdimension),
Determinismus (§G5 (copilot-instructions.md)), Never-worsen-Router in Phase 07.
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 48000


def _tone_clipped(dur_s: float = 2.0) -> np.ndarray:
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 440 * t) * 1.5
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _save_identity_onnx(path) -> None:
    """Schreibt ein Identity-ONNX (1, 1, T) → (1, 1, T) mit dynamischer Zeitachse."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, None])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, None])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "aspade_identity", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_plugin_unavailable_without_model(tmp_path):
    from plugins.aspade_declipper_plugin import AspadeDeclipperPlugin

    plug = AspadeDeclipperPlugin(model_dir=str(tmp_path))
    assert plug.is_available() is False
    x = _tone_clipped()
    assert plug.declip(x, SR, 0.99) is None


def test_plugin_identity_onnx_inference(tmp_path):
    from plugins.aspade_declipper_plugin import AspadeDeclipperPlugin

    _save_identity_onnx(tmp_path / "aspade_declipper.onnx")
    plug = AspadeDeclipperPlugin(model_dir=str(tmp_path))
    assert plug.is_available() is True

    x = _tone_clipped(2.0)
    out = plug.declip(x, SR, 0.99)
    assert out is not None
    assert out.shape == x.shape
    # Identity-Modell → Ausgabe ≈ Eingabe (bis auf Peak-Normierung/Fenster-Ränder).
    np.testing.assert_allclose(out, x, atol=5e-2)


def test_plugin_identity_deterministic(tmp_path):
    from plugins.aspade_declipper_plugin import AspadeDeclipperPlugin

    _save_identity_onnx(tmp_path / "aspade_declipper.onnx")
    plug = AspadeDeclipperPlugin(model_dir=str(tmp_path))
    x = _tone_clipped(2.0)
    r1 = plug.declip(x, SR, 0.99)
    r2 = plug.declip(x, SR, 0.99)
    np.testing.assert_array_equal(r1, r2)  # §G5 (copilot-instructions.md)


def test_phase07_router_rejects_identity_candidate(monkeypatch):
    """Never-worsen: Identity-Kandidat verbessert den Harmonik-Proxy nicht → aspade_used False."""
    from backend.core.phases.phase_07_declipper import DeclipperPhase

    class _FakeAspade:
        def is_available(self) -> bool:
            return True

        def declip(self, audio, _sr, _t):
            return np.asarray(audio, dtype=np.float32).copy()

    monkeypatch.setattr("plugins.aspade_declipper_plugin.get_aspade_declipper_plugin", lambda: _FakeAspade())
    # Schwere Clips: lange Runs ≥ 50 ms → A-SPADE-Zweig wird versucht.
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 440 * t) * 4.0
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    result = DeclipperPhase().process(x, sample_rate=sr)
    assert result.metrics.get("aspade_used") is False
    assert np.isfinite(result.audio).all()


def test_phase07_router_fallback_when_unavailable(monkeypatch):
    """Plugin ohne Modell → CQT-Diff/PCHIP-Pfad, aspade_used False, kein Crash (§V6 (copilot-instructions.md))."""
    from backend.core.phases.phase_07_declipper import DeclipperPhase

    class _NoModel:
        def is_available(self) -> bool:
            return False

        def declip(self, _a, _sr, _t):
            return None

    monkeypatch.setattr("plugins.aspade_declipper_plugin.get_aspade_declipper_plugin", lambda: _NoModel())
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.clip(np.sin(2 * np.pi * 440 * t) * 4.0, -1.0, 1.0).astype(np.float32)
    result = DeclipperPhase().process(x, sample_rate=sr)
    assert result.metrics.get("aspade_used") is False
    assert result.metrics.get("declip_applied") is True
    assert np.isfinite(result.audio).all()
