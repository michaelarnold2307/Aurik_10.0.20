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
    """Schreibt ein Identity-ONNX (1, 1, 512, T) → (1, 1, 512, T), dynamische Zeitachse.

    Entspricht dem APPLADE-Vertrag: Magnitude-Patch (Frequenz 512 × Zeit T).
    """
    pytest.importorskip("onnx")
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 512, None])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 512, None])
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
    """APPLADE-Pfad um ein Identity-DNN: ADMM läuft, Ausgabe valide + endlich."""
    from plugins.aspade_declipper_plugin import AspadeDeclipperPlugin

    _save_identity_onnx(tmp_path / "aspade_declipper.onnx")
    plug = AspadeDeclipperPlugin(model_dir=str(tmp_path))
    assert plug.is_available() is True

    x = _tone_clipped(2.0)
    out = plug.declip(x, SR, 0.99)
    assert out is not None
    assert out.shape == x.shape
    assert np.isfinite(out).all()
    assert float(np.sqrt(np.mean(out**2))) > 1e-4  # nicht stumm


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


def test_plugin_resampling_robustheit(tmp_path):
    """D-K4-1-Voraussetzung: sr ≠ 48 kHz ⇒ deterministischer Resampling-Pfad
    statt None-Ersatzpfad (robuste Modellreparatur nach Resampling)."""
    from plugins.aspade_declipper_plugin import AspadeDeclipperPlugin

    _save_identity_onnx(tmp_path / "aspade_declipper.onnx")
    plug = AspadeDeclipperPlugin(model_dir=str(tmp_path))
    assert plug.is_available() is True
    sr = 32000
    t = np.arange(sr) / sr  # 1 s
    x32 = np.clip(np.sin(2 * np.pi * 440.0 * t) * 1.4, -0.99, 0.99).astype(np.float32)
    out = plug.declip(x32, sr, 0.99)
    assert out is not None, "sr ≠ 48 kHz muss den Resampling-Pfad nutzen (kein None-Ersatzpfad)"
    assert out.shape == x32.shape
    assert np.isfinite(out).all()
    out2 = plug.declip(x32, sr, 0.99)
    assert np.array_equal(out, out2), "Determinismus (§G5 (copilot-instructions.md))"
    # ungültige Rate fail-closed (§V6 (copilot-instructions.md))
    assert plug.declip(x32, 0, 0.99) is None


def _spikes_schwellennah(n: int = 48000) -> np.ndarray:
    """±0,90-Einzelsample-Clips — der Träumer-Fall (2026-11-13): nach AA-Resampling
    liegen die Plateaus unter der Schwelle, nur die 48-kHz-Maske sieht sie."""
    x = np.zeros(n, dtype=np.float32)
    x[::2400] = 0.90
    x[1200::2400] = -0.90
    return x


def test_region_idx_48k_findet_schwellennahe_plateaus():
    """2026-11-13-Befund: das alte 16-kHz-Gate verpasste schwelle-nahe Clips
    (Träumer Δ = 0,0) — die 48-kHz-Maske muss sie markieren."""
    from scipy.signal import resample_poly

    from plugins.aspade_declipper_plugin import _mask16_from_mask48, _region_idx_48k

    x = _spikes_schwellennah()
    theta = 0.85
    mask = np.abs(x) >= theta
    assert mask.any()
    x16 = resample_poly(x.astype(np.float64), 1, 3).astype(np.float64)

    # Defekt-Beweis: das alte 16-kHz-Gate hätte gar nichts gesehen
    alt_h = (x16 > theta) | (np.abs(x16 - theta) < 1e-9)
    alt_l = x16 < -theta
    assert not alt_h.any() and not alt_l.any(), "Annahme verletzt: AA hätte die Spikes gedämpft"

    m16 = _mask16_from_mask48(mask, len(x16))
    idx = _region_idx_48k(x16, m16)
    assert idx["H"].any() and idx["L"].any()
    assert (~idx["R"]).sum() >= 1


def test_declip_applade_bearbeitet_schwellennahe_clips(tmp_path):
    """E2E: vor dem Fix gab der Zweig bei schwelle-nahen Clips den Eingang
    unverändert zurück (Passthrough) — jetzt werden sie bearbeitet."""
    from plugins.aspade_declipper_plugin import AspadeDeclipperPlugin

    _save_identity_onnx(tmp_path / "aspade_declipper.onnx")
    plug = AspadeDeclipperPlugin(model_dir=str(tmp_path))
    assert plug.is_available() is True

    x = _spikes_schwellennah()
    out = plug.declip(x, SR, 0.85)
    assert out is not None
    assert out.shape == x.shape
    assert np.isfinite(out).all()
    assert np.abs(out - x).max() > 1e-6, "Clip-Regionen müssen bearbeitet werden (kein stummes Passthrough)"
