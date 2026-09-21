"""Regressionstest für den PANNs-ROCm-Provider-Filter (§SOTA-Fix 2026-09-16).

Produktionsbefund (überwachter 225-s-Test-Track-Lauf): PANNs' lokaler
Provider-Filter verglich (Name, Options)-TUPEL gegen String-Namen —
der ROCm-GPU-Provider wurde dadurch IMMER verworfen und PANNs lief
dauerhaft CPU-only. Dieser Test friert das korrigierte Verhalten ein.
"""

from __future__ import annotations

from unittest import mock

import pytest


def _make_panns() -> object:
    from pathlib import Path

    from plugins.panns_plugin import PANNsPlugin

    plugin = PANNsPlugin.__new__(PANNsPlugin)
    plugin._ONNX_PATH = Path("/tmp/fake_panns.onnx")
    plugin._session = None
    plugin._device = "cpu"
    plugin._use_fp16 = False
    return plugin


@pytest.mark.unit
class TestPannsRocmProviderFilter:
    def test_tuple_providers_survive_filter_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import plugins.panns_plugin as panns_mod

        plugin = _make_panns()
        monkeypatch.setattr(plugin, "_detect_gpu", lambda: ("cuda", True))
        monkeypatch.setattr(panns_mod, "_MIOPEN_POOL_FAILED_THIS_PROCESS", False)

        _tuple_providers = [("ROCMExecutionProvider", {"device_id": 0, "gpu_mem_limit": 100}), "CPUExecutionProvider"]
        monkeypatch.setattr("backend.core.ml_memory_budget.try_allocate", lambda name, size_gb: True)
        monkeypatch.setattr(
            "backend.core.ml_device_manager.get_ort_providers", lambda plugin_name: list(_tuple_providers)
        )

        import onnxruntime as ort

        fake_session = mock.MagicMock()
        fake_session.get_providers.return_value = ["ROCMExecutionProvider"]
        captured: dict = {}

        def _fake_inference_session(path, sess_options=None, providers=None):
            captured["providers"] = providers
            return fake_session

        monkeypatch.setattr(ort, "get_available_providers", lambda: ["ROCMExecutionProvider", "CPUExecutionProvider"])
        monkeypatch.setattr(ort, "InferenceSession", _fake_inference_session)
        monkeypatch.setattr(ort, "SessionOptions", mock.MagicMock)
        monkeypatch.setattr(ort, "GraphOptimizationLevel", mock.MagicMock())

        plugin._load_onnx()

        assert plugin._device == "cuda"
        # Der GPU-Provider MUSS den Filter überstehen (Tupel-Name wird ausgepackt).
        assert captured["providers"] is not None
        assert any((p[0] if isinstance(p, tuple) else str(p)) == "ROCMExecutionProvider" for p in captured["providers"])
        assert plugin._session is fake_session

    def test_unavailable_provider_falls_back_to_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import plugins.panns_plugin as panns_mod

        plugin = _make_panns()
        monkeypatch.setattr(plugin, "_detect_gpu", lambda: ("cuda", True))
        monkeypatch.setattr(panns_mod, "_MIOPEN_POOL_FAILED_THIS_PROCESS", False)
        monkeypatch.setattr("backend.core.ml_memory_budget.try_allocate", lambda name, size_gb: True)
        monkeypatch.setattr(
            "backend.core.ml_device_manager.get_ort_providers",
            lambda plugin_name: [("CUDAExecutionProvider", {}), "CPUExecutionProvider"],
        )

        import onnxruntime as ort

        fake_session = mock.MagicMock()
        fake_session.get_providers.return_value = ["CPUExecutionProvider"]
        captured: dict = {}

        def _fake_inference_session(path, sess_options=None, providers=None):
            captured["providers"] = providers
            return fake_session

        # CPU-only-ORT-Build: CUDA nicht registriert → CPU-only (fail-closed
        # §V6 (copilot-instructions.md)).
        monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
        monkeypatch.setattr(ort, "InferenceSession", _fake_inference_session)
        monkeypatch.setattr(ort, "SessionOptions", mock.MagicMock)
        monkeypatch.setattr(ort, "GraphOptimizationLevel", mock.MagicMock())

        plugin._load_onnx()

        assert captured["providers"] == ["CPUExecutionProvider"]
        assert plugin._session is fake_session


@pytest.mark.unit
def test_multi_window_parallel_bit_identical_to_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    """§PERF-R SOTA4-1 (2026-09-18): Die Fallback-Fenster (0.25/0.75) sind
    unabhängig; die parallele Inferenz (INFER_PARALLEL > 1) muss
    bit-identisch zum sequenziellen Pfad sein (Maximum in fester
    Reihenfolge)."""
    import numpy as np

    import plugins.panns_plugin as panns_mod

    class _In:
        name = "input"

    class _FakeSession:
        def get_inputs(self):
            return [_In()]

        def run(self, output_names, feed):
            x = np.asarray(feed["input"], dtype=np.float64)  # [1, 320000]
            m = float(x.mean())
            scores = np.zeros(527, dtype=np.float32)
            scores[0] = m
            scores[1] = 1.0 - m
            # Singing voice (Index 27): Mitte < 0.35 (löst Fallback aus),
            # Seitenfenster teils hoch — Inhalt-abhängig, deterministisch.
            scores[27] = 0.9 if m < 0.45 else 0.1
            return [scores[np.newaxis, :]]

    # Ramp-Signal (25 s @ 32 kHz) ⇒ Fenster-Mittel 0.35 / 0.50 / 0.65
    n = 25 * 32_000
    audio = (np.arange(n, dtype=np.float32) / n)[np.newaxis, :]

    def _run(parallel: int) -> dict[str, float]:
        plugin = panns_mod.PANNsPlugin.__new__(panns_mod.PANNsPlugin)
        plugin._session = _FakeSession()
        plugin._device = "cpu"
        plugin._use_fp16 = False
        plugin.INFER_PARALLEL = parallel
        panns_mod._tags_cache.clear()
        return plugin.get_tags(audio, 32_000)

    seq = _run(0)
    par = _run(2)

    assert set(seq) == set(par)
    for k in seq:
        assert seq[k] == par[k], f"Tag {k} divergiert: {seq[k]} vs {par[k]}"
    # Singing-Voice-Score muss durch das Maximum-Merge angehoben sein (0.1 → 0.9)
    assert seq.get("Singing voice", 0.0) > 0.8
