"""Regressionstest für den PANNs-ROCm-Provider-Filter (§SOTA-Fix 2026-09-16).

Produktionsbefund (überwachter 225-s-Elke-Best-Lauf): PANNs' lokaler
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
