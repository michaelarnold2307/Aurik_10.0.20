"""§SOTA-Fix 2026-09-15 — ORT-Provider-Validierung (C++-Abort-Schutz).

Produktionsbefund: ``ort.InferenceSession(..., providers=[..., 'ROCMExecutionProvider', ...])``
bricht auf einem CPU-only-ORT-Build auf C++-Ebene ab (std::terminate/SIGABRT,
„Unknown Provider Type"), wenn der MIGraphX-GPU-Detektor gleichzeitig aktiv ist
(AMD 7900 XTX). Fix: ``_filter_to_available_ort_providers`` in
``ml_device_manager`` filtert Provider fail-closed gegen
``ort.get_available_providers()`` (§V6 (copilot-instructions.md): Warnung,
nie stiller Ersatzpfad).

Autor: Aurik Testing Team
"""

import pytest

from backend.core.ml_device_manager import _filter_to_available_ort_providers


class TestFilterToAvailableOrtProviders:
    def test_unknown_provider_dropped_with_cpu_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "onnxruntime.get_available_providers", lambda: ["AzureExecutionProvider", "CPUExecutionProvider"]
        )
        out = _filter_to_available_ort_providers(
            [("ROCMExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
            "TestPlugin",
        )
        assert out == ["CPUExecutionProvider"]

    def test_all_unknown_yields_cpu(self, monkeypatch):
        monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["CPUExecutionProvider"])
        out = _filter_to_available_ort_providers(["CUDAExecutionProvider"], "TestPlugin")
        assert out == ["CPUExecutionProvider"]

    def test_known_providers_preserved(self, monkeypatch):
        monkeypatch.setattr(
            "onnxruntime.get_available_providers",
            lambda: ["ROCMExecutionProvider", "CPUExecutionProvider"],
        )
        providers = [("ROCMExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        out = _filter_to_available_ort_providers(providers, "TestPlugin")
        assert out == providers

    def test_ort_probe_failure_is_cpu_fail_closed(self, monkeypatch):
        def _boom():
            raise RuntimeError("kein ORT")

        monkeypatch.setattr("onnxruntime.get_available_providers", _boom)
        out = _filter_to_available_ort_providers(["ROCMExecutionProvider"], "TestPlugin")
        assert out == ["CPUExecutionProvider"]
