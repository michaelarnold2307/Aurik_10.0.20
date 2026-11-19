"""
ONNX Runtime Infrastructure for AURIK v8

This module provides ONNX Runtime integration for 3-6× speedup of ML model inference.

Key Components:
- OptimizedONNXModel: ONNX Runtime wrapper for model inference
- ONNXConverter: PyTorch → ONNX converter
- ModelQuantizer: FP32 → INT8 quantization
- FallbackManager: Automatic ONNX → PyTorch fallback
- ONNXModelWithFallback: Combined ONNX + PyTorch with auto fallback

Expected Speedup:
- ONNX: 1.5-2× faster than PyTorch
- Quantization: 2-3× additional speedup
- Total: 3-6× faster inference

Usage:
    from backend.core.onnx import OptimizedONNXModel, ModelQuantizer

    # Load ONNX model
    model = OptimizedONNXModel("model.onnx", model_type='denoising')

    # Process audio
    output = model.process(audio)

    # Quantize model
    quantizer = ModelQuantizer()
    quantizer.quantize("model.onnx", "model_quantized.onnx")

§Fix 2026-09-22 (Spec 24 — kein stiller Ausfall, kein Import-Crash):
Die Submodule werden über PEP 562 (`__getattr__`) lazy importiert. Vorher
scheiterte bereits `import backend.core.onnx.quantizer` ohne onnxruntime, weil
der Paket-`__init__` alle Submodule eifrig importierte (runtime.py zieht
onnxruntime auf Modulebene). Jetzt ist das Paket selbst importierbar; der
onnxruntime-Bedarf entsteht erst beim tatsächlichen Nutzen der Runtime-Klassen.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "ConversionConfig",
    "FallbackEvent",
    # Fallback
    "FallbackManager",
    "FallbackReason",
    "FallbackStats",
    # Model Info
    "ModelInfo",
    # Quantizer
    "ModelQuantizer",
    "ModelSpecificConverter",
    # Converter
    "ONNXConverter",
    "ONNXInferenceSession",
    "ONNXModelStatus",
    "ONNXModelWithFallback",
    # Plugin Manager
    "ONNXPluginManager",
    "ONNXProvider",
    # Runtime
    "OptimizedONNXModel",
    "QuantizationConfig",
    "QuantizationType",
    "load_model",
    "process_audio",
]

# Attributname -> (Submodul, Symbol) für den Lazy-Import.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ConversionConfig": (".converter", "ConversionConfig"),
    "ModelSpecificConverter": (".converter", "ModelSpecificConverter"),
    "ONNXConverter": (".converter", "ONNXConverter"),
    "FallbackEvent": (".fallback", "FallbackEvent"),
    "FallbackManager": (".fallback", "FallbackManager"),
    "FallbackReason": (".fallback", "FallbackReason"),
    "FallbackStats": (".fallback", "FallbackStats"),
    "ONNXModelWithFallback": (".fallback", "ONNXModelWithFallback"),
    "ModelInfo": (".model_info", "ModelInfo"),
    "ONNXModelStatus": (".model_info", "ONNXModelStatus"),
    "ONNXPluginManager": (".plugin_manager", "ONNXPluginManager"),
    "load_model": (".plugin_manager", "load_model"),
    "process_audio": (".plugin_manager", "process_audio"),
    "ModelQuantizer": (".quantizer", "ModelQuantizer"),
    "QuantizationConfig": (".quantizer", "QuantizationConfig"),
    "QuantizationType": (".quantizer", "QuantizationType"),
    "ONNXInferenceSession": (".runtime", "ONNXInferenceSession"),
    "ONNXProvider": (".runtime", "ONNXProvider"),
    "OptimizedONNXModel": (".runtime", "OptimizedONNXModel"),
}


def __getattr__(name: str) -> Any:
    """PEP 562: Lazy-Import des zugehörigen Submoduls beim ersten Zugriff."""
    try:
        _submodule, _symbol = _LAZY_EXPORTS[name]
    except KeyError as _missing:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from _missing
    _module = importlib.import_module(_submodule, __name__)
    _value = getattr(_module, _symbol)
    globals()[name] = _value
    return _value
