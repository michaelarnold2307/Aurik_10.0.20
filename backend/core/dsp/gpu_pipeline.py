"""
dsp/gpu_pipeline.py — Compatibility-Stub → leitet auf CPUPipeline weiter
=========================================================================

GPU-Beschleunigung wurde wegen Inkompatibilitäten deaktiviert.
Dieses Modul leitet alle Aufrufe transparent auf die CPUPipeline weiter.

Author: Aurik Development Team
Version: 1.0.0 (CPU-only redirect)
"""

import warnings

from dsp.cpu_pipeline import CPUPipeline, PipelineStats

warnings.warn(
    "dsp.gpu_pipeline ist deprecated — nutze dsp.cpu_pipeline.CPUPipeline",
    DeprecationWarning,
    stacklevel=1,
)

GPUPipeline = CPUPipeline
GPUPipelineStats = PipelineStats

__all__ = ["CPUPipeline", "GPUPipeline", "GPUPipelineStats", "PipelineStats"]
