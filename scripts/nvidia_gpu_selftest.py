#!/usr/bin/env python3
"""GPU-Selftest für NVIDIA-/ROCm-/CPU-Hosts (R3, 2026-09-14).

Prüft ohne Hard-Fail, welche ML-Pfade auf DIESEM Host GPU-fähig sind —
auf einem NVIDIA-System (CUDA) genauso wie auf ROCm oder CPU:
  1. Backend-Detektion (torch.version.hip/cuda, ORT-Provider)
  2. Torch-GPU-Fähigkeit (CUDA/ROCm-Gerät verfügbar?)
  3. ORT-Provider-Kette (CUDA/ROCM/MIGraphX in der Kette?)
  4. MuQ-MOS auf GPU (10-s-Clip, falls Modell vorhanden)
  5. DiffWave-Torch-Modul (Checkpoint vorhanden? Device-Auflösung)
  6. CQTdiff+: Schrittzahl-Adaption (35 auf GPU-Host, 3 auf CPU)

Report: docs/reports/current/<datum>_gpu_selftest.json
Exit-Code 0 auch bei übersprungenen Checks (kein Fail-closed auf fehlende Modelle).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_REPORT_DIR = _ROOT / "docs" / "reports" / "current"
sys.path.insert(0, str(_ROOT))  # backend-/plugins-Importe aus scripts/ heraus


def _backend() -> str:
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return "rocm"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def _torch_gpu_ready() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        x = torch.zeros(2, 3, device=torch.device("cuda"))
        _ = x.sum().item()
        return True
    except Exception:
        return False


def _ort_gpu_chain() -> list[str]:
    try:
        from backend.core.gpu_model_registry import get_onnx_providers

        _p = get_onnx_providers(str(_ROOT / "models" / "cqtdiff" / "score_network.onnx"))
        return [str(x) for x in _p]
    except Exception:
        return []


def main() -> int:
    checks: dict[str, object] = {}
    checks["backend"] = _backend()
    checks["torch_gpu_ready"] = _torch_gpu_ready()
    _chain = _ort_gpu_chain()
    checks["ort_provider_chain"] = _chain
    checks["ort_chain_has_gpu"] = any("CUDA" in c or "ROCM" in c or "MIGraphX" in c for c in _chain)

    # MuQ-MOS (10-s-Clip) — nur wenn Modell vorhanden.
    try:
        import plugins.muq_plugin as _muq

        if _muq.is_available():
            _rng = np.random.RandomState(0)
            _clip = _rng.randn(10 * 48000).astype(np.float32) * 0.05
            _mos = _muq.estimate_muq_mos(_clip, 48000)
            checks["muq_mos"] = {"available": True, "mos": None if _mos is None else round(float(_mos), 3)}
        else:
            checks["muq_mos"] = {"available": False}
    except Exception as _exc:
        checks["muq_mos"] = {"available": False, "error": str(_exc)[:120]}

    # DiffWave-Torch (Checkpoint-Prüfung + Device-Auflösung ohne Inferenz).
    try:
        import backend.core.dsp.diffwave_torch_inpaint as _dw

        checks["diffwave_torch"] = {
            "finetuned_ckpt": _dw._CKPT_FINETUNED.is_file(),
            "base_ckpt": _dw._CKPT_BASE.is_file(),
            "device": _dw._device(),
        }
    except Exception as _exc:
        checks["diffwave_torch"] = {"error": str(_exc)[:120]}

    # CQTdiff+: Schrittzahl-Adaption dokumentieren (35 auf GPU-Host, sonst 3).
    try:
        from plugins.cqtdiff_plus_plugin import CQTdiffPlusPlugin

        checks["cqtdiff_steps"] = {
            "cpu_steps": CQTdiffPlusPlugin.DIFFUSION_STEPS,
            "gpu_steps": CQTdiffPlusPlugin._FULL_STEPS,
            "note": "35 Schritte aktiv, sobald TorchScript-Modell auf GPU lädt (_device != cpu)",
        }
    except Exception as _exc:
        checks["cqtdiff_steps"] = {"error": str(_exc)[:120]}

    report = {"date": f"{date.today().isoformat()}", "checks": checks}
    out = _REPORT_DIR / f"{date.today().isoformat()}_gpu_selftest.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
