"""Verify the effective ROCm runtime path for Aurik restoration plugins."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _torch_status() -> dict[str, object]:
    try:
        import torch

        status: dict[str, object] = {
            "import_ok": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "hip_version": getattr(getattr(torch, "version", None), "hip", None),
        }
        if status["cuda_available"]:
            tensor = torch.zeros(64, dtype=torch.float32, device="cuda")
            status["device_name"] = torch.cuda.get_device_name(0)
            status["tensor_device"] = str(tensor.device)
            torch.cuda.synchronize()
        return status
    except Exception as exc:
        return {"import_ok": False, "error": str(exc)}


def _ort_status() -> dict[str, object]:
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        return {
            "import_ok": True,
            "providers": providers,
            "rocm_provider": "ROCMExecutionProvider" in providers,
        }
    except Exception as exc:
        return {"import_ok": False, "error": str(exc)}


def _aurik_status() -> dict[str, object]:
    try:
        from backend.core.ml_device_manager import get_ml_device_manager, warmup_rocm_gpu

        manager = get_ml_device_manager()
        warmup_ok = warmup_rocm_gpu()
        return {
            "backend": manager.get_gpu_backend().value,
            "gpu_available": manager.is_gpu_available(),
            "warmup_ok": warmup_ok,
            "torch_device_AudioSR": manager.get_torch_device("AudioSR"),
            "torch_device_SGMSE": manager.get_torch_device("SGMSE"),
            "ort_DeepFilterNetV3": manager.get_ort_providers("DeepFilterNetV3"),
            "ort_PANNs": manager.get_ort_providers("PANNs"),
        }
    except Exception as exc:
        return {"error": str(exc)}


def main() -> int:
    report = {
        "python": sys.executable,
        "torch": _torch_status(),
        "onnxruntime": _ort_status(),
        "aurik": _aurik_status(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    torch_ok = bool(report["torch"].get("cuda_available")) if isinstance(report["torch"], dict) else False
    ort_ok = bool(report["onnxruntime"].get("rocm_provider")) if isinstance(report["onnxruntime"], dict) else False
    aurik_ok = False
    if isinstance(report["aurik"], dict):
        aurik_ok = (
            report["aurik"].get("backend") == "rocm"
            and report["aurik"].get("torch_device_AudioSR") == "cuda"
            and isinstance(report["aurik"].get("ort_DeepFilterNetV3"), list)
            and "ROCMExecutionProvider" in report["aurik"].get("ort_DeepFilterNetV3", [])
        )
    return 0 if torch_ok and ort_ok and aurik_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
