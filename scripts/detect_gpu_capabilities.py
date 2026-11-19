"""GPU-Fähigkeiten-Erkennung für Aurik.

§15.5: Erkennt verfügbare GPUs und deren ML-Backends
(CUDA, MPS, ROCm, DirectML, CPU). Exportiert nach gpu_capabilities.json.

Nutzung:
    python scripts/detect_gpu_capabilities.py
    python scripts/detect_gpu_capabilities.py --json
    python scripts/detect_gpu_capabilities.py --json --output capabilities.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Pfad sicherstellen
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.ml.backend_router import MLEngineConfig, detect_gpu_capabilities

logger = logging.getLogger(__name__)


def _check_library(name: str, import_path: str) -> dict:
    """Prüft ob eine Bibliothek importierbar ist."""
    try:
        __import__(import_path)
        return {"name": name, "available": True, "error": None}
    except ImportError as e:
        return {"name": name, "available": False, "error": str(e)}


def _estimate_vram_mb(config: MLEngineConfig) -> int | None:
    """Schätzt verfügbaren VRAM (nur für CUDA/ROCm realistisch)."""
    provider = config.provider
    if provider == "cpu":
        return None
    try:
        if provider == "cuda":
            import onnxruntime as ort  # type: ignore

            # CUDA-VRAM via onnxruntime ist begrenzt — Fallback auf Treiber-API
            return None  # onnxruntime exponiert VRAM nicht direkt
        if provider in ("rocm", "mps", "directml"):
            return None
    except Exception as _e:
        logger.debug("%s: non-critical exception: %s", __name__, _e)
    return None


def _get_onnxruntime_info() -> dict:
    """ONNX-Runtime-Version und verfügbare Provider."""
    try:
        import onnxruntime as ort

        return {
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
            "available": True,
        }
    except ImportError:
        return {"version": None, "providers": [], "available": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU-Fähigkeiten erkennen")
    parser.add_argument("--json", action="store_true", help="Nur JSON ausgeben")
    parser.add_argument("--output", default="gpu_capabilities.json", help="Ausgabedatei")
    args = parser.parse_args()

    # ── GPU erkennen ──────────────────────────────────────────────────────
    config = detect_gpu_capabilities()

    # ── Bibliotheken prüfen ───────────────────────────────────────────────
    libraries = [
        _check_library("onnxruntime", "onnxruntime"),
        _check_library("onnxruntime-gpu (CUDA)", "onnxruntime.transformers"),
        _check_library("torch (PyTorch)", "torch"),
        _check_library("torch.cuda", "torch.cuda"),
        _check_library("coremltools (Apple)", "coremltools"),
    ]

    ort_info = _get_onnxruntime_info()
    vram_mb = _estimate_vram_mb(config)

    # ── Ergebnis bauen ────────────────────────────────────────────────────
    result = {
        "engine_config": asdict(config),
        "onnxruntime": ort_info,
        "libraries": libraries,
        "estimated_vram_mb": vram_mb,
        "summary": {
            "recommended_backend": config.provider.upper(),
            "gpu_available": config.provider != "cpu",
            "onnx_providers": config.onnx_providers,
            "fallback_to_cpu": config.fallback_to_cpu,
        },
    }

    output_path = _PROJECT_ROOT / args.output
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(result["summary"], indent=2))
        return 0

    # ── Konsolen-Ausgabe ──────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print("  Aurik GPU-Fähigkeiten-Erkennung")
    print(f"{sep}")
    print(f"  Erkanntes Backend:       {config.provider.upper()}")
    print(f"  GPU verfügbar:           {'✅ Ja' if config.provider != 'cpu' else '❌ Nein (CPU)'}")
    print(f"  ONNX-Provider:           {', '.join(config.onnx_providers)}")
    print(f"  CPU-Fallback:            {'✅ Aktiv' if config.fallback_to_cpu else '❌ Aus'}")
    print(f"  ONNX-Runtime:            {ort_info['version'] or 'Nicht installiert'}")
    print(f"  VRAM geschätzt:          {vram_mb if vram_mb else 'N/A'}")
    print(f"{sep}\n")

    print("## Verfügbare Bibliotheken\n")
    for lib in libraries:
        status = "✅" if lib["available"] else "❌"
        error = f" ({lib['error']})" if lib["error"] else ""
        print(f"  {status} {lib['name']}{error}")

    print(f"\n📄 JSON-Report: {output_path}")
    print("✅ GPU-Erkennung abgeschlossen.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
