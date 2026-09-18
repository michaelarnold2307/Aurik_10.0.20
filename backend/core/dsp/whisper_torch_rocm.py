"""§SOTA-ML-V6 — Whisper-Tiny-Torch-ROCm-Encoder (korrekter GPU-Pfad).

Bug-Jagd 2026-09-18: Die ORT-ROCm-Kernels rechnen den Whisper-Encoder
numerisch falsch — auf echten Mel-Feeds rel ≈ 0.98 (max|Δ| 13.5) mit
Default-Optimierern und rel ≈ 0.68 mit ORT_DISABLE_ALL (Softmax-/Attention-
Kernels; 5. bestätigter ORT-ROCm-Defekt). MIGraphX platziert für dieses
Modell nicht. Deshalb wurde WhisperTiny/WhisperTurbo aus der GPU-Liste
genommen — der CPU-Pfad ist korrekt, aber langsam (98 ms je 30-s-Fenster).

Dieser Baustein liefert den hochwertigen GPU-Ersatz: den HF-Whisper-Tiny-
Encoder (transformers, Gewichte identisch zum ONNX-Export — Parität
torch vs. ONNX-CPU: max|Δ| ≈ 2.6e-3, rel ≈ 1.5e-4) auf PyTorch-ROCm:
~6.9 ms je 30-s-Fenster (7900 XTX, ~14× schneller als der korrekte
CPU-Pfad), deterministisch bit-identisch (§G5 (copilot-instructions.md)).

Fail-closed (§V6 (copilot-instructions.md)): Ohne torch/CUDA/transformers
oder ohne lokal gecachten HF-Snapshot liefert get_whisper_torch_core()
None — der Aufrufer bleibt auf dem ONNX-CPU-Pfad.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# HF-Snapshot-Name (lokal gecacht, deterministisch — kein Netz zur Laufzeit).
_SNAPSHOT_PREFIX = "models--openai--whisper-tiny"

_lock = threading.Lock()
_core = None
_core_resolved = False


def _find_snapshot() -> str | None:
    """Sucht den lokal gecachten HF-Snapshot (ohne Netz)."""
    from pathlib import Path

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for entry in hub.glob(f"{_SNAPSHOT_PREFIX}/snapshots/*"):
        if (entry / "model.safetensors").exists() and (entry / "config.json").exists():
            return str(entry)
    return None


def _build_core():
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import WhisperFeatureExtractor, WhisperModel  # pylint: disable=import-outside-toplevel

    snapshot = _find_snapshot()
    if snapshot is None:
        raise RuntimeError("Whisper-Tiny-Snapshot nicht lokal gecacht (HF-Hub) — ONNX-CPU-Pfad bleibt")
    model = WhisperModel.from_pretrained(snapshot, local_files_only=True).eval()  # nosec B615 — commit-gepinnter Snapshot, kein Netz
    extractor = WhisperFeatureExtractor.from_pretrained(snapshot, local_files_only=True)  # nosec B615
    return {"model": model, "extractor": extractor}


def get_whisper_torch_core() -> dict | None:
    """Lazy-Singleton: {model, extractor} auf ROCm oder None
    (fail-closed, §V6 (copilot-instructions.md))."""
    global _core, _core_resolved
    with _lock:
        if _core_resolved:
            return _core
        _core_resolved = True
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if not torch.cuda.is_available():
                logger.debug("§SOTA-ML-V6 Whisper-Torch-ROCm nicht verfügbar — ONNX-CPU-Pfad bleibt")
                return None
            bundle = _build_core()
            bundle["model"] = bundle["model"].to("cuda")
            _core = bundle
            logger.info("§SOTA-ML-V6 Whisper-Tiny-Torch-Encoder auf ROCm geladen (%s)", torch.cuda.get_device_name(0))
            return _core  # type: ignore[no-any-return]
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("§SOTA-ML-V6 Whisper-Torch-Kern nicht ladbar: %s — ONNX-CPU-Pfad bleibt", exc)
            return None


def encode_whisper_torch_mel(core: dict, mel: np.ndarray) -> np.ndarray:
    """Encoder-Vorwärtslauf auf einem bereits berechneten Whisper-Mel.

    Nutzt die Mel-Pipeline des Aufrufers (identisch zum ONNX-Pfad), nur der
    Encoder-Backend wechselt → minimaler Eingriff in die Pipeline-Parität.

    Args:
        core: Bundle aus get_whisper_torch_core().
        mel: float32 [1, 80, 3000] (Whisper-Log-Mel, gepaddet auf 30 s).

    Returns: float32 [1, T, 384], NaN/Inf-geschützt
    (§0a (copilot-instructions.md)). Deterministisch.
    """
    import torch  # pylint: disable=import-outside-toplevel

    m = np.asarray(mel, dtype=np.float32)
    if m.ndim != 3 or m.shape[1] != 80:
        raise ValueError(f"Whisper-Torch: erwartet [B,80,T]-Mel, bekam {m.shape}")
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        out = (
            core["model"]
            .encoder(torch.from_numpy(m).to(next(core["model"].parameters()).device))
            .last_hidden_state.cpu()
            .numpy()
        )
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)  # type: ignore[no-any-return]


def encode_whisper_torch(core: dict, audio_16k: np.ndarray) -> np.ndarray:
    """Whisper-Encoder-Vorwärtslauf: 16-kHz-Mono → last_hidden_state.

    Args:
        core: Bundle aus get_whisper_torch_core().
        audio_16k: float32 Mono-Signal mit 16 kHz.

    Returns: float32 [1, T, 384] (Whisper-Tiny, T = n_frames/2),
    NaN/Inf-geschützt (§0a (copilot-instructions.md)). Deterministisch.
    """
    import torch  # pylint: disable=import-outside-toplevel

    x = np.asarray(audio_16k, dtype=np.float32)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"Whisper-Torch: erwartet 16-kHz-Mono (1-D), bekam {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    feats = core["extractor"](x, sampling_rate=16000, return_tensors="pt")
    mel = feats.input_features.to(next(core["model"].parameters()).device)
    with torch.no_grad():
        out = core["model"].encoder(mel).last_hidden_state.cpu().numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)  # type: ignore[no-any-return]
