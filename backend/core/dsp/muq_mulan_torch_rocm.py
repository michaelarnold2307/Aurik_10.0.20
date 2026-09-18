"""§SOTA-ML-V8 — MuQ-MuLan-Torch-ROCm-Kern (korrekter GPU-Pfad für den 768-d-Witness).

Bug-Jagd 2026-09-18: Die ORT-ROCm-Kernels rechnen den MuQ-MuLan-Audio-Turm
numerisch falsch — Audio-Embedding max|Δ| ≈ 2.8, rel ≈ 0.59 vs. ORT-CPU auf
echten Musik-Feeds (Softmax-/Attention-Kernels; 6. bestätigter ORT-ROCm-Defekt).
Der ONNX-Pfad bleibt deshalb auf CPU (Registry-Verdict „cpu").

Dieser Baustein liefert den hochwertigen GPU-Ersatz: der Audio-Turm aus
plugins/_vendor_muq/muq_mulan (AudioSpectrogramTransformerPretrained, exakt
wie scripts/export_muq_mulan_onnx.py aufbaut) mit Gewichten aus den LOKALEN
Artefakten models/muq_mulan/mulan/pytorch_model.bin (mulan.audio.*) und dem
MuQ-Encoder aus models/muq_mulan/ (model.safetensors). Kein Netz zur Laufzeit.
Parität vs. ONNX-CPU wird im GPU-Smoke/Unit-Test nachgewiesen (rel ≤ 1e-3,
dieselbe Schwelle wie im Export-Skript).

Fail-closed (§V6 (copilot-instructions.md)): Ohne torch/CUDA/Checkpoints
liefert get_muq_mulan_torch_core() None — der Aufrufer bleibt auf dem
ONNX-CPU-Pfad.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_MUQ_DIR = _ROOT / "models" / "muq_mulan"
_MULAN_CKPT = _MUQ_DIR / "mulan" / "pytorch_model.bin"
_SR = 24000
_CLIP_S = 10.0
_N_SAMPLES = int(_SR * _CLIP_S)  # 240000

_lock = threading.Lock()
_core = None
_core_resolved = False


def _build_core():
    import torch  # pylint: disable=import-outside-toplevel

    from plugins._vendor_muq.muq_mulan.models.audio import (  # pylint: disable=import-outside-toplevel
        AudioSpectrogramTransformerPretrained,
    )

    if not _MULAN_CKPT.is_file():
        raise RuntimeError(
            "MuQ-MuLan-Checkpoint fehlt (models/muq_mulan/mulan/pytorch_model.bin) — ONNX-CPU-Pfad bleibt"
        )

    tower = AudioSpectrogramTransformerPretrained(
        model_name=str(_MUQ_DIR),  # 'muq' im Namen → lokaler MuQ-Encoder via from_pretrained(local_dir)
        dim=768,
        model_dim=1024,
        sr=_SR,
        tf_depth=0,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        use_layer_idx=-1,
        frozen_pretrained=True,
    )
    state = torch.load(str(_MULAN_CKPT), map_location="cpu", weights_only=False)
    prefix = "mulan.audio."
    audio_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not audio_state:
        raise RuntimeError("MuQ-MuLan-Checkpoint: keine mulan.audio.*-Keys — Struktur unerwartet")
    missing, unexpected = tower.load_state_dict(audio_state, strict=False)
    if missing:
        raise RuntimeError(f"MuQ-MuLan-Checkpoint: kritische Keys fehlen {list(missing)[:6]}")
    if unexpected:
        logger.debug("MuQ-MuLan-Checkpoint: unerwartete Keys ignoriert: %s", list(unexpected)[:5])
    del state
    tower.eval()
    for p in tower.parameters():
        p.requires_grad_(False)
    return tower


def get_muq_mulan_torch_core():
    """Lazy-Singleton: Audio-Turm auf ROCm oder None (fail-closed, §V6 (copilot-instructions.md))."""
    global _core, _core_resolved
    with _lock:
        if _core_resolved:
            return _core
        _core_resolved = True
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if not torch.cuda.is_available():
                logger.debug("§SOTA-ML-V8 MuQ-MuLan-Torch-ROCm nicht verfügbar — ONNX-CPU-Pfad bleibt")
                return None
            _core = _build_core().to("cuda")
            logger.info("§SOTA-ML-V8 MuQ-MuLan-Audio-Turm auf ROCm geladen (%s)", torch.cuda.get_device_name(0))
            return _core
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("§SOTA-ML-V8 MuQ-MuLan-Torch-Kern nicht ladbar: %s — ONNX-CPU-Pfad bleibt", exc)
            return None


def embed_muq_mulan_torch(core, waveform: np.ndarray) -> np.ndarray:
    """Audio-Turm-Vorwärtslauf: 24-kHz-Waveform → 768-d-Audio-Embedding.

    Args:
        core: Modul aus get_muq_mulan_torch_core() (eval, auf cuda).
        waveform: float32 Mono-Signal, 240000 Samples (10 s @ 24 kHz).

    Returns: float32 [1, 768], NaN/Inf-geschützt
    (§0a (copilot-instructions.md)). Deterministisch.
    """
    import torch  # pylint: disable=import-outside-toplevel

    w = np.asarray(waveform, dtype=np.float32)
    if w.ndim == 1:
        w = w[None, :]
    if w.ndim != 2 or w.shape[1] != _N_SAMPLES:
        raise ValueError(f"MuQ-MuLan-Torch: erwartet [B,{_N_SAMPLES}]-Waveform, bekam {w.shape}")
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        out = core(torch.from_numpy(w).to(next(core.parameters()).device)).cpu().numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)  # type: ignore[no-any-return]
