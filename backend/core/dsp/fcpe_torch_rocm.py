"""§SOTA-ML-V7 — FCPE-Torch-ROCm-Kern (korrekter GPU-Pfad für Pitch-Extraktion).

Bug-Jagd 2026-09-18: Die ORT-ROCm-Kernels rechnen den FCPE-Conformer
numerisch falsch — Salience max|Δ| ≈ 0.04, rel ≈ 0.19 vs. ORT-CPU auf
echten Mel-Feeds (Softmax-/Attention-Kernels; 3. bestätigter ORT-ROCm-Defekt).
Deshalb bleibt der ONNX-Pfad auf CPU (Registry-Verdict "cpu").

Dieser Baustein liefert den hochwertigen GPU-Ersatz: CFNaiveMelPE aus
torchfcpe (models/fcpe/torchfcpe/, Quell-Import wie in
scripts/export_fcpe_onnx.py dokumentiert — spec_from_file_location umgeht
das __init__ mit pretty_midi-Dep; local_attention wird gemockt, da
local_heads=0 sie nie aktiviert). Gewichte aus models/fcpe/torchfcpe/
assets/fcpe.pt (dasselbe Checkpoint wie der ONNX-Export; Key-Remapping
stack.* → input_stack.*, decoder._layers.* → net.encoder_layers.*,
dense_out.* → output_proj.*). Parität vs. ONNX-CPU wird im GPU-Smoke und
Unit-Test nachgewiesen (rel ≤ 1e-3).

Fail-closed (§V6 (copilot-instructions.md)): Ohne torch/CUDA/Checkpoint
liefert get_fcpe_torch_core() None — der Aufrufer bleibt auf dem
ONNX-CPU-Pfad (FCPE → CREPE → pYIN-Kaskade bleibt intakt).

Status 2026-09-18: GELÖST — das Aurik-Checkpoint ``models/fcpe/fcpe.pt``
(input_channel=128, n_chans=512, n_layers=6, out_dims=360; mit Attention)
ist wieder verfügbar; der Kern lädt es direkt (Fallback: torchfcpe/assets/).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_FCPE_DIR = _ROOT / "models" / "fcpe"
# Primär: models/fcpe/fcpe.pt (Aurik-Checkpoint); historisch lag es unter
# torchfcpe/assets/ (Fallback bleibt erhalten).
_CHECKPOINT = _FCPE_DIR / "fcpe.pt"
_CHECKPOINT_LEGACY = _FCPE_DIR / "torchfcpe" / "assets" / "fcpe.pt"

_lock = threading.Lock()
_core = None
_core_resolved = False


def _build_core():
    import importlib.util as _ilu
    import sys as _sys
    import types as _types

    import torch  # pylint: disable=import-outside-toplevel

    if not _CHECKPOINT.exists() and not _CHECKPOINT_LEGACY.exists():
        raise RuntimeError("FCPE-Checkpoint fehlt (models/fcpe/fcpe.pt) — ONNX-CPU-Pfad bleibt")
    _ckpt_path = _CHECKPOINT if _CHECKPOINT.exists() else _CHECKPOINT_LEGACY

    # Quell-Import wie im Export-Skript (§7.1 export_fcpe_onnx.py).
    _mock_la = _types.ModuleType("local_attention")
    _mock_la.LocalAttention = type(  # type: ignore[attr-defined]
        "LocalAttention", (object,), {"__init__": lambda *a, **kw: None}
    )
    _sys.modules.setdefault("local_attention", _mock_la)

    def _load_mod(name: str, path: Path):
        spec = _ilu.spec_from_file_location(name, path)
        mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    _pkg = _types.ModuleType("torchfcpe")
    _sys.modules.setdefault("torchfcpe", _pkg)
    _load_mod("torchfcpe.model_conformer_naive", _FCPE_DIR / "torchfcpe" / "model_conformer_naive.py")
    _tfm = _load_mod("torchfcpe.models", _FCPE_DIR / "torchfcpe" / "models.py")

    ckpt = torch.load(str(_ckpt_path), map_location="cpu")
    config = ckpt["config"]
    model_cfg = config["model"]
    n_chans = int(model_cfg["n_chans"])  # 512
    model = _tfm.CFNaiveMelPE(
        input_channels=int(model_cfg["input_channel"]),
        out_dims=int(model_cfg["out_dims"]),
        hidden_dims=n_chans,
        n_layers=int(model_cfg["n_layers"]),
        n_heads=n_chans // 64,
        f0_max=float(model_cfg["f0_max"]),
        f0_min=float(model_cfg["f0_min"]),
        use_fa_norm=False,
        conv_only=False,
        conv_dropout=0.0,
        atten_dropout=0.0,
    )
    new_sd: dict[str, object] = {}
    for k, v in ckpt["model"].items():
        if k.startswith("stack."):
            new_sd["input_stack." + k[len("stack.") :]] = v
        elif k.startswith("decoder._layers."):
            new_sd["net.encoder_layers." + k[len("decoder._layers.") :]] = v
        elif k.startswith("dense_out."):
            new_sd["output_proj." + k[len("dense_out.") :]] = v
        else:
            new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    critical = [k for k in missing if not k.startswith("gaussian_blurred")]
    if critical:
        raise RuntimeError(f"FCPE-Checkpoint: kritische Keys fehlen {critical}")
    if unexpected:
        logger.debug("FCPE-Checkpoint: unerwartete Keys ignoriert: %s", list(unexpected)[:5])
    return model.eval()


def get_fcpe_torch_core():
    """Lazy-Singleton: CFNaiveMelPE auf ROCm oder None (fail-closed, §V6 (copilot-instructions.md))."""
    global _core, _core_resolved
    with _lock:
        if _core_resolved:
            return _core
        _core_resolved = True
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if not torch.cuda.is_available():
                logger.debug("§SOTA-ML-V7 FCPE-Torch-ROCm nicht verfügbar — ONNX-CPU-Pfad bleibt")
                return None
            _core = _build_core().to("cuda")
            logger.info("§SOTA-ML-V7 FCPE-Kern auf ROCm geladen (%s)", torch.cuda.get_device_name(0))
            return _core
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("§SOTA-ML-V7 FCPE-Torch-Kern nicht ladbar: %s — ONNX-CPU-Pfad bleibt", exc)
            return None


def salience_fcpe_torch(core, mel: np.ndarray) -> np.ndarray:
    """FCPE-Vorwärtslauf: Mel → Salience (Wahrscheinlichkeiten über 360 Pitch-Klassen).

    Args:
        core: Modul aus get_fcpe_torch_core() (eval, auf cuda).
        mel: float32 [B, T, 128] (FCPE-Mel, wie der ONNX-Eingang).

    Returns: float32 [B, T, 360] in [0, 1], NaN/Inf-geschützt
    (§0a (copilot-instructions.md)). Deterministisch.
    """
    import torch  # pylint: disable=import-outside-toplevel

    m = np.asarray(mel, dtype=np.float32)
    if m.ndim != 3 or m.shape[2] != 128:
        raise ValueError(f"FCPE-Torch: erwartet [B,T,128]-Mel, bekam {m.shape}")
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        out = core(torch.from_numpy(m).to(next(core.parameters()).device)).cpu().numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)  # type: ignore[no-any-return]
