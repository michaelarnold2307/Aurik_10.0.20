"""
Export MP-SENet generator checkpoint to ONNX.

Loads best_ckpt/g_best_vb (VoiceBank+DEMAND) and exports to
models/mp_senet/mp_senet.onnx with dynamic T-axis.

Usage:
    python scripts/export_mp_senet_onnx.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
MP_SENET_DIR = REPO_ROOT / "models" / "mp_senet"
CHECKPOINT = MP_SENET_DIR / "best_ckpt" / "g_best_vb"
CONFIG_FILE = MP_SENET_DIR / "best_ckpt" / "config.json"
OUTPUT_ONNX = MP_SENET_DIR / "mp_senet.onnx"

# Add mp_senet source to path
sys.path.insert(0, str(MP_SENET_DIR))


# ---------------------------------------------------------------------------
# Minimal AttrDict + model import
# ---------------------------------------------------------------------------
class AttrDict(dict):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, item: str) -> Any:
        return self[item]


def load_config(path: Path) -> AttrDict:
    with open(path) as f:
        return AttrDict(json.loads(f.read()))


# ---------------------------------------------------------------------------
# Monkey-patch: inject LearnableSigmoid2d without loading matplotlib/pesq
# utils.py imports matplotlib (broken under NumPy 2.x) only for plot helpers —
# the class itself is pure torch, so we inject it directly into sys.modules.
# ---------------------------------------------------------------------------
import types as _types


def _make_utils_stub() -> _types.ModuleType:
    import torch as _torch
    import torch.nn as _nn

    mod = _types.ModuleType("utils")

    class LearnableSigmoid2d(_nn.Module):
        def __init__(self, in_features: int, beta: float = 1.0) -> None:
            super().__init__()
            self.beta = beta
            self.slope = _nn.Parameter(_torch.ones(in_features, 1))
            self.slope.requiresGrad = True  # type: ignore[attr-defined]

        def forward(self, x: _torch.Tensor) -> _torch.Tensor:
            return self.beta * _torch.sigmoid(self.slope * x)

    mod.LearnableSigmoid2d = LearnableSigmoid2d  # type: ignore[attr-defined]
    return mod


sys.modules.setdefault("utils", _make_utils_stub())

# pesq is only used in training helpers (pesq_score / eval_pesq), not in forward()
try:
    pass
except ImportError:
    sys.modules.setdefault("pesq", _types.ModuleType("pesq"))


def build_model(h: AttrDict) -> nn.Module:
    from models.model import MPNet

    return MPNet(h)  # type: ignore[no-any-return]


def export(args: argparse.Namespace) -> None:
    checkpoint: Path = args.checkpoint
    config_file: Path = args.config
    output_onnx: Path = args.output

    print(f"Config:     {config_file}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output:     {output_onnx}")

    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        sys.exit(1)

    h = load_config(config_file)

    model = build_model(h)
    state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    # VoiceBank-Releases nutzen {"generator": …}; Aurik-Finetune (§v10.17) nutzt
    # {"model_state_dict": …} (train_mp_senet_musik.py).
    if "generator" in state_dict:
        model.load_state_dict(state_dict["generator"])
    elif "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    print("✅ Checkpoint loaded")

    # Dummy inputs: [B=1, F=n_fft//2+1, T=32]
    F = h.n_fft // 2 + 1  # 201
    T = 32
    noisy_amp = torch.randn(1, F, T)
    noisy_pha = torch.randn(1, F, T)

    with torch.no_grad():
        amp_out, pha_out, com_out = model(noisy_amp, noisy_pha)
    print(f"Forward OK — amp_out shape: {amp_out.shape}")

    print("Exporting to ONNX …")
    torch.onnx.export(
        model,
        (noisy_amp, noisy_pha),
        str(output_onnx),
        input_names=["noisy_amp", "noisy_pha"],
        output_names=["denoised_amp", "denoised_pha", "denoised_com"],
        dynamic_axes={
            "noisy_amp": {0: "batch", 2: "time"},
            "noisy_pha": {0: "batch", 2: "time"},
            "denoised_amp": {0: "batch", 2: "time"},
            "denoised_pha": {0: "batch", 2: "time"},
            "denoised_com": {0: "batch", 2: "time"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # Legacy-Exporter (Torch 2.11): MPNet-DenseBlock ist dynamo-inkompatibel
    )

    size_mb = output_onnx.stat().st_size / 1024 / 1024
    print(f"✅ Exported: {output_onnx}  ({size_mb:.1f} MB)")

    # Quick validation + Parität Torch vs. ONNX (§III.9/§v10.18)
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
        out = sess.run(
            None,
            {
                "noisy_amp": noisy_amp.numpy(),
                "noisy_pha": noisy_pha.numpy(),
            },
        )
        with torch.no_grad():
            ref = [t.numpy() for t in model(noisy_amp, noisy_pha)]
        for r, o, name in zip(ref, out, ("amp", "pha", "com")):
            diff = float(np.abs(r - o).max())
            print(f"  Parität {name}: max|Δ| = {diff:.3e}")
        print(f"✅ ONNX validation OK — output shapes: {[o.shape for o in out]}")
    except ImportError:
        print("onnxruntime nicht installiert — Validierung übersprungen")
    except Exception as e:
        print(f"⚠️  ONNX validation error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MP-SENet generator checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_ONNX)
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
