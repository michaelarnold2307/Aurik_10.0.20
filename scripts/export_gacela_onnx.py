#!/usr/bin/env python3
"""Exportiert den GACELA-ML-Kern (BorderEncoder ×2 + Generator) nach ONNX.

§v10-GACELA-ONNX (2026-09-10): GACELA (GAN Audio Inpainting, 22,05 kHz) füllt
Spektrogramm-Lücken; der ML-Kern ist rein real (weight_norm-Conv2d, Linear,
ConvTranspose2d, Tanh) — ONNX-fähig. Die Mel-Filterbank-Projektion, die
Denormalisierung exp(25·(y−1)) und die PGHI-Phasenrekonstruktion bleiben im
Plugin (plugins/gacela_plugin.py).

I/O-Vertrag (statisch, Batch=1):
  ctx_l/ctx_r: [1, 1, 80, 240]  Mel-Spektrogramm der Ränder (TIME_AVG=2)
  noise:       [1, 4, 5, 15]    Zufallsvektor
  output:      [1, 1, 256, 256] Gap-Spektrogramm, Tanh ∈ [-1,1]
(Achtung: Docstring in gacela_plugin.inpaint nennt veraltete Shapes
[1,16,5,8]/[1,1,512,64] — real sind es [1,16,5,15] und [1,1,256,256].)

Parität: ONNX vs. eager Wrapper auf Zufallsdaten (rel < 1e-3).

Nutzung:
    .venv_aurik/bin/python scripts/export_gacela_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MDL_ROOT = _ROOT / "models" / "gacela"
_CKPT = _MDL_ROOT / "model" / "01_400000.pt"
_OUT = _MDL_ROOT / "model" / "gacela_core.onnx"


def _build_wrapper():
    import torch  # pylint: disable=import-outside-toplevel
    from torch import nn

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    if str(_MDL_ROOT) not in sys.path:
        sys.path.insert(0, str(_MDL_ROOT))

    from model.borderEncoder import BorderEncoder  # type: ignore
    from model.generator import Generator  # type: ignore

    from plugins.gacela_plugin import (  # type: ignore
        _BE_PARAMS,
        _GEN_IN_SHAPE,
        _GEN_PARAMS,
    )

    encoders = [BorderEncoder(_BE_PARAMS) for _ in range(2)]
    generator = Generator(_GEN_PARAMS, _GEN_IN_SHAPE)

    ckpt = torch.load(str(_CKPT), map_location="cpu", weights_only=True)
    generator.load_state_dict(ckpt["generator"])
    for enc, sd in zip(encoders, ckpt["encoders"]):
        enc.load_state_dict(sd)
    generator.eval()
    for enc in encoders:
        enc.eval()

    class GacelaCore(nn.Module):
        """ML-Kern: Mel-Kontexte + Noise → Gap-Spektrogramm."""

        def __init__(self, enc_l: nn.Module, enc_r: nn.Module, gen: nn.Module):
            super().__init__()
            self.enc_l = enc_l
            self.enc_r = enc_r
            self.gen = gen

        def forward(self, ctx_l: torch.Tensor, ctx_r: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            x = torch.cat([self.enc_l(ctx_l), self.enc_r(ctx_r), noise], dim=1)
            return self.gen(x)  # [1,1,256,256] ∈ [-1,1]

    wrapper = GacelaCore(encoders[0], encoders[1], generator).eval()
    return wrapper


def main() -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    wrapper = _build_wrapper()
    n_p = sum(p.numel() for p in wrapper.parameters()) / 1e6
    print(f"[gacela-export] Modell geladen ({n_p:.2f}M Parameter)")

    rng = torch.Generator().manual_seed(0)
    ctx_l = torch.rand(1, 1, 80, 240, generator=rng)
    ctx_r = torch.rand(1, 1, 80, 240, generator=rng) * 0.9
    noise = torch.rand(1, 4, 5, 15, generator=rng)

    with torch.no_grad():
        ref = wrapper(ctx_l, ctx_r, noise)
    print(f"[gacela-export] PyTorch-Referenz: {tuple(ref.shape)}")

    torch.onnx.export(
        wrapper,
        (ctx_l, ctx_r, noise),
        str(_OUT),
        opset_version=17,
        input_names=["ctx_l", "ctx_r", "noise"],
        output_names=["gap"],
        do_constant_folding=True,
    )
    print(f"[gacela-export] ONNX geschrieben: {_OUT} ({_OUT.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    sess = ort.InferenceSession(str(_OUT), providers=["CPUExecutionProvider"])
    out = sess.run(
        None,
        {
            "ctx_l": ctx_l.numpy(),
            "ctx_r": ctx_r.numpy(),
            "noise": noise.numpy(),
        },
    )[0]
    err = float(np.abs(out - ref.numpy()).max())
    rel = err / float(np.abs(ref.numpy()).max())
    print(f"[gacela-verify] ONNX vs PyTorch: max|Δ|={err:.3e} rel={rel:.3e}")
    assert rel < 1e-3, "Parität verletzt"
    print("[gacela-export] ERFOLG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
