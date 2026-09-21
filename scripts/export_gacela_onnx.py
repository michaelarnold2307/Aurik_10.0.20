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

# F2-Vocal-Finetune (§v10.25, train_gacela_vocal_inpaint.py, md=32):
# anderer Generator-Input (1440) + Border-nfilter; Checkpoint speichert kein cfg.
_FT_GEN_PARAMS = {
    "stride": [2, 2, 2, 2, 2],
    "nfilter": [256, 128, 64, 32, 1],
    "shape": [[4, 4], [4, 4], [8, 8], [8, 8], [8, 8]],
    "padding": [[1, 1], [1, 1], [3, 3], [3, 3], [3, 3]],
    "residual_blocks": 2,
    "full": 8192,
    "summary": True,
    "data_size": 2,
    "in_conv_shape": [16, 2],
}
_FT_BORDER_PARAMS = {
    "nfilter": [32, 64, 32, 16],
    "shape": [[5, 5], [5, 5], [5, 5], [5, 5]],
    "stride": [2, 2, 2, 2],
    "data_size": 2,
    "border_scale": 1,
    "width_full": None,
}
_FT_GENERATOR_INPUT = 1440


def _build_wrapper(ckpt: Path):
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

    ckpt_raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    generator.load_state_dict(ckpt_raw["generator"])
    for enc, sd in zip(encoders, ckpt_raw["encoders"]):
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


def _build_ft_wrapper(ckpt: Path):
    """F2-Vocal-Finetune-Kern (md=32): BorderEncoder ×2 + Generator(1440).

    I/O (statisch, Batch=1, gemessen 2026-09-20):
      ctx_l/ctx_r: [1, 1, 80, 120]  Mel der 480-Frame-Ränder nach time_average(4)
      noise:       [1, 4, 5, 8]
      output:      [1, 1, 512, 64]  Gap-Spektrogramm (GAP_BINS=64), Tanh ∈ [-1,1]
    """
    import torch  # pylint: disable=import-outside-toplevel
    from torch import nn

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    if str(_MDL_ROOT) not in sys.path:
        sys.path.insert(0, str(_MDL_ROOT))

    from model.borderEncoder import BorderEncoder  # type: ignore
    from model.generator import Generator  # type: ignore

    encoders = [BorderEncoder(_FT_BORDER_PARAMS) for _ in range(2)]
    generator = Generator(_FT_GEN_PARAMS, _FT_GENERATOR_INPUT)

    ckpt_raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    generator.load_state_dict(ckpt_raw["generator"])
    for enc, sd in zip(encoders, ckpt_raw["encoders"]):
        enc.load_state_dict(sd)
    generator.eval()
    for enc in encoders:
        enc.eval()

    class GacelaFtCore(nn.Module):
        def __init__(self, enc_l: nn.Module, enc_r: nn.Module, gen: nn.Module):
            super().__init__()
            self.enc_l = enc_l
            self.enc_r = enc_r
            self.gen = gen

        def forward(self, ctx_l: torch.Tensor, ctx_r: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            x = torch.cat([self.enc_l(ctx_l), self.enc_r(ctx_r), noise], dim=1)
            return self.gen(x)  # [1,1,512,64] ∈ [-1,1]

    wrapper = GacelaFtCore(encoders[0], encoders[1], generator).eval()
    return wrapper, generator


def main() -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    # argv-Übersteuerung für Fine-Tune-Checkpoints (§v10.25 Export-only):
    #   python scripts/export_gacela_onnx.py <checkpoint.pt> [output.onnx] [ft]
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else _CKPT
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else _OUT
    ft_variant = len(sys.argv) > 3 and sys.argv[3] == "ft"

    if ft_variant:
        wrapper, gen = _build_ft_wrapper(ckpt)
    else:
        wrapper = _build_wrapper(ckpt)
    n_p = sum(p.numel() for p in wrapper.parameters()) / 1e6
    print(f"[gacela-export] Modell geladen ({n_p:.2f}M Parameter, Checkpoint: {ckpt})")

    rng = torch.Generator().manual_seed(0)
    if ft_variant:
        # F2-Geometrie: Mel 80 × 120 (480 Frames / avg 4), Noise [1,4,5,8]
        ctx_l = torch.rand(1, 1, 80, 120, generator=rng)
        ctx_r = torch.rand(1, 1, 80, 120, generator=rng) * 0.9
        noise = torch.rand(1, 4, 5, 8, generator=rng)
    else:
        ctx_l = torch.rand(1, 1, 80, 240, generator=rng)
        ctx_r = torch.rand(1, 1, 80, 240, generator=rng) * 0.9
        noise = torch.rand(1, 4, 5, 15, generator=rng)

    with torch.no_grad():
        ref = wrapper(ctx_l, ctx_r, noise)
    print(f"[gacela-export] PyTorch-Referenz: {tuple(ref.shape)}")

    # Dynamo-Export (statisch): weight_norm wird via dynamo zerlegt (wie der
    # Basis-Export vom 2026-09-10, vgl. FILE_REGISTRY); Legacy-Tracer scheitert
    # an weight_norm in Torch 2.11.
    exported = torch.export.export(wrapper, (ctx_l, ctx_r, noise))
    onnx_program = torch.onnx.export(exported, dynamo=True)
    onnx_program.save(str(out_path))
    print(f"[gacela-export] ONNX geschrieben: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
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
