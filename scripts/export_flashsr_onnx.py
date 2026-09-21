#!/usr/bin/env python3
"""Exportiert den FlashSR/FastAudioSR-Kern (16 → 48 kHz) nach ONNX.

§v10.25 Export-only: f4-Finetune-Checkpoint (output/_training_archive_20260920/
f4_flashsr/best.pt, 2026-09-19) ist neuer als models/flashsr/flashsr.onnx
(2026-08-10) — Training erübrigt sich, nur Export + A/B (§6 der Spec).

Checkpoint-Format: FASR (models/flashsr/FastAudioSR) lädt torch.load(ckpt)["model"]
— das f4-Checkpoint-Format ({"model": …}) entspricht exakt dem FASR-Format.

Usage:
    venv_rocm72/bin/python scripts/export_flashsr_onnx.py \
        --checkpoint output/_training_archive_20260920/f4_flashsr/best.pt \
        --output models/flashsr/flashsr_f4.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_FLASHSR_DIR = _ROOT / "models" / "flashsr"


def main() -> int:
    parser = argparse.ArgumentParser(description="FlashSR/FastAudioSR → ONNX (§v10.25)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_FLASHSR_DIR / "FastAudioSR" / "SR48k.pth",
        help="FASR-Checkpoint (Default: upstream SR48k.pth)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_FLASHSR_DIR / "flashsr.onnx",
        help="Ziel-ONNX (Default: models/flashsr/flashsr.onnx)",
    )
    parser.add_argument("--seq", type=int, default=16000, help="Dummy-Sequenzlänge (16 kHz)")
    parser.add_argument(
        "--static",
        action="store_true",
        help="Statischer Export (keine dynamischen Achsen) — Deployment-Modus mit "
        "Plugin-Padding auf _FLASHSR_CHUNK_16K+Overlap",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint nicht gefunden: {args.checkpoint}")

    sys.path.insert(0, str(_FLASHSR_DIR))
    from FastAudioSR import FASR  # pylint: disable=import-outside-toplevel

    fasr = FASR(str(args.checkpoint))
    model = fasr.model.eval().to("cpu")  # FASR lädt auf cuda:0 — Export deterministisch auf CPU

    # Lite-Rezeptur aus models/flashsr/onnx_conversion.py: UpSample1d → F.interpolate
    # (dynamische Zeitachse, ~2× schneller; nötig, weil das Plugin variable
    # Chunks 64000+Overlap speist — statische Exports brechen am End-Chunk).
    class FastUpSample1d(torch.nn.Module):
        def __init__(self, scale_factor: int = 2):
            super().__init__()
            self.scale_factor = scale_factor

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(
                x, scale_factor=float(self.scale_factor), mode="linear", align_corners=False
            )

    def replace_upsample(module: torch.nn.Module) -> None:
        for name, child in list(module.named_children()):
            if child.__class__.__name__ == "UpSample1d":
                setattr(module, name, FastUpSample1d())
            else:
                replace_upsample(child)

    replace_upsample(model)

    # Replicate-Pad des LowPassFilters äquivalent über cat/expand ersetzen:
    # Torch 2.11 faltet onnx::Pad-Konstanten im Legacy-Tracer nicht mehr
    # (Export-Abbruch), Dynamo spezialisiert seq hart. Die cat-Variante ist für
    # pad < Länge (immer gegeben: pad ≤ 6 < T) numerisch identisch zu
    # F.pad(mode="replicate") und tracet in beiden Exportern sauber.
    # Nur für den (blockierten) Dynamik-Versuch nötig — statischer Dynamo-Export
    # erzielt mit dem Original-Graphen die beste Parität (8.7e-5).
    if not args.static:
        import torch.nn.functional as _F
        from FastAudioSR.filter import LowPassFilter1d as _LPF

        def _export_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            if self.padding:
                channels = x.shape[1]
                # Concat fester Anzahl statt expand/repeat: symbolische Expand-Formen
                # scheitern im Legacy-Tracer (Produktionsbefund 2026-09-20).
                pad_l = torch.cat([x[:, :, :1]] * int(self.pad_left), dim=-1)
                pad_r = torch.cat([x[:, :, -1:]] * int(self.pad_right), dim=-1)
                x = torch.cat([pad_l, x, pad_r], dim=-1)
            return _F.conv1d(
                x,
                self.filter if channels == 1 else self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            )

        _LPF.forward = _export_forward  # type: ignore[method-assign]

        # speechsr.Generator: int(x.shape[-1] * 3) backt die Länge in den Graphen
        # (dynamo spezialisiert, legacy bäckt statisch). scale_factor=3 ist für
        # Vielfache von T numerisch identisch und symbolisch sauber.
        from FastAudioSR.speechsr import Generator as _Gen

        def _export_gen_forward(self: torch.nn.Module, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
            x = self.conv_pre(x)
            if g is not None:
                x = x + self.cond(g)
            x = _F.interpolate(x, scale_factor=3, mode="linear")
            xs = self.resblocks[2](x)
            xs += self.resblocks[0](x)
            xs = xs / 2
            xs = self.activation_post(xs)
            x = self.conv_post(xs)
            return torch.tanh(x)

        _Gen.forward = _export_gen_forward  # type: ignore[method-assign]
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[flashsr-export] Modell geladen ({n_p:.2f}M Parameter, {args.checkpoint})")

    dummy = torch.randn(1, 1, args.seq)
    with torch.no_grad():
        ref = model(dummy)
    print(f"[flashsr-export] PyTorch-Referenz: {tuple(ref.shape)}")

    # Export: statisch → Dynamo (exakte Parität, war 8.7e-5); dynamische Achsen
    # scheitern in Torch 2.11 an symbolischen Pad/Expand/Slice-Formen (dokumentierte
    # Regressionen) — Deployment-Pfad ist daher statisch bei
    # _FLASHSR_CHUNK_16K+Overlap (68000) + Plugin-Padding.
    if args.static:
        exported = torch.export.export(model, (dummy,))
        onnx_program = torch.onnx.export(exported, dynamo=True)
        onnx_program.save(str(args.output))
    else:
        torch.onnx.export(
            model,
            (dummy,),
            str(args.output),
            input_names=["x"],
            output_names=["audio_48k"],
            dynamic_axes={"x": {0: "batch", 2: "time"}, "audio_48k": {0: "batch", 2: "time"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[flashsr-export] ONNX geschrieben: {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")

    # Parität ONNX vs. PyTorch (§III.9/§v10.18)
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    # Dynamo-Export benennt den Input nach dem PyTorch-Arg-Namen.
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: dummy.numpy()})[0]
    err = float(np.abs(out - ref.numpy()).max())
    rel = err / float(np.abs(ref.numpy()).max())
    print(f"[flashsr-verify] ONNX vs PyTorch: max|Δ|={err:.3e} rel={rel:.3e}")
    if rel > 1e-3:
        print("[flashsr-export] FEHLER: Parität verletzt")
        return 1
    print("[flashsr-export] ERFOLG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
