#!/usr/bin/env python3
"""Exportiert den BigVGAN-v2-Generator von PyTorch nach ONNX.

Der Checkpoint wird als State-Dict geladen und mit der exakt im Plugin
verwendeten BigVGAN-Konfiguration rekonstruiert. Exportiert wird nur der
Generator (Mel-Spektrogramm -> Waveform); die Mel-Extraktion bleibt im
Plugin, damit sie deterministisch und unabhängig von ONNX Runtime bleibt.

Nutzung:
    .venv_aurik/bin/python scripts/export_bigvgan_v2_onnx.py
    .venv_aurik/bin/python scripts/export_bigvgan_v2_onnx.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DEFAULT = ROOT / "models" / "bigvgan" / "bigvgan_v2.pth"
OUTPUT_DEFAULT = ROOT / "models" / "bigvgan" / "bigvgan_v2.onnx"
MEL_BANDS = 128
EXPORT_MEL_FRAMES = 64


def _build_model(checkpoint: Path):
    import torch  # pylint: disable=import-outside-toplevel

    try:
        from bigvgan import AttrDict, BigVGAN  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("bigvgan fehlt: pip install bigvgan --no-deps") from exc

    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state_dict = raw.get("generator") if isinstance(raw, dict) and "generator" in raw else raw
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint enthält kein Generator-State-Dict")

    mel_weight = state_dict.get("conv_pre.weight_v")
    num_mels = int(mel_weight.shape[1]) if mel_weight is not None else MEL_BANDS
    if num_mels != MEL_BANDS:
        raise RuntimeError(f"Checkpoint erwartet {num_mels} Mel-Bänder, Plugin liefert {MEL_BANDS}")

    upsample_weight = state_dict.get("ups.0.0.weight_v")
    upsample_initial_channel = int(upsample_weight.shape[0]) if upsample_weight is not None else 512
    config = AttrDict(
        {
            "resblock": "1",
            "upsample_rates": [8, 4, 2, 2, 2, 2],
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": [16, 8, 4, 4, 4, 4],
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "activation": "snakebeta",
            "snake_logscale": True,
            "use_tanh_at_final": False,
            "use_bias_at_final": False,
            "num_mels": num_mels,
            "sampling_rate": 44100,
            "n_fft": 2048,
            "hop_size": 512,
            "win_size": 2048,
            "fmin": 0,
            "fmax": None,
        }
    )
    model = BigVGAN(config, use_cuda_kernel=False)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"State-Dict passt nicht: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return model.eval()


def export(checkpoint: Path, output: Path, *, force: bool = False, opset: int = 18) -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint}")
    if output.exists() and not force:
        print(f"Existiert bereits: {output} (verwende --force zum Überschreiben)")
        return 0

    model = _build_model(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, MEL_BANDS, EXPORT_MEL_FRAMES), dtype=torch.float32)
    with torch.no_grad():
        reference = model(dummy).detach().cpu().numpy()

    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        input_names=["mel"],
        output_names=["audio"],
        opset_version=opset,
        dynamo=True,
        do_constant_folding=True,
    )

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"mel": dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(reference - actual)))
    scale = max(float(np.max(np.abs(reference))), 1e-8)
    relative = max_abs / scale
    print(f"Export OK: {output} ({output.stat().st_size / 1e6:.1f} MB)")
    print(f"Parität: max_abs={max_abs:.3e}, relative={relative:.3e}")
    if relative >= 1e-3:
        raise RuntimeError("ONNX-vs.-PyTorch-Parität verletzt (relative Abweichung >= 1e-3)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return export(args.checkpoint, args.output, force=args.force, opset=args.opset)


if __name__ == "__main__":
    sys.exit(main())
