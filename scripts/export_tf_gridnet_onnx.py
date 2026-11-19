"""scripts/export_tf_gridnet_onnx.py — ONNX-Export eines TF-GridNet-Checkpoints (M9-V2, 2026-11-14).

Der M9-V2-Neulauf läuft mit ``--skip-onnx`` — der Export gehört NICHT ins
Training, sondern erst nach dem bestandenen Gate (die deployten
Produktions-Artefakte bleiben unangetastet). Dieses Skript exportiert ein
``tf_gridnet_best.pt`` aus einem Kandidaten-Ordner nach ``<ziel>.onnx`` —
deterministisch, CPU-only, ohne Flag-Flip (§G5 (copilot-instructions.md)).

Verwendung nach dem V2-Lauf:
``.venv_aurik/bin/python -B scripts/export_tf_gridnet_onnx.py
--checkpoint models/tf_gridnet_v2/tf_gridnet_best.pt
--out models/tf_gridnet_v2/tf_gridnet_quality.onnx``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def exportiere(checkpoint: Path, out: Path) -> Path:
    import torch

    from backend.core.dsp.tf_gridnet_denoiser import build_tf_gridnet_model
    from scripts.train_tf_gridnet_denoiser import N_FREQ, export_onnx, set_determinism

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint fehlt: {checkpoint}")
    set_determinism(42)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model = build_tf_gridnet_model(n_freq=N_FREQ)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, out, device=torch.device("cpu"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="tf_gridnet_best.pt des Kandidaten-Ordners")
    ap.add_argument(
        "--out", type=Path, required=True, help="Ziel-ONNX (z.B. models/tf_gridnet_v2/tf_gridnet_quality.onnx)"
    )
    args = ap.parse_args(argv)
    try:
        ziel = exportiere(args.checkpoint, args.out)
    except (FileNotFoundError, KeyError) as exc:
        print(f"ABBRUCH: {exc}")
        return 2
    print(f"ONNX exportiert: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
