"""Export Apollo (Band-Sequence Mamba) als TorchScript.

Apollo-Quellcode: models/apollo/look2hear/
Checkpoint:       models/apollo/pytorch_model.bin  (state_dict-Format)
Ausgabe:          models/apollo/apollo_model.pt    (TorchScript, ~64 MB)

Warum TorchScript statt ONNX:
    Das ONNX-Modell nutzt org.pytorch.aten:ATen Custom-Ops (stft, istft, complex,
    real, imag), die onnxruntime ohne spezielle C++-Extensions nicht ausführen kann.
    TorchScript umgeht das vollständig und läuft auf jedem CPU.

Usage:
    .venv_aurik/bin/python scripts/export_apollo_torchscript.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "models" / "apollo"))  # look2hear verfügbar machen

CHECKPOINT = ROOT / "models" / "apollo" / "pytorch_model.bin"
OUTPUT_PT = ROOT / "models" / "apollo" / "apollo_model.pt"


def _load_model():
    import torch
    from look2hear.models.apollo import Apollo

    ck = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    args = ck["model_args"]  # {'sr': 44100, 'win': 20, 'feature_dim': 256, 'layer': 6}

    print(f"Model args: {args}")
    model = Apollo(**args)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    torch.set_num_threads(os.cpu_count() or 4)
    return model, args


def _export(model, args) -> None:
    import torch

    sr = args["sr"]  # 44100
    # Dummy-Input: Mono, 3 Sekunden bei 44100 Hz
    dummy = torch.zeros(1, 1, sr * 3)

    print("Starte TorchScript-Export ...", flush=True)

    # torch.jit.trace — Apollo hat keine datenabhängigen Kontrollfluesse im forward()
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)

    # Validierung
    with torch.no_grad():
        out = traced(dummy)
    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} != {dummy.shape}"
    print(f"✓ Forward-Test OK — Shape: {list(out.shape)}, Range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    traced.save(str(OUTPUT_PT))
    size_mb = OUTPUT_PT.stat().st_size / 1024 / 1024
    print(f"✓ TorchScript gespeichert — {size_mb:.1f} MB → {OUTPUT_PT}")

    # Reload-Validierung
    reloaded = torch.jit.load(str(OUTPUT_PT), map_location="cpu")
    with torch.no_grad():
        out2 = reloaded(dummy)
    max_diff = (out - out2).abs().max().item()
    print(f"✓ Reload-Validierung OK — Max-Diff: {max_diff:.2e}")

    # SHA256
    sha = hashlib.sha256(OUTPUT_PT.read_bytes()).hexdigest()
    print(f"  SHA256: {sha}")
    print(f"  Size:   {OUTPUT_PT.stat().st_size}")


if __name__ == "__main__":
    print("=== Apollo TorchScript Export ===")
    print(f"Checkpoint: {CHECKPOINT}")
    if not CHECKPOINT.exists():
        print(f"FEHLER: Checkpoint nicht gefunden: {CHECKPOINT}", file=sys.stderr)
        sys.exit(1)

    model, args = _load_model()
    _export(model, args)
    print(f"\n✓ Exportiert: {OUTPUT_PT}")
    print("  Nächster Schritt: Aurik starten — Apollo wird automatisch geladen.")
