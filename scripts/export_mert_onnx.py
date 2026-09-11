#!/usr/bin/env python3
"""§P1-4c (2026-09-09): MERT-v1-330M → ONNX-Export + Parität.

Ziel: models/mert/mert.onnx — der MERT-Plugin-ONNX-Pfad erwartet genau diese
Datei (fehlt → 1.3-GB-HF/transformers-Fallback lädt in JEDEM Run auf CPU).
Export: MERTModel (HubertModel-Subklasse, lokale custom modeling_MERT.py)
input_values (24 kHz) → last_hidden_state; dynamische Zeitachse, opset 17,
Legacy-Exporter (dynamo=False).

Usage:
    .venv_aurik/bin/python scripts/export_mert_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "mert-v1-330m"))

MODEL_DIR = ROOT / "models" / "mert-v1-330m"
OUT = ROOT / "models" / "mert" / "mert_330m.onnx"  # bestehendes mert.onnx = 95M-Variante (768) — nicht überschreiben


def main() -> int:
    import numpy as np  # pylint: disable=import-outside-toplevel
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    if OUT.exists():
        print(f"existiert bereits: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        return 0

    from modeling_MERT import MERTModel  # pylint: disable=import-outside-toplevel

    model = MERTModel.from_pretrained(str(MODEL_DIR))
    model.eval()
    print(f"Modell geladen: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M Parameter")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 24000), dtype=torch.float32)  # 1 s @ 24 kHz
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(OUT),
            input_names=["input_values"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_values": {0: "batch", 1: "time"}, "last_hidden_state": {0: "batch", 1: "time"}},
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    print(f"Export OK: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    # ── Parität: torch vs. ONNX auf identischem Eingang ─────────────────
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((1, 48000)) * 0.05).astype(np.float32)  # 2 s
    with torch.no_grad():
        y_torch = model(torch.from_numpy(x)).last_hidden_state.numpy()
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"input_values": x})[0]
    diff = float(np.abs(y_torch - y_onnx).max())
    rel = float(np.abs(y_torch - y_onnx).mean() / (np.abs(y_torch).mean() + 1e-9))
    print(f"Parität: max_abs={diff:.3e} rel_mean={rel:.3e} (Toleranz 1e-3)")
    return 0 if diff < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
