#!/usr/bin/env python3
"""MuQ-MuLan Audio-Turm → ONNX (Aurik, ROCm).

Exportiert den Audio-Turm von OpenMuQ/MuQ-MuLan-large (Tencent AILab,
arXiv:2501.01108, Gewichte CC-BY-NC-4.0) als ONNX:

    Waveform (1, 240000) @ 24 kHz  →  Audio-Embedding (1, 768)

Graph: MuQ-Encoder (MelSTFT + Conv-Subsampling + Conformer, eingefroren)
→ letzte Hidden-Layer (B, T, 1024) → Linear(1024→768) → Zeit-Mean-Pool.
(tf_depth=0 → kein Refinement-Transformer; aggregator ungenutzt im MuQ-Pfad.)

Voraussetzungen:
  - models/muq_mulan/            MuQ-Encoder (config.json + model.safetensors)
  - models/muq_mulan/mulan/      MuQ-MuLan-Checkpoint (config.json + pytorch_model.bin)

Verwendung:
  .venv_aurik/bin/python scripts/export_muq_mulan_onnx.py

Ausgabe:
  models/muq_mulan/muq_mulan.onnx  (fp32, statisches Shape 10 s)

Validierung: torch vs. ONNX-Runtime (CPU) mit rel-Toleranz 1e-3.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.onnx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_MUQ_DIR = _PROJECT_ROOT / "models" / "muq_mulan"
_MULAN_DIR = _MUQ_DIR / "mulan"
_OUT_ONNX = _MUQ_DIR / "muq_mulan.onnx"
_SR = 24000
_CLIP_S = 10.0
_N_SAMPLES = int(_SR * _CLIP_S)  # 240000
_OPSET = 18


def _build_audio_tower() -> torch.nn.Module:
    """Audio-Turm exakt wie MuQMuLan (tf_depth=0 → Identity-Transformer)."""
    from plugins._vendor_muq.muq_mulan.models.audio import AudioSpectrogramTransformerPretrained

    tower = AudioSpectrogramTransformerPretrained(
        model_name=str(_MUQ_DIR),  # 'muq' in name → lokaler MuQ-Encoder via from_pretrained(local_dir)
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
    return tower


def _load_audio_tower_weights(tower: torch.nn.Module) -> None:
    """Lädt die mulan.audio.*-Gewichte aus dem MuQ-MuLan-Checkpoint."""
    ckpt = _MULAN_DIR / "pytorch_model.bin"
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint fehlt: {ckpt}")
    print(f"Lade Checkpoint-Keys aus {ckpt.name} (2,65 GB, dauert ~1–2 min) …", flush=True)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    # MuQMuLan-Struktur: mulan.audio.<Turm-Attribut>… (Turm = AudioSpectrogramTransformerPretrained)
    prefix = "mulan.audio."
    audio_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not audio_state:
        print("Keine mulan.audio.*-Keys gefunden — Checkpoint-Struktur unerwartet.")
        raise SystemExit(1)
    print(f"{len(audio_state)} Audio-Turm-Tensoren extrahiert", flush=True)
    missing, unexpected = tower.load_state_dict(audio_state, strict=False)
    if missing:
        print(f"WARNUNG fehlende Keys: {missing[:6]}")
    if unexpected:
        print(f"WARNUNG unerwartete Keys: {unexpected[:6]}")
    del state
    tower.eval()
    for p in tower.parameters():
        p.requires_grad_(False)


def main() -> int:
    t_start = time.perf_counter()
    torch.manual_seed(0)
    tower = _build_audio_tower()
    _load_audio_tower_weights(tower)

    dummy = torch.zeros(1, _N_SAMPLES, dtype=torch.float32)

    # Referenz-Ausgabe (torch)
    with torch.no_grad():
        ref = tower(dummy, return_all_layers=False, return_mean=True)
    print(f"Torch-Referenz: shape={tuple(ref.shape)}", flush=True)

    print(f"ONNX-Export (dynamo, opset {_OPSET}, statisch {_N_SAMPLES} Samples) …", flush=True)

    torch.onnx.export(
        tower,
        (dummy,),
        str(_OUT_ONNX),
        export_params=True,
        opset_version=_OPSET,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["audio_embedding"],
        dynamic_axes=None,
        dynamo=True,
    )

    # ── Validierung: ONNX Runtime (CPU) vs. torch ──────────────────────────
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(_OUT_ONNX), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"waveform": dummy.numpy()})[0]
        rel = float(np.max(np.abs(ort_out - ref.numpy())) / max(float(np.max(np.abs(ref.numpy()))), 1e-9))
        print(f"ORT-Validierung: shape={ort_out.shape} max_rel_diff={rel:.3e}", flush=True)
        if rel > 1e-3:
            print("WARNUNG: rel-Differenz > 1e-3 — Export prüfen.")
            return 2
    except Exception as exc:  # pragma: no cover
        print(f"ORT-Validierung fehlgeschlagen: {exc}")
        return 2

    size_mb = _OUT_ONNX.stat().st_size / 1e6
    print(f"OK: {_OUT_ONNX} ({size_mb:.0f} MB) in {time.perf_counter() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
