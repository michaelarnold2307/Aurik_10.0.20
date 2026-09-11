#!/usr/bin/env python3
"""Extrahiert den MuQ-Eval-A1-Head aus dem großen A1-Checkpoint.

MuQ-Eval (Zhu & Li 2026, arXiv:2603.22677, MIT) veröffentlicht den trainierten
A1-Qualitätskopf (Attention-Pooling + 2-Layer-MLP, ~1,3 M Parameter,
SRCC 0.957 system-level auf MusicEval) nur eingebettet in den 1,34 GB großen
`model_state_dict.pt` (zhudi2825/MuQ-Eval-A1 auf HuggingFace). Dieses Skript
trennt die Head-Gewichte ab und legt sie klein (~5 MB) unter
`models/muq/muq_eval_a1_head.pt` ab — danach aktiviert `plugins/muq_plugin.py`
automatisch den gelernten MOS (`estimate_muq_mos`).

Verwendung (einmalig, nach Download des großen Checkpoints):
    .venv_aurik/bin/python scripts/extract_muq_eval_a1_head.py /pfad/model_state_dict.pt

Key-Mapping (MusicQualityModel, A1-frozen):
    encoder.pooling.*  -> pooling.*   (AttentionPooling: attention.0/.2)
    heads.MI.*        -> head.*      (PredictionHead: mlp.0/.3)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUT = _PROJECT_ROOT / "models" / "muq_mulan" / "muq_eval_a1_head.pt"


def main() -> int:
    if len(sys.argv) < 2:
        print("Verwendung: extract_muq_eval_a1_head.py <model_state_dict.pt>")
        return 2
    src = Path(sys.argv[1])
    if not src.exists():
        print(f"Datei nicht gefunden: {src}")
        return 2

    state = torch.load(str(src), map_location="cpu", weights_only=False)
    head: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.startswith("encoder.pooling."):
            head["pooling." + key[len("encoder.pooling.") :]] = tensor
        elif key.startswith("heads.MI."):
            head["head." + key[len("heads.MI.") :]] = tensor

    if not head:
        print("Keine A1-Head-Keys gefunden — ist dies das model_state_dict.pt von zhudi2825/MuQ-Eval-A1?")
        return 1

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head, str(_OUT))
    n_params = sum(int(t.numel()) for t in head.values())
    print(f"{len(head)} Tensoren, {n_params} Parameter -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
