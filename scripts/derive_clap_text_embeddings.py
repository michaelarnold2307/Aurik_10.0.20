#!/usr/bin/env python3
"""§P1-5b (2026-09-09): LAION-CLAP Text-Embeddings einmalig ableiten.

Der ONNX-Pfad des CLAP-Plugins benötigt ``models/clap/text_embeddings.npy``
(527 × 512) — fehlt die Datei, springt das Plugin auf den 2,2-GB-PyTorch-Pfad
zurück. Dieses Skript rekonstruiert die Embeddings aus dem Text-Tower des
PyTorch-Checkpoints (roberta, 208 Keys unter ``module.text_branch.*``) und
speichert sie als float32-npy. Deterministisch (CPU, eval, no_grad).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "models" / "clap" / "music_audioset_epoch_15_esc_90.14.pt"
LABELS = ROOT / "models" / "clap" / "class_labels" / "audioset_class_labels_indices.json"
OUT = ROOT / "models" / "clap" / "text_embeddings.npy"


def main() -> int:
    if OUT.exists():
        print(f"existiert bereits: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        return 0
    from transformers import AutoModel, AutoTokenizer  # pylint: disable=import-outside-toplevel

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd: dict[str, torch.Tensor] = ck["state_dict"]
    # module.text_branch.* → roberta-keys
    text_sd: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("module.text_branch."):
            nk = k[len("module.text_branch.") :]
            if nk in ("embeddings.position_ids",):  # int-buffer, kein Gewicht
                continue
            text_sd[nk] = v
    print(f"text_branch-Gewichte: {len(text_sd)} Keys")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base")
    missing, unexpected = model.load_state_dict(text_sd, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    labels = json.loads(LABELS.read_text())
    # JSON-Format: {label_name: index} — die 527 Label-Strings sind die Keys,
    # sortiert nach ihrem Index (entspricht der CLAP-Trainingsreihenfolge).
    texts = [k for k, _v in sorted(labels.items(), key=lambda kv: int(kv[1]))]
    assert len(texts) == 527, len(texts)

    # CLAP-Text-Pipeline: roberta-CLS → text_projection (Linear+LN+Linear) →
    # text_transform (Linear+LN+LN+Linear) → L2-Norm.
    import torch.nn as nn  # pylint: disable=import-outside-toplevel

    _proj0 = nn.Linear(768, 512)
    _proj0.load_state_dict(
        {"weight": sd["module.text_projection.0.weight"], "bias": sd["module.text_projection.0.bias"]}
    )
    _proj2 = nn.Linear(512, 512)
    _proj2.load_state_dict(
        {"weight": sd["module.text_projection.2.weight"], "bias": sd["module.text_projection.2.bias"]}
    )
    _tf0 = nn.Linear(512, 512)
    _tf0.load_state_dict(
        {
            "weight": sd["module.text_transform.sequential.0.weight"],
            "bias": sd["module.text_transform.sequential.0.bias"],
        }
    )
    _tf3 = nn.Linear(512, 512)
    _tf3.load_state_dict(
        {
            "weight": sd["module.text_transform.sequential.3.weight"],
            "bias": sd["module.text_transform.sequential.3.bias"],
        }
    )
    _ln_p = nn.LayerNorm(512)
    _ln_t1 = nn.LayerNorm(512)
    _ln_t2 = nn.LayerNorm(512)
    _proj0.eval()
    _proj2.eval()
    _tf0.eval()
    _tf3.eval()
    _ln_p.eval()
    _ln_t1.eval()
    _ln_t2.eval()

    embs: list[np.ndarray] = []
    with torch.no_grad():
        for t in texts:
            tok = tokenizer(t, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            out = model(**tok)
            x = out.last_hidden_state[:, 0, :]  # CLS-Token (CLAP-Konvention)
            x = _proj0(x)
            x = _ln_p(x)
            x = _proj2(x)
            x = _tf0(x)
            x = _ln_t1(x)
            x = _ln_t2(x)
            x = _tf3(x)
            x = torch.nn.functional.normalize(x, dim=-1)
            embs.append(x[0].cpu().numpy().astype(np.float32))
    arr = np.stack(embs)  # (527, 768)
    np.save(OUT, arr)
    print(f"geschrieben: {OUT} shape={arr.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
