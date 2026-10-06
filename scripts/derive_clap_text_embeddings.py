#!/usr/bin/env python3
"""§P1-5b (2026-09-09, rev. 2026-10-06): LAION-CLAP Text-Embeddings ableiten.

Der ONNX-Pfad des CLAP-Plugins benötigt ``models/clap/text_embeddings.npy`` in der
Reihenfolge der Aurik-Tags (``INSTRUMENT_TAGS`` + ``GENRE_TAGS`` + ``MATERIAL_TAGS``,
siehe ``plugins/laion_clap_plugin.py``).

**Befund 2026-10-06 (Korrektur der Vorversion):** Die erste Fassung schrieb 527
**AudioSet**-Klassen-Embeddings und nutzte dabei eine falsche Projektionskette
(CLS statt ``pooler_output``, zusätzliches LayerNorm, Verkettung von
``text_projection`` *und* ``text_transform``). Der Plugin-ONNX-Pfad schnitt davon
die ersten 43 Zeilen ab und beschriftete sie als Aurik-Tags — die Scores waren also
sowohl strukturell falsch als auch falsch benannt, und die Text-Encoder von ONNX-
und PyTorch-Pfad waren unkorreliert (Cosinus 0,02).

**Kanonische Pipeline:** ``get_text_embedding()`` des Checkpoints ruft
``encode_text()`` auf, das ist ``pooler_output -> text_projection``
(``Linear -> Aktivierung -> Linear``). ``encode_text`` ist damit die einzige
maßgebliche Text-Funktion; ``text_transform`` erzeugt in ``forward()`` eine
ZWEITE, separate Einbettung (``text_features_mlp``) und gehört nicht in den Pfad.

Das Skript rekonstruiert diese Funktion **vollständig lokal und offline**: Die
Text-Turm-Architektur wird mit den checkpoint-kompatiblen Dimensionen (514
Positions- und 1 Token-Type-Embedding, fairseq-RoBERTa) aufgebaut und mit den
Gewichten aus dem Checkpoint befüllt. Es wird KEIN HuggingFace-Download benötigt.
Fehlt auch nur ein trainiertes Gewicht, bricht das Skript ab (fail-closed) —
sonst entstünde wieder eine still degradierte Einbettung (§V6 copilot-instructions.md).

Deterministisch (CPU, eval, no_grad), gleicher Input ⇒ bit-identische Ausgabe.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "models" / "clap" / "music_audioset_epoch_15_esc_90.14.pt"
OUT = ROOT / "models" / "clap" / "text_embeddings.npy"

# Projektwurzel importierbar machen (Tag-Quelle liegt in plugins/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Checkpoint-kompatible RoBERTa-Dimensionen (fairseq-Variante des CLAP-Checkpoints).
# Mit der HF-Variante (512/2) würden die trainierten Positions-Embeddings verworfen.
_MAX_POS = 514
_TYPE_VOCAB = 1
_VOCAB = 50265
_HIDDEN = 768
_LAYERS = 12
_HEADS = 12
_INTERMEDIATE = 3072
_JOINT = 512


def _aurik_tags() -> list[str]:
    """Aurik-Tag-Reihenfolge des Plugin-ONNX-Pfads (EINE Quelle, §G9 copilot-instructions.md)."""
    from plugins.laion_clap_plugin import GENRE_TAGS, INSTRUMENT_TAGS, MATERIAL_TAGS

    return list(INSTRUMENT_TAGS) + list(GENRE_TAGS) + list(MATERIAL_TAGS)


def _load_text_weights(sd: dict) -> dict:
    """Text-Turm-Gewichte aus dem Checkpoint, ohne int-Buffer wie ``position_ids``."""
    out: dict = {}
    for k, v in sd.items():
        if not k.startswith("module.text_branch."):
            continue
        nk = k[len("module.text_branch.") :]
        if nk == "embeddings.position_ids":  # int-Buffer, kein trainierbares Gewicht
            continue
        out[nk] = v
    return out


def _write_atomic(path: Path, arr) -> None:
    """Schreibt die Matrix atomar und legt vorher ein Backup der Vorversion an."""
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        logger.info("Vorversion gesichert: %s", backup.name)
    tmp = path.with_name(path.name + ".tmp.npy")
    try:
        # np.save hängt ".npy" an einen suffixlosen Pfad → Name endet bereits auf .npy
        np.save(tmp, arr)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="LAION-CLAP Text-Embeddings (Aurik-Tags) ableiten")
    ap.add_argument("--force", action="store_true", help="vorhandenes Artefakt überschreiben")
    ap.add_argument(
        "--verify-pt",
        action="store_true",
        help="Ergebnis zusätzlich gegen den PyTorch-Pfad des Plugins prüfen (lädt 2,2 GB)",
    )
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        logger.info("Artefakt existiert bereits: %s — mit --force neu erzeugen", OUT)
        return 0

    import torch
    import torch.nn as nn
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

    texts = _aurik_tags()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]

    cfg = RobertaConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        intermediate_size=_INTERMEDIATE,
        max_position_embeddings=_MAX_POS,
        type_vocab_size=_TYPE_VOCAB,
    )
    model = RobertaModel(cfg)
    text_sd = _load_text_weights(sd)
    state = model.state_dict()
    kept = {k: v for k, v in text_sd.items() if k in state and state[k].shape == v.shape}
    dropped = [k for k in text_sd if k not in kept]
    if dropped:
        # Fail-closed: ein fehlendes trainiertes Gewicht würde die Einbettung still
        # degradieren (§V6 copilot-instructions.md).
        logger.error("ABBRUCH: %d Text-Turm-Gewichte nicht übernehmbar: %s", len(dropped), dropped[:6])
        return 2
    model.load_state_dict(kept, strict=False)
    model.eval()
    logger.info("RoBERTa-Text-Turm: %d/%d Gewichte übernommen (Dimensionen 514/1)", len(kept), len(text_sd))

    proj0 = nn.Linear(_HIDDEN, _JOINT)
    proj0.load_state_dict(
        {"weight": sd["module.text_projection.0.weight"], "bias": sd["module.text_projection.0.bias"]}
    )
    proj2 = nn.Linear(_JOINT, _JOINT)
    proj2.load_state_dict(
        {"weight": sd["module.text_projection.2.weight"], "bias": sd["module.text_projection.2.bias"]}
    )
    proj0.eval()
    proj2.eval()

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True)

    def encode(batch: list[str]):
        with torch.no_grad():
            # Tokenisierung EXAKT wie der Plugin-Pfad (hook.py::tokenizer),
            # damit Artefakt und PyTorch-Pfad dieselbe Funktion abbilden.
            tok = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
            out = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            # Kanonisch = encode_text(): pooler_output -> text_projection
            return proj2(torch.relu(proj0(out.pooler_output)))

    embs = encode(texts)
    embs = torch.nn.functional.normalize(embs, dim=-1)
    arr = embs.cpu().numpy().astype("float32")

    if arr.shape != (len(texts), _JOINT):
        logger.error("ABBRUCH: unerwartete Form %s, erwartet (%d, %d)", arr.shape, len(texts), _JOINT)
        return 3
    norms = np.linalg.norm(arr, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        logger.error("ABBRUCH: Embeddings sind nicht L2-normalisiert (min=%.6f, max=%.6f)", norms.min(), norms.max())
        return 4

    _write_atomic(OUT, arr)
    logger.info("geschrieben: %s shape=%s (%d Aurik-Tags)", OUT, arr.shape, len(texts))

    if args.verify_pt:
        from plugins.laion_clap_plugin import get_laion_clap

        clap = get_laion_clap()
        if not clap._ensure_pt_loaded():
            logger.error("Paritätsprüfung nicht möglich: PyTorch-Checkpoint nicht ladbar")
            return 5
        pt = clap._clap_model.get_text_embedding(texts, use_tensor=False)
        pt = np.asarray(pt, dtype=np.float64)
        ref = arr.astype(np.float64)
        cos = np.sum(
            (pt / (np.linalg.norm(pt, axis=1, keepdims=True) + 1e-12))
            * (ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)),
            axis=1,
        )
        logger.info("Parität ONNX-Artefakt vs. PyTorch-Pfad: cos mean=%.6f min=%.6f", cos.mean(), cos.min())
        if cos.min() < 0.9999:
            logger.error("Parität verfehlt (§III.9 copilot-instructions.md): min-cos=%.6f", cos.min())
            return 6

    return 0

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
