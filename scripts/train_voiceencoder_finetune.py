"""F9 (D-K4-5, SOTA-M11): Musik-Vokal-Finetune des Resemblyzer-VoiceEncoders.

Ziel (Roadmap F9): Der Vokal-Identitaets-Witness (Phase 65) soll auch auf
VERARBEITETEN Vocals (Hall, Saettigung, Bandlimit) scharf trennen — heute wird
er dort stumpf (D-K4-5-Befund). Der deployte Encoder
(`models/resemblyzer/resemblyzer_voice_encoder.onnx`, exportiert aus
`pretrained.pt`) ist sprach-trainiert; dieser Trainer feintunt die
Torch-Gewichte (`models/resemblyzer/resemblyzer/pretrained.pt`) auf
MUSDB18-HQ-Vocals: Track = Saenger-Identitaet (keine externen Labels),
Kontrastiv-Aufgabe (NT-Xent) ueber d-Vektoren mit Augmentations-Konsistenz.

KETTE-4-VERTRAG (scripts/gpu_kette4_modellkoepfe.sh R3): aufrufbar als
`--musdb-root data/musdb18hq --epochs 30 --device cuda --out-dir models/resemblyzer/finetune`.

Determinismus (§G5 (copilot-instructions.md)): Seed 42, sortierte Track-Liste,
keine Zeit-/Zufalls-Quellen ausser dem gepinnten numpy-RNG; cudnn.benchmark=False.

KEINE Flag-Flips, KEIN Deploy: schreibt ausschliesslich Checkpoint + Manifest
nach `--out-dir` (save_guarded, backend/core/training_artifacts.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample_poly, sosfilt

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_RESEMBLYZER_PKG = _ROOT / "models" / "resemblyzer"
_SR_TARGET = 16_000  # Witness-/VoiceEncoder-Arbeitsrate (train/serve-konsistent)
_SEG_S = 3.0
_AUG_KINDS = ("clean", "hall", "saettigung", "bandlimit")
_MANIFEST_SCHEMA = "aurik.voiceencoder_finetune/1"


def _mono_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    """Stereo->Mono (kanal-sicher) + Resample auf 16 kHz, deterministisch; int16 -> [-1, 1]."""
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    if x.ndim == 1:
        mono = x
    elif x.shape[0] == 2:  # channels-first (C, N) — Pipeline-Layout AGENTS.md §3
        mono = x.mean(axis=0)
    elif x.shape[1] == 2:  # (N, C) von wavfile
        mono = x.mean(axis=1)
    else:
        mono = x.mean(axis=-1)
    if sr_in != _SR_TARGET:
        g = math.gcd(_SR_TARGET, sr_in)
        mono = resample_poly(mono.astype(np.float64), _SR_TARGET // g, sr_in // g)
    return mono.astype(np.float32)


def _augment(x: np.ndarray, kind: str, rng: np.random.Generator, sr: int) -> np.ndarray:
    """Deterministische Verarbeitungsketten der D-K4-5-Befundklasse (§V7 (copilot-instructions.md): eine Kette je Klasse, keine phasen-individuellen Schwellen)."""
    if kind == "clean":
        return x
    if kind == "hall":  # gefalteter Hall: gefaerbtes IR (exponentiell abklingendes Rauschen)
        t60 = rng.uniform(0.4, 1.2)
        n = int(t60 * sr)
        ir = rng.standard_normal(n).astype(np.float64) * np.exp(-3.0 * np.arange(n) / (t60 * sr))
        ir[:48] *= np.linspace(0.0, 1.0, 48)  # direkter Schall
        y = np.convolve(x.astype(np.float64), ir, mode="full")[: x.size]
        y /= max(np.max(np.abs(y)), 1e-6)
        wet = rng.uniform(0.15, 0.45)
        return (x * (1.0 - wet) + y * wet).astype(np.float32)
    if kind == "saettigung":
        drive = rng.uniform(1.5, 4.0)
        y = np.tanh(x.astype(np.float64) * drive) / np.tanh(drive)
        return y.astype(np.float32)
    if kind == "bandlimit":
        cutoff = float(rng.uniform(2500.0, 6500.0))
        sos = butter(4, cutoff / (0.5 * sr), btype="low", output="sos")
        return sosfilt(sos, x).astype(np.float32)
    raise ValueError(f"Unbekannte Augmentations-Klasse: {kind}")


def _iter_tracks(musdb_root: Path) -> list[Path]:
    train = Path(musdb_root) / "train"
    if not train.is_dir():
        raise FileNotFoundError(f"MUSDB18-HQ train/ fehlt unter {train} (--musdb-root)")
    tracks = sorted(p.parent for p in train.glob("*/vocals.wav"))
    if not tracks:
        raise FileNotFoundError(f"Keine vocals.wav unter {train} gefunden")
    return tracks


def _build_batches(tracks: list[Path], rng: np.random.Generator, batch: int) -> list[dict]:
    """Deterministische Positiv/Negativ-Batches: je Track 2 Sichten (Anker + augmentierter Positiv-Partner) => NT-Xent hat echte Positive; Negative = andere Tracks im Batch."""
    seg = int(_SEG_S * _SR_TARGET)
    by_track: dict[str, list[np.ndarray]] = {}
    for t in tracks:
        out = wavfile.read(str(t / "vocals.wav"))
        if isinstance(out, tuple):  # Bug 12, §V36 (VERBOTEN.md): index-basierte Entpackung mit Prüfung
            sr_in, wav = out
        else:
            sr_in, wav = _SR_TARGET, out
        x = _mono_16k(wav, sr_in)
        n = x.size // seg
        if n < 2:
            continue
        starts = np.linspace(0, x.size - seg, min(n, 8), dtype=int)
        by_track[str(t)] = [x[s : s + seg] for s in starts]
    ids = sorted(by_track)
    rng.shuffle(ids)
    batches = []
    for i in range(0, max(1, len(ids) - batch + 1), batch):
        group = ids[i : i + batch]
        if len(group) < 2:
            continue
        views: list[np.ndarray] = []
        labels: list[int] = []
        for idx, tid in enumerate(group):
            segs = by_track[tid]
            anchor = segs[int(rng.integers(len(segs)))]
            partner = segs[int(rng.integers(len(segs)))]
            kind = _AUG_KINDS[int(rng.integers(len(_AUG_KINDS)))]
            views.append(anchor)
            views.append(_augment(partner, kind, rng, _SR_TARGET))
            labels.extend([idx, idx])
        batches.append({"views": views, "labels": labels})
    return batches


def _nt_xent_torch(z, labels: np.ndarray, tau: float):
    """NT-Xent in Torch (Gradient zu z bleibt erhalten — kein Detach, §V6 (copilot-instructions.md))."""
    import torch

    sim = z @ z.T / tau
    mask_self = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask_self, float("-inf"))
    pos = torch.as_tensor(
        (labels[:, None] == labels[None, :]) & ~np.eye(len(labels), dtype=bool),
        dtype=torch.bool,
        device=z.device,
    )
    num = (torch.exp(sim) * pos.float()).sum(dim=1)
    den = torch.exp(sim).sum(dim=1)
    loss = -torch.mean(torch.log(num / den.clamp(min=1e-12)))
    return loss


def _verification_accuracy(embeddings: dict[str, np.ndarray], rng: np.random.Generator, pairs: int = 64) -> float:
    ids = sorted(embeddings)
    same = 0
    total = 0
    for _ in range(pairs):
        if rng.random() < 0.5:
            t = ids[int(rng.integers(len(ids)))]
            e = embeddings[t]
            i, j = int(rng.integers(len(e))), int(rng.integers(len(e)))
            while j == i:
                j = int(rng.integers(len(e)))
            sim = float(e[i] @ e[j])
            label = 1.0
        else:
            t1, t2 = ids[int(rng.integers(len(ids)))], ids[int(rng.integers(len(ids)))]
            while t2 == t1:
                t2 = ids[int(rng.integers(len(ids)))]
            sim = float(embeddings[t1][0] @ embeddings[t2][0])
            label = 0.0
        same += 1.0 if (sim > 0.0) == (label > 0.5) else 0.0
        total += 1
    return same / max(total, 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="F9: VoiceEncoder-Musik-Vokal-Finetune (D-K4-5)")
    p.add_argument("--musdb-root", required=True, type=Path)
    p.add_argument("--epochs", required=True, type=int)
    p.add_argument("--device", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--smoke", action="store_true", help="1 Epoch, kleine Batches")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch < 2:
        logger.error("epochs >= 1 und batch >= 2 erforderlich")
        return 2
    try:
        import torch  # lazy: CLI-/Daten-Tests laufen ohne Torch (§V6 (copilot-instructions.md): kein Silent-Failure)
    except ImportError as exc:
        logger.error("torch fehlt — Training nur im venv_aurik (.venv_aurik): %s", exc)
        return 3
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np_rng = np.random.default_rng(args.seed)

    sys.path.insert(0, str(_RESEMBLYZER_PKG))
    try:
        from resemblyzer import VoiceEncoder  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.error("resemblyzer-Paket nicht ladbar (%s) — %s", _RESEMBLYZER_PKG, exc)
        return 3

    tracks = _iter_tracks(args.musdb_root)
    if args.smoke:
        tracks = tracks[:4]
    logger.info("Tracks: %d (smoke=%s)", len(tracks), args.smoke)

    try:
        encoder = VoiceEncoder(args.device)
    except Exception as exc:  # §V6 (copilot-instructions.md): Fallback mit Begründung statt Silent-Failure
        logger.warning(
            "VoiceEncoder auf %s nicht ladbar (%s) — faelle auf CPU zurueck (G5: Determinismus bleibt, Seed fix)",
            args.device,
            exc,
        )
        encoder = VoiceEncoder("cpu")
    model_mod = encoder.model if hasattr(encoder, "model") else encoder

    batches = _build_batches(tracks, np_rng, args.batch)
    if not batches:
        logger.error("Keine Batches gebaut (MUSDB-Vocals zu kurz?)")
        return 2

    # Validierungs-Tracks (deterministisch): letzte 20 % — Never-worsen-Basis vor dem Training.
    split = max(1, int(len(tracks) * 0.8))
    val_tracks = sorted({str(t) for t in tracks[split:]})
    val_ids = [str(t) for t in tracks[split:]]
    val_ids = [v for v in val_ids if v in val_tracks]

    def _embed_val() -> dict[str, np.ndarray]:
        emb: dict[str, np.ndarray] = {}
        seg = int(_SEG_S * _SR_TARGET)
        for v in val_tracks:
            out = wavfile.read(str(Path(v) / "vocals.wav"))
            if isinstance(out, tuple):  # Bug 12, §V36 (VERBOTEN.md): index-basierte Entpackung mit Prüfung
                sr_in, wav = out
            else:
                sr_in, wav = _SR_TARGET, out
            x = _mono_16k(wav, sr_in)
            starts = np.linspace(0, max(1, x.size - seg), 2, dtype=int)
            emb[v] = np.stack([np.asarray(encoder.embed_utterance(x[s : s + seg]), dtype=np.float64) for s in starts])
        return emb

    val_rng = np.random.default_rng(args.seed + 1)
    base_acc = _verification_accuracy(_embed_val(), val_rng)
    logger.info("Basis-Validierungs-Genauigkeit: %.3f", base_acc)

    opt = torch.optim.Adam(model_mod.parameters(), lr=args.lr)
    epochs = 1 if args.smoke else args.epochs
    best_acc = base_acc
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    def _checkpoint(acc: float, ep: int, loss: float) -> None:
        from backend.core.training_artifacts import save_guarded

        save_guarded(out_dir / "voiceencoder_vocal_ft.pt", lambda p: torch.save(model_mod.state_dict(), p))
        manifest = {
            "schema": _MANIFEST_SCHEMA,
            "seed": args.seed,
            "epochs": ep,
            "lr": args.lr,
            "batch": args.batch,
            "tracks": len(tracks),
            "final_loss": loss,
            "val_acc_base": base_acc,
            "val_acc": acc,
            "augmentations": list(_AUG_KINDS),
            "basis": str(_RESEMBLYZER_PKG / "resemblyzer" / "pretrained.pt"),
            "ausgabe": str(out_dir / "voiceencoder_vocal_ft.pt"),
            "never_worsen": (
                "Checkpoint nur geschrieben, wenn Validierungs-Genauigkeit >= Basis - 0.01 "
                "(sonst bleibt der letzte gute Stand, §V6 (copilot-instructions.md))"
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    for ep in range(1, epochs + 1):
        total = 0.0
        for b in batches:
            zs = [encoder.embed_utterance(a) for a in b["views"]]
            z = torch.stack([torch.as_tensor(e, dtype=torch.float32) for e in zs])
            z = torch.nn.functional.normalize(z, dim=1)
            labels = np.asarray(b["labels"], dtype=np.int64)
            loss = _nt_xent_torch(z, labels, 0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        mean_loss = total / max(len(batches), 1)
        logger.info("Epoch %d/%d loss=%.6f", ep, epochs, mean_loss)
        acc = _verification_accuracy(_embed_val(), val_rng)
        logger.info("Validierungs-Genauigkeit: %.3f (Basis %.3f)", acc, base_acc)
        if acc >= best_acc - 0.01:  # Never-worsen (§V6 (copilot-instructions.md)): nur besserer Stand wird persistiert
            best_acc = acc
            _checkpoint(acc, ep, mean_loss)
    logger.info("Fertig: Checkpoint + Manifest unter %s (val_acc=%.3f)", out_dir, best_acc)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
