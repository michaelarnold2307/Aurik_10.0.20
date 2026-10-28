"""SOTA-M8 (D-K4-1): Training des neuronalen A-SPADE-Klassen-Declipper (CHALLENGER zu Phase 07).

Material: MUSDB18-HQ-Vocals (MUSTER: scripts/generate_synthetic_degraded_vocals.py,
``data/musdb18hq/<split>/<track>/vocals.wav``) plus deterministische
Clipping-Generierung pro Segment (§G5 (copilot-instructions.md)):
gleicher Seed ⇒ gleiche (clean, geclippt, theta)-Paare, auch über Resume hinweg.

Vertrag:
    - Seeds pro Session (torch/np/random); die Batch-Ziehung je Schritt ist eine
      reine Funktion von (seed, epoch, step) — Resume bleibt reproduzierbar
      (§G5 (copilot-instructions.md), §G5 (GEBOTE.md)).
    - Checkpoints atomar + rotierendes Backup
      (backend.core.training_artifacts.save_guarded, §v10.16-Artefaktschutz).
    - ONNX-Export (opset 17, dynamische Zeitachse, dynamo=False wie
      scripts/train_bw_v5.py) + Torch↔ONNX-CPU-Paritäts-Gate rel ≤ 1e-3 auf
      strukturierten Feeds (§III.9 (copilot-instructions.md)), fail-closed.
    - KEIN automatischer Flag-Flip: ``use_aspade_declipper`` bleibt False bis
      ΔSDR-Gate auf geclipptem Real-Material + blindes A/B +
      Hörordnungs-Sign-off (§v10.802 (copilot-instructions.md)).

Beispiel (Smoke):
    .venv_aurik/bin/python scripts/train_aspade_declipper.py \
        --musdb-root data/musdb18hq --max-tracks 2 --steps-per-epoch 2 --epochs 1

Beispiel (Training + Resume):
    .venv_aurik/bin/python scripts/train_aspade_declipper.py \
        --musdb-root data/musdb18hq --epochs 30
    .venv_aurik/bin/python scripts/train_aspade_declipper.py \
        --musdb-root data/musdb18hq --epochs 60 --resume models/aspade/checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.dsp.aspade_declipper import (  # sys.path-Siehe train_bw_v5.py
    DEFAULT_DILATIONS,
    AspadeClassNet,
    build_model,
)
from backend.core.training_artifacts import save_guarded

logger = logging.getLogger(__name__)

SR = 48_000
SEGMENT_SAMPLES = 48_000  # 1 s Trainingsssegmente @ 48 kHz
DEFAULT_SEED = 42  # Determinismus §G5 (copilot-instructions.md)
THETA_MIN = 0.15  # Clipping-Schwelle als Anteil der Ziel-Peak-Amplitude
THETA_MAX = 0.95
PARITY_REL_MAX = 1e-3  # §III.9 (copilot-instructions.md)


def _seed_for(seed: int, *parts: object) -> int:
    """Deterministischer Teil-Seed (§G5 (GEBOTE.md)) — MUSTER generate_synthetic_degraded_vocals._seed_for."""
    raw = "|".join([str(seed), *(str(p) for p in parts)])
    h = int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:8], 16)
    return int((seed + h) % (2**31 - 1))


def collect_vocals(musdb_root: Path, splits: tuple[str, ...] = ("train",)) -> list[Path]:
    """MUSDB18-HQ-Vocals je Split: <split>/<track>/vocals.wav (MUSTER generate_synthetic_degraded_vocals)."""
    found: list[Path] = []
    for split in splits:
        found.extend(sorted((musdb_root / split).glob("*/vocals.wav")))
    return found


class _VocalPool:
    """Segment-Zugriff auf MUSDB18-HQ-Vocals.

    Reiner Datencache (max. 2 Tracks) — kein Song-State (§V8/§G1 (copilot-instructions.md)).
    Segmentweise Lese-Zugriffe via soundfile +
    polyphase Resampling (44,1 kHz → 48 kHz) sind deterministisch.
    """

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths
        self._index: list[tuple[Path, int, int]] = []  # (Pfad, Datei-Rate, n_seg)
        for p in paths:
            info = sf.info(str(p))
            n48 = int(info.frames * SR // int(info.samplerate))
            n_seg = n48 // SEGMENT_SAMPLES
            if n_seg > 0:
                self._index.append((p, int(info.samplerate), n_seg))
        if not self._index:
            raise FileNotFoundError(
                f"Keine MUSDB18-HQ-Vocals mit ≥{SEGMENT_SAMPLES} Samples unter {paths!r} — Abbruch (§V7 (copilot-instructions.md))"
            )
        self._cache: dict[int, np.ndarray] = {}

    def n_segments(self, track_idx: int) -> int:
        return self._index[track_idx][2]

    @property
    def n_tracks(self) -> int:
        return len(self._index)

    def track_key(self, track_idx: int) -> str:
        return self._index[track_idx][0].parent.name

    def segment(self, track_idx: int, seg_idx: int) -> np.ndarray:
        path, sr_file, n_seg = self._index[track_idx]
        seg_idx = int(np.clip(seg_idx, 0, max(n_seg - 1, 0)))
        y = self._track(track_idx, sr_file)
        s0 = seg_idx * SEGMENT_SAMPLES
        if s0 + SEGMENT_SAMPLES <= len(y):
            return y[s0 : s0 + SEGMENT_SAMPLES].copy()
        return np.pad(y[s0:], (0, SEGMENT_SAMPLES - len(y[s0:]))).astype(np.float32)

    def _track(self, track_idx: int, sr_file: int) -> np.ndarray:
        cached = self._cache.get(track_idx)
        if cached is not None:
            return cached
        path = self._index[track_idx][0]
        with sf.SoundFile(str(path)) as f:
            block = f.read(dtype="float32", always_2d=True)
        y = block.mean(axis=1).astype(np.float32)  # Mono-Mix wie train_cantus._load_mono
        if sr_file != SR:
            g = gcd(SR, sr_file)
            y = resample_poly(y.astype(np.float64), SR // g, sr_file // g).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)  # §0a (copilot-instructions.md)
        if len(self._cache) >= 2:
            self._cache.clear()  # begrenzter Datencache
        self._cache[track_idx] = y
        return y


def make_clipped_pair(
    clean: np.ndarray, seed: int, track_key: str, seg_idx: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministisches (Ziel, geclippter Eingang, theta)-Tripel (§G5 (copilot-instructions.md))."""
    rng = np.random.default_rng(_seed_for(seed, track_key, seg_idx))
    peak = float(np.max(np.abs(clean))) if clean.size else 0.0
    if peak < 1e-6:
        quiet = np.zeros_like(clean, dtype=np.float32)
        return quiet, quiet.copy(), 1.0
    target = (0.95 * clean / peak).astype(np.float32)
    theta = float(rng.uniform(THETA_MIN, THETA_MAX))
    clipped = np.clip(target, -theta, theta).astype(np.float32)
    return target, clipped, theta


def _step_rng(seed: int, epoch: int, step: int) -> np.random.Generator:
    """Batch-RNG als reine Funktion von (seed, epoch, step) — Resume-reproduzierbar (§G5, copilot-instructions.md)."""
    return np.random.default_rng(_seed_for(seed, f"e{epoch}", f"s{step}"))


def _mr_stft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-Resolution-Magnituden-Loss (2 FFT-Größen, MUSTER train_bw_v5.py)."""
    loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    for n_fft in (512, 1024):
        win = torch.hann_window(n_fft, device=pred.device)
        p = torch.stft(pred.squeeze(1), n_fft=n_fft, hop_length=n_fft // 4, window=win, return_complex=True)
        t = torch.stft(target.squeeze(1), n_fft=n_fft, hop_length=n_fft // 4, window=win, return_complex=True)
        loss = loss + F.l1_loss(p.abs(), t.abs())
    return loss / 2.0


def _sample_batch(
    pool: _VocalPool, rng: np.random.Generator, batch_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets, clipeds, thetas = [], [], []
    for _ in range(batch_size):
        ti = int(rng.integers(pool.n_tracks))
        si = int(rng.integers(pool.n_segments(ti)))
        clean = pool.segment(ti, si)
        target, clipped, theta = make_clipped_pair(clean, seed, pool.track_key(ti), si)
        targets.append(target)
        clipeds.append(clipped)
        thetas.append(theta)
    return (
        np.stack(clipeds)[None, :, :].transpose(1, 0, 2),  # (B, 1, T) unten via reshape
        np.stack(targets),
        np.asarray(thetas, dtype=np.float32),
    )


def train(
    *,
    musdb_root: Path,
    splits: tuple[str, ...],
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    steps_per_epoch: int,
    stages: int,
    channels: int,
    kernel_size: int,
    seed: int,
    device_str: str,
    resume: Path | None,
    max_tracks: int,
    skip_export: bool,
) -> Path:
    # §G5 (copilot-instructions.md): Seeds pro Session — reproduzierbare Läufe.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA angefordert, aber nicht verfügbar — Rückfall auf CPU (§V6 (copilot-instructions.md))")
        device = torch.device("cpu")
    print(f"Device: {device} (Seed: {seed})")

    paths = collect_vocals(musdb_root, splits)
    if max_tracks > 0:
        paths = paths[:max_tracks]
    if not paths:
        raise FileNotFoundError(
            f"Keine vocals.wav unter {musdb_root}/{splits} — Abbruch (§V7 (copilot-instructions.md))"
        )
    pool = _VocalPool(paths)
    print(f"MUSDB18-HQ-Vocals: {pool.n_tracks} Tracks, Segmente je Track ≥ {pool.n_segments(0)}")

    model = build_model(
        stages=stages, channels=channels, kernel_size=kernel_size, dilations=DEFAULT_DILATIONS, seed=seed
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AspadeClassNet: {n_params:,} params | stages={stages} channels={channels} kernel={kernel_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1))
    start_epoch = 0
    best_loss = float("inf")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    ckpt_best = out_dir / "aspade_class_v1.pt"

    if resume is not None:
        payload = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_loss = float(payload.get("best_loss", best_loss))
        print(f"Resume: {resume} — weiter bei Epoche {start_epoch} (best_loss={best_loss:.4f})")

    aspade_config = {
        "stages": int(stages),
        "channels": int(channels),
        "kernel_size": int(kernel_size),
        # §G9 (copilot-instructions.md): eine Quelle — DEFAULT_DILATIONS aus dem Modul.
        "dilations": tuple(int(d) for d in DEFAULT_DILATIONS),
    }

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for step in range(steps_per_epoch):
            rng = _step_rng(seed, epoch, step)
            inp_np, tgt_np, theta_np = _sample_batch(pool, rng, batch_size, seed)
            inp = torch.from_numpy(inp_np.reshape(batch_size, 1, -1)).to(device)
            tgt = torch.from_numpy(tgt_np.reshape(batch_size, 1, -1)).to(device)
            theta_t = torch.from_numpy(theta_np).to(device).view(-1, 1, 1)

            optimizer.zero_grad()
            pred = model(inp, theta_t)
            clipped_mask = (inp.abs() >= theta_t - 1e-6).to(pred.dtype)
            l1 = F.l1_loss(pred, tgt)
            region = (torch.abs(pred - tgt) * clipped_mask).sum() / clipped_mask.sum().clamp(min=1.0)
            loss = l1 + 2.0 * region + 0.5 * _mr_stft_loss(pred, tgt)

            if not torch.isfinite(loss):
                # §0a/§V6 (copilot-instructions.md): nicht-finiter Loss wird laut übersprungen.
                logger.warning("Epoch %d Schritt %d: nicht-finiter Loss — Schritt übersprungen (§0a)", epoch + 1, step)
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(steps_per_epoch, 1)
        scheduler.step()
        print(f"Epoch {epoch + 1:3d}/{epochs} | Loss {avg_loss:.4f} | {time.time() - t0:.0f}s", flush=True)

        base_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "aspade_config": aspade_config,
            "epoch": epoch,
            "loss": avg_loss,
            "best_loss": min(best_loss, avg_loss),
            "seed": seed,
        }
        save_guarded(ckpt_latest, base_payload)  # §v10.16-Artefaktschutz
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_guarded(ckpt_best, {**base_payload, "best_loss": best_loss})

    print(f"Best loss: {best_loss:.4f}")

    if skip_export:
        print("ONNX-Export übersprungen (--skip-export)")
        return ckpt_best

    model.eval()
    export_onnx(model, out_dir / "aspade_class_v1.onnx", SEGMENT_SAMPLES)
    rel = check_onnx_parity(model, out_dir / "aspade_class_v1.onnx", seed)
    print(
        f"ONNX-Parität (Torch vs. ONNX-CPU): rel={rel:.2e} (Gate ≤ {PARITY_REL_MAX:.0e}, §III.9 (copilot-instructions.md))"
    )
    if rel > PARITY_REL_MAX:
        # fail-closed wie scripts/export_cantus_onnx.py — kein stilles Ausspeichern (§V6, copilot-instructions.md).
        logger.error("ONNX-Paritäts-Gate verfehlt (rel=%.2e > %.0e) — Abbruch Exit 2", rel, PARITY_REL_MAX)
        raise SystemExit(2)
    return ckpt_best


def export_onnx(model: AspadeClassNet, onnx_path: Path, seg_samples: int) -> None:
    """ONNX-Export (opset 17, dynamische Zeitachse, dynamo=False wie train_bw_v5.py)."""
    model.eval()
    # Device-Paritaet (§III.9 (copilot-instructions.md)): Dummies auf das
    # Modell-Device — CPU-Dummy + CUDA-Modell crasht in conv1d (Ketten-Befund).
    _dev = next(model.parameters()).device
    dummy_y = torch.zeros(1, 1, seg_samples, dtype=torch.float32, device=_dev)
    dummy_theta = torch.tensor([[[0.3]]], dtype=torch.float32, device=_dev)
    torch.onnx.export(
        model,
        (dummy_y, dummy_theta),
        str(onnx_path),
        input_names=["y", "theta"],
        output_names=["x"],
        dynamic_axes={"y": {2: "time"}, "x": {2: "time"}},
        opset_version=17,
        dynamo=False,
    )
    import onnx  # pylint: disable=import-outside-toplevel

    onnx.checker.check_model(str(onnx_path))
    print(f"ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")


def check_onnx_parity(model: AspadeClassNet, onnx_path: Path, seed: int) -> float:
    """Torch↔ONNX-CPU-Parität auf strukturiertem Feed (§III.9 (copilot-instructions.md))."""
    import onnxruntime as ort  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(_seed_for(seed, "onnx-parity"))
    n = 8192
    t = np.arange(n, dtype=np.float32) / SR
    clean = (0.7 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 660.0 * t)).astype(np.float32)
    theta = float(rng.uniform(THETA_MIN, THETA_MAX))
    clipped = np.clip(clean, -theta, theta).astype(np.float32)
    y_np = clipped[None, None, :]
    theta_np = np.array([[[theta]]], dtype=np.float32)

    with torch.inference_mode():
        _dev2 = next(model.parameters()).device
        ref = model(torch.from_numpy(y_np).to(_dev2), torch.from_numpy(theta_np).to(_dev2)).detach().cpu().numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: y_np, sess.get_inputs()[1].name: theta_np}
    got = sess.run(None, feed)[0]
    return float(np.max(np.abs(ref - got)) / (float(np.max(np.abs(ref))) + 1e-9))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="SOTA-M8 (D-K4-1): A-SPADE-Klassen-Declipper-Training (MUSDB18-HQ-Vocals)")
    p.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    p.add_argument("--splits", type=str, default="train", help="kommagetrennte MUSDB18-HQ-Splits (Default: train)")
    p.add_argument("--out-dir", type=Path, default=Path("models/aspade"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--stages", type=int, default=4, help="ADMM-Stufen (A-SPADE-Klasse)")
    p.add_argument("--channels", type=int, default=24)
    p.add_argument("--kernel-size", type=int, default=7)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Determinismus §G5 (copilot-instructions.md)")
    p.add_argument("--device", type=str, default="cpu", help="cpu oder cuda (GPU-Training)")
    p.add_argument("--resume", type=Path, default=None, help="checkpoint_latest.pt zum Fortsetzen")
    p.add_argument("--max-tracks", type=int, default=0, help="0 = alle Tracks (Smoke: 1–2)")
    p.add_argument("--skip-export", action="store_true", help="ONNX-Export + Paritäts-Gate überspringen")
    args = p.parse_args()

    train(
        musdb_root=args.musdb_root,
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        steps_per_epoch=args.steps_per_epoch,
        stages=args.stages,
        channels=args.channels,
        kernel_size=args.kernel_size,
        seed=args.seed,
        device_str=args.device,
        resume=args.resume,
        max_tracks=args.max_tracks,
        skip_export=args.skip_export,
    )
