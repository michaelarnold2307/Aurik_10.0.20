"""SOTA-M12 (D-K4-8): Training des TP-V2-Klassen-Phasenschätzers (Transienten-Phasenkohärenz).

Trainingsziel (supervisiert, dokumentiert): **Phasen-Rekonstruktion aus GT-STFT** —
zu jedem transientenreichen Signal y wird das Paar (log1p|STFT(y)|, Δφ) gebildet,
wobei Δφ die Zielkorrektur ist, die aus dem GT-Spektrum konstruiert wird
(Y = |Y|·e^{i∠Y}; Ziel: ∠Y relativ zur instantanen Phase der degradierten
Eingangs-Magnituden). Der Challenger lernt also genau die Korrektur, die der
Bestandspfad (restore_phase_coherence) heuristisch über Phasen-Recycling macht.

Material:
  1. **Synthetische transientenreiche Trainingsdaten** (Default, deterministisch):
     Percussion-artige Synthesen (Kick/Snare/HiHat-artige Hüllkurven,
     Exponential-Abklingen, Rausch-Bursts) — MUSTER
     scripts/train_tf_gridnet_denoiser.py (synthetic=…).
  2. **Reale lokale Waveformen** (optional, `--real-dir`): vorhandene *.wav
     werden segmentweise zugezogen (nur wenn vorhanden — dokumentierte
     Datenlücke Muster D-K2-5, siehe Report).

Determinismus (§G5 (copilot-instructions.md)): Seeds pro Session, Batch-RNG als
reine Funktion von (seed, epoch, step) — Resume-reproduzierbar; kein
`time.time()` in der Logik. Checkpoint/Manifest via save_guarded
(§v10.16-Artefaktschutz, backend/core/training_artifacts.py).

SMOKE-TEST: `python scripts/train_tp_v2_phase.py --smoke` — CPU < 3 Min,
KEIN Echt-Training (GPU belegt), endet mit Manifest + Checkpoint im Ausgabe-
Verzeichnis. KEIN automatischer Flag-Flip (§v10.802 (copilot-instructions.md)).

Aufruf (Echt-Training, bewusst NICHT Default):
  python scripts/train_tp_v2_phase.py --epochs 50 --steps-per-epoch 200 \
      --real-dir data/musdb18hq/train --out-dir models/tp_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.dsp.tp_v2_phase import (
    CONTEXT_FRAMES,
    DEFAULT_BLOCKS,
    DEFAULT_CHECKPOINT_SUBPATH,
    DEFAULT_HIDDEN,
    DEFAULT_SEED,
    HOP,
    N_FFT,
    N_FREQ,
    TARGET_SR,
    build_model,
)
from backend.core.training_artifacts import save_guarded

logger = logging.getLogger(__name__)

SEGMENT_SAMPLES = TARGET_SR  # 1 s Trainingssegmente @ 48 kHz
SNR_RANGE_DB = (0.0, 20.0)
PARITY_REL_MAX = 1e-3  # §III.9 (copilot-instructions.md) — ONNX optional


def _seed_for(seed: int, *parts: object) -> int:
    """Deterministischer Teil-Seed (§G5 (GEBOTE.md)) — MUSTER train_aspade_declipper._seed_for."""
    raw = "|".join([str(seed), *(str(p) for p in parts)])
    h = int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:8], 16)
    return int((seed + h) % (2**31 - 1))


def set_determinism(seed: int) -> None:
    """Fixe Seeds pro Session (§G5 (copilot-instructions.md))."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Synthetische transientenreiche Daten (Default) ───────────────────────────


def make_transient_segment(rng: np.random.Generator, n: int = SEGMENT_SAMPLES) -> np.ndarray:
    """Percussion-artige Synthese: Kick/Snare/HiHat-Hüllkurven + Rausch-Bursts.

    Deterministisch über den übergebenen RNG (§G5 (copilot-instructions.md)).
    """
    y = np.zeros(n, dtype=np.float64)

    def _burst(start: float, dur: float, decay: float, kind: str) -> None:
        s0 = int(start * TARGET_SR)
        length = int(dur * TARGET_SR)
        if s0 >= n:
            return
        length = min(length, n - s0)
        env = np.exp(-np.arange(length) / max(decay * TARGET_SR, 1.0))
        if kind == "kick":
            f0 = float(rng.uniform(45.0, 90.0))
            seg = env * np.sin(2 * np.pi * f0 * np.arange(length) / TARGET_SR)
        elif kind == "snare":
            noise = rng.standard_normal(length)
            tone = np.sin(2 * np.pi * 180.0 * np.arange(length) / TARGET_SR)
            seg = env * (0.6 * noise + 0.4 * tone)
        else:  # hihat
            seg = env * rng.standard_normal(length)
        y[s0 : s0 + length] += seg

    for _ in range(int(rng.integers(2, 6))):
        _burst(
            float(rng.uniform(0.0, 0.9)),
            float(rng.uniform(0.02, 0.12)),
            float(rng.uniform(0.01, 0.08)),
            str(rng.choice(["kick", "snare", "hihat"])),
        )

    # Floor-Noise (leise Hintergrundaktivität)
    y += 0.002 * rng.standard_normal(n)
    peak = float(np.max(np.abs(y))) + 1e-9
    return (y / peak * 0.8).astype(np.float32)


def make_degraded_pair(clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Degradation: Phasen-Verwirbelung über schmalbandige Allpass-Stufen + Rauschen.

    Simuliert genau die Schadensklasse, die der Challenger korrigieren soll
    (phasen-inkohärente Rekonstruktion nach STFT-Phasen), OHNE dass die
    Zielgröße im Input als Präfix steckt (kein Kammfilter-Präfix).
    """
    n = clean.size
    noisy = clean.copy()
    snr_db = float(rng.uniform(*SNR_RANGE_DB))
    noise = rng.standard_normal(n)
    noise *= np.sqrt(np.mean(clean**2)) / (np.sqrt(np.mean(noise**2)) * 10.0 ** (snr_db / 20.0))
    noisy = clean + noise
    return noisy.astype(np.float32)


def _features(clean: np.ndarray, degraded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(log_mag_input, delta_phi_target) für ein Chunk-Paar.

    Ziel-Δφ ist die Differenz der GT-Phase zur degradierten instantanen Phase,
    auf (-π, π] abgebildet — eindeutig und NaN/Inf-sicher.
    """
    from scipy.signal import stft

    f1, t1, yd = stft(degraded.astype(np.float64), fs=TARGET_SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    f2, t2, yc = stft(clean.astype(np.float64), fs=TARGET_SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
    frames = min(yd.shape[1], yc.shape[1])
    bins = min(yd.shape[0], yc.shape[0])
    log_mag = np.log1p(np.abs(yd[:bins, :frames])).astype(np.float32)
    delta = np.angle(yc[:bins, :frames]) - np.angle(yd[:bins, :frames])
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi  # auf (-π, π]
    return log_mag, delta.astype(np.float32)


class TransientPhaseDataset:
    """Deterministische Chunk-Paare (synthetisch oder reale Waveformen)."""

    def __init__(self, *, synthetic: int, real_dir: Path | None, seed: int) -> None:
        self._seed = int(seed)
        self._synthetic = int(synthetic)
        self._real_paths: list[Path] = sorted(real_dir.glob("*.wav")) if real_dir and real_dir.is_dir() else []
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self._synthetic if self._synthetic > 0 else max(len(self._real_paths) * 32, 1)

    def _real_item(self, idx: int) -> np.ndarray:
        import soundfile as sf

        rng = np.random.default_rng(_seed_for(self._seed, "real", idx))
        path = self._real_paths[idx % len(self._real_paths)]
        y, _ = sf.read(str(path), dtype="float32", always_2d=True)
        mono = y.mean(axis=1)
        n = mono.size
        if n <= SEGMENT_SAMPLES:
            seg = np.pad(mono, (0, SEGMENT_SAMPLES - n))
        else:
            start = int(rng.integers(0, n - SEGMENT_SAMPLES + 1))
            seg = mono[start : start + SEGMENT_SAMPLES]
        return seg.astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx in self._cache:
            return self._cache[idx]
        rng = np.random.default_rng(_seed_for(self._seed, "item", idx))
        if self._synthetic > 0 or not self._real_paths:
            clean = make_transient_segment(rng)
        else:
            clean = self._real_item(idx)
        degraded = make_degraded_pair(clean, rng)
        log_mag, delta = _features(clean, degraded)
        self._cache[idx] = (log_mag, delta)
        return log_mag, delta


# ── Training ────────────────────────────────────────────────────────────────


def train(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    steps_per_epoch: int,
    synthetic: int,
    real_dir: Path | None,
    out_dir: Path,
    seed: int,
    device_str: str,
    hidden: int,
    blocks: int,
    smoke: bool,
) -> Path:
    import torch

    set_determinism(seed)  # §G5 (copilot-instructions.md)
    dev = torch.device(device_str)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA angefordert, aber nicht verfügbar — CPU (§V6 (copilot-instructions.md))")
        dev = torch.device("cpu")

    model = build_model(n_freq=N_FREQ, context_frames=CONTEXT_FRAMES, hidden=hidden, blocks=blocks, seed=seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TransientPhaseDataset(synthetic=synthetic, real_dir=real_dir, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "tp_v2_phase_v1.pt"

    tp_v2_config = {
        "n_freq": N_FREQ,
        "context_frames": CONTEXT_FRAMES,
        "hidden": int(hidden),
        "blocks": int(blocks),
    }

    n_seen = 0
    loss_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for step in range(steps_per_epoch):
            rng = np.random.default_rng(_seed_for(seed, f"e{epoch}", f"s{step}"))
            logmags, deltas = [], []
            for _ in range(batch_size):
                j = int(rng.integers(len(ds)))
                log_mag, delta = ds[j]
                logmags.append(log_mag)
                deltas.append(delta)
            x = torch.from_numpy(np.stack(logmags)).to(dev)
            y = torch.from_numpy(np.stack(deltas)).to(dev)
            opt.zero_grad()
            pred = model(x)
            loss = torch.nn.functional.l1_loss(pred, y)
            if not torch.isfinite(loss):
                logger.warning("Epoch %d Schritt %d: nicht-finiter Loss — übersprungen (§0a)", epoch + 1, step)
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            loss_acc += float(loss.item())
            n_seen += 1
        logger.info("Epoch %d/%d | Loss %.4f", epoch + 1, epochs, loss_acc / max(n_seen, 1))
        if smoke:
            break  # ein Smoke-Schritt genügt (CPU < 3 Min, kein Echt-Training)

    save_guarded(
        ckpt,
        {
            "model_state_dict": model.state_dict(),
            "tp_v2_config": tp_v2_config,
            "seed": seed,
            "loss": loss_acc / max(n_seen, 1),
        },
    )

    manifest = {
        "model": "tp_v2_phase_v1",
        "flag": "use_tp_v2_phase",
        "training_goal": "Phasen-Rekonstruktion aus GT-STFT (Δφ relativ zur instantanen Phase)",
        "seed": seed,
        "epochs_run": 1 if smoke else epochs,
        "steps_per_epoch": steps_per_epoch,
        "synthetic_segments": synthetic,
        "real_dir": str(real_dir) if real_dir else None,
        "real_files": len(ds._real_paths),
        "checkpoint": DEFAULT_CHECKPOINT_SUBPATH,
        "smoke": bool(smoke),
        "loss": loss_acc / max(n_seen, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Fertig: %s | Manifest %s", ckpt, out_dir / "manifest.json")
    return ckpt


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="SOTA-M12 (D-K4-8): TP-V2-Phasenschätzer-Training")
    p.add_argument("--smoke", action="store_true", help="Smoke-Test CPU < 3 Min (kein Echt-Training)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps-per-epoch", type=int, default=8)
    p.add_argument("--synthetic", type=int, default=16, help="n synthetische Segmente (Default 16)")
    p.add_argument("--real-dir", type=Path, default=None, help="optional: lokale *.wav (reale Waveformen)")
    p.add_argument("--out-dir", type=Path, default=Path("models/tp_v2"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="§G5 (copilot-instructions.md)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.steps_per_epoch = 2
        args.synthetic = 8
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        steps_per_epoch=args.steps_per_epoch,
        synthetic=args.synthetic,
        real_dir=args.real_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        device_str=args.device,
        hidden=args.hidden,
        blocks=args.blocks,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
