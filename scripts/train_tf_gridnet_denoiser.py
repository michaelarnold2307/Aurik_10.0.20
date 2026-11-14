"""SOTA-M9 (D-K4-2): Training des TF-GridNet-Klassen-Vollband-Denoisers.

Trainiert die TF-GridNet-Klassen-Architektur aus
`backend/core/dsp/tf_gridnet_denoiser.py` (Intra-Frame-Vollband-BiLSTM +
Inter-Frame-Cross-Frame-BiLSTM, strikt nicht-kausal) auf Paar-Daten
(sauber/degradiert) und exportiert das ONNX-Qualitätsmodell
`models/tf_gridnet/tf_gridnet_quality.onnx` im Vertrag des Inferenz-Wrappers
(Input ``log_mag`` (1, F, T) mit F = N_FFT//2+1, T = model_t_frames();
Output ``mask`` (1, F, T) in [0, 1]).

Determinismus (§G5 (copilot-instructions.md)):
    - `--seed` (Default 2026) für random/numpy/torch(+CUDA), cuDNN deterministic.
    - Chunk-Ziehung deterministisch pro (Seed, Index) — kein `time.time()` in
      Entscheidungslogik.

Resume: `--resume models/tf_gridnet/checkpoint_latest.pt` setzt Modell,
Optimizer, Scheduler und Epoche fort (§G5 (copilot-instructions.md)).

ONNX-Export (Standard am Trainingsende, `--skip-onnx` zum Abschalten):
    - Statischer Graph über die CHUNK-Fenstergröße des Wrappers (der dynamische
      LSTM-Export scheitert an torch-2.11-Dynamo-Shape-Constraints; der Wrapper
      padded jedes Inferenz-Fenster exakt auf `window_samples()`, siehe
      Vertrag in backend/core/dsp/tf_gridnet_denoiser.py).
    - `onnx.checker` + Paritäts-Nachweis ONNX-CPU gegen Torch auf strukturierten
      Feeds (normal/sane/const05 analog `scripts/onnx_gpu_compat_scan.py`),
      rel ≤ 1e-3 (§III.9 (copilot-instructions.md)). Fehler ist ein harter
      Abbruch — ein numerisch abweichendes Artefakt wird nicht stillgeschwiegen
      (§V6 (copilot-instructions.md)).

Aktivierung im Produktivpfad erst nach Never-worsen-Gate auf den 3
Referenz-Songs + Hörordnungs-Sign-off (§v10.802 (copilot-instructions.md));
das Flag `use_tf_gridnet_quality` bleibt bis dahin False.

Usage:
    python scripts/train_tf_gridnet_denoiser.py --synthetic 64 --epochs 1
    python scripts/train_tf_gridnet_denoiser.py --clean-dir data/clean \
        --noisy-dir data/noisy --epochs 50 --resume models/tf_gridnet/checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backend.core.dsp.tf_gridnet_denoiser import (
    N_FFT,
    TARGET_SR,
    build_tf_gridnet_model,
    model_t_frames,
    stft_magnitude,
    window_samples,
)
from backend.core.training_artifacts import save_guarded

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = _PROJECT / "models" / "tf_gridnet"
LATEST_PT = CHECKPOINT_DIR / "checkpoint_latest.pt"
BEST_PT = CHECKPOINT_DIR / "tf_gridnet_best.pt"
DEFAULT_ONNX = CHECKPOINT_DIR / "tf_gridnet_quality.onnx"

SNR_RANGE_DB = (0.0, 20.0)
N_FREQ = N_FFT // 2 + 1


# ── Determinismus (§G5 (copilot-instructions.md)) ────────────────────────────


def set_determinism(seed: int) -> None:
    """Fixe Seeds pro Session; keine zeitabhängigen Entscheidungen."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Datensatz ────────────────────────────────────────────────────────────────


def _target_mask(mag_clean: np.ndarray, mag_noisy: np.ndarray) -> np.ndarray:
    """Ideales Ratio-Mask, auf [0, 1] begrenzt (Supervised-Mask-Target)."""
    return np.clip(mag_clean / (mag_noisy + 1e-8), 0.0, 1.0).astype(np.float32)


def _features(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(log_mag, target_mask) für einen Chunk der Wrapper-Fenstergröße."""
    mag_c = stft_magnitude(clean)
    mag_n = stft_magnitude(noisy)
    log_mag = np.log1p(mag_n).astype(np.float32)
    return log_mag, _target_mask(mag_c, mag_n)


class PairedChunkDataset(Dataset):
    """Chunk-Paare (sauber/degradiert); deterministische Chunk-Ziehung.

    Datei-Modus: Paare aus `--clean-dir`/`--noisy-dir` (gleicher Dateiname).
    Synthetik-Modus (Smoke/CI): Sinus + strukturiertes Rauschen mit festem
    Seed pro Index — reproduzierbar (§G5 (copilot-instructions.md)).
    """

    def __init__(
        self,
        clean_dir: Path | None,
        noisy_dir: Path | None,
        *,
        synthetic: int = 0,
        seed: int = 2026,
        exclude_pairs: set[str] | None = None,
    ) -> None:
        self._chunk = window_samples()
        self._seed = int(seed)
        if synthetic > 0:
            self._pairs: list[tuple[Path, Path]] = []
            self._synthetic = int(synthetic)
        else:
            if clean_dir is None or noisy_dir is None:
                raise ValueError("PairedChunkDataset: --clean-dir/--noisy-dir oder --synthetic nötig")
            noisy_index = {p.name: p for p in sorted(noisy_dir.glob("*.wav"))}
            self._pairs = [(p, noisy_index[p.name]) for p in sorted(clean_dir.glob("*.wav")) if p.name in noisy_index]
            # Leak-Schutz (2026-11-14, M9-Gate-Befund): Paare von Gate-Tracks
            # ausschließen — das Gate misst sonst auf Trainingsmaterial.
            if exclude_pairs:
                self._pairs = [(c, n) for c, n in self._pairs if c.name not in exclude_pairs]
            if not self._pairs:
                raise ValueError(
                    f"Keine passenden wav-Paare in {clean_dir} / {noisy_dir} (nach Ausschluss: {len(exclude_pairs) if exclude_pairs else 0})"
                )
            self._synthetic = 0
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self._synthetic if self._synthetic > 0 else len(self._pairs) * 32

    def _synth_item(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self._seed + idx * 7919)
        t = np.arange(self._chunk, dtype=np.float64) / TARGET_SR
        f0 = 110.0 + (idx % 24) * 22.0
        clean = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(2 * np.pi * 2.0 * f0 * t)
        snr_db = SNR_RANGE_DB[0] + (SNR_RANGE_DB[1] - SNR_RANGE_DB[0]) * rng.rand()
        noise = rng.randn(self._chunk)
        noise *= np.sqrt(np.mean(clean**2)) / (np.sqrt(np.mean(noise**2)) * 10.0 ** (snr_db / 20.0))
        return clean.astype(np.float32), (clean + noise).astype(np.float32)

    def _file_item(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        import librosa

        pair = self._pairs[idx % len(self._pairs)]
        rng = np.random.RandomState(self._seed + idx * 7919)
        clean, _ = librosa.load(str(pair[0]), sr=TARGET_SR, mono=True)
        noisy, _ = librosa.load(str(pair[1]), sr=TARGET_SR, mono=True)
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]
        if n <= self._chunk:
            clean = np.pad(clean, (0, self._chunk - n))
            noisy = np.pad(noisy, (0, self._chunk - n))
        else:
            start = int(rng.randint(0, n - self._chunk + 1))
            clean = clean[start : start + self._chunk]
            noisy = noisy[start : start + self._chunk]
        return clean.astype(np.float32), noisy.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx in self._cache:
            log_mag, target = self._cache[idx]
        else:
            clean, noisy = self._synth_item(idx) if self._synthetic > 0 else self._file_item(idx)
            log_mag, target = _features(clean, noisy)
            self._cache[idx] = (log_mag, target)
        return {
            "log_mag": torch.from_numpy(log_mag.copy()),
            "target": torch.from_numpy(target.copy()),
        }


# ── Training ─────────────────────────────────────────────────────────────────


def train(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    steps_per_epoch: int,
    resume: str | None,
    seed: int,
    clean_dir: Path | None,
    noisy_dir: Path | None,
    synthetic: int,
    device: str,
    hidden: int,
    blocks: int,
    skip_onnx: bool,
    grad_accum: int = 1,
    exclude_pairs: set[str] | None = None,
    checkpoint_dir: Path | None = None,
) -> Path:
    set_determinism(seed)  # §G5 (copilot-instructions.md)
    # M9-V2 (2026-11-14): Kandidaten-Läufe schreiben in einen eigenen Ordner,
    # damit die deployten Produktions-Artefakte (models/tf_gridnet/) bis zum
    # Gate-Bestehen unangetastet bleiben.
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINT_DIR
    latest_pt = ckpt_dir / "checkpoint_latest.pt"
    best_pt = ckpt_dir / "tf_gridnet_best.pt"
    onnx_pt = ckpt_dir / "tf_gridnet_quality.onnx"
    if int(grad_accum) < 1:
        raise ValueError(f"--grad-accum muss >= 1 sein (erhalten: {grad_accum})")
    grad_accum = int(grad_accum)
    dev = torch.device(device)
    model = build_tf_gridnet_model(n_freq=N_FREQ, hidden=hidden, n_blocks=blocks).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)

    start_epoch, best_val = 0, float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("val_loss", best_val))
        logger.info("Resume ab Epoche %d (best_val=%.4f)", start_epoch, best_val)

    logger.info(
        "Effektive Batch-Größe %d (batch_size=%d × grad_accum=%d)",
        batch_size * grad_accum,
        batch_size,
        grad_accum,
    )
    dataset = PairedChunkDataset(clean_dir, noisy_dir, synthetic=synthetic, seed=seed, exclude_pairs=exclude_pairs)
    val_set = PairedChunkDataset(
        clean_dir,
        noisy_dir,
        synthetic=max(synthetic, 0),
        seed=seed + 500_000,
        exclude_pairs=exclude_pairs,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        train_loss, seen = 0.0, 0
        # --steps-per-epoch zaehlt OPTIMIZER-Schritte; mit Akkumulation braucht
        # jeder davon grad_accum Micro-Batches. Sonst saehe der Lauf mit
        # --grad-accum 4 nur ein Viertel der Samples je Epoche.
        micro_steps = int(steps_per_epoch) * grad_accum
        for step, batch in enumerate(train_loader):
            if step >= micro_steps:
                break
            log_mag = batch["log_mag"].to(dev)
            target = batch["target"].to(dev)
            if step % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)
            pred = model(log_mag)
            loss = F.mse_loss(pred, target)
            # Gradient-Akkumulation: die Inter-Frame-BiLSTM-Aktivierungen des
            # 4-s-CHUNK (b·f Sequenzen) sprengen sonst 24 GiB (Befund 2026-11-06:
            # OOM bei batch_size=4, 4,42 GiB angefragt). Rechnerisch identisch zu
            # einem großen Batch, nur mit kleinerem Peak.
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0 or step + 1 == micro_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            train_loss += float(loss.item())
            seen += 1
        scheduler.step()

        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                if vn >= 20:
                    break
                pred = model(vb["log_mag"].to(dev))
                val_loss += float(F.mse_loss(pred, vb["target"].to(dev)).item())
                vn += 1
        avg_train = train_loss / max(seen, 1)
        avg_val = val_loss / max(vn, 1)
        logger.info(
            "Ep %d/%d | Tr %.4f | Val %.4f | %.0fs",
            epoch + 1,
            epochs,
            avg_train,
            avg_val,
            time.time() - t0,
        )

        save_guarded(
            latest_pt,
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": avg_val,
                "seed": seed,
                "grad_accum": grad_accum,
            },
        )
        if avg_val < best_val:
            best_val = avg_val
            save_guarded(
                best_pt,
                {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "val_loss": avg_val, "seed": seed},
            )
            logger.info(">> Best: %.4f", best_val)

    if not skip_onnx:
        model.load_state_dict(torch.load(best_pt, map_location=dev, weights_only=False)["model_state_dict"])
        export_onnx(model, onnx_pt, device=dev)
    logger.info("Fertig. best_val=%.4f | %s", best_val, best_pt)
    return best_pt


# ── ONNX-Export + Paritäts-Nachweis (§III.9 (copilot-instructions.md)) ───────


def _verify_onnx_parity(model: torch.nn.Module, onnx_path: Path, device: torch.device) -> None:
    """ONNX-CPU gegen Torch auf strukturierten Feeds (normal/sane/const05)."""
    import onnxruntime as ort

    rng = np.random.RandomState(2026)
    t_model = model_t_frames()
    feeds = {
        "normal": (0.7 * rng.randn(1, N_FREQ, t_model)).astype(np.float32),
        "sane": np.clip(0.25 * rng.randn(1, N_FREQ, t_model) + 0.6, 0.0, 2.0).astype(np.float32),
        "const05": np.full((1, N_FREQ, t_model), 0.5, dtype=np.float32),
    }
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ONNX-CPU only: kein Paritäts-Nachweis für Nicht-CPU-EPs vorhanden
    # (§III.9 (copilot-instructions.md)).
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    worst = 0.0
    with torch.no_grad():
        for name, feed in feeds.items():
            out = sess.run(None, {in_name: feed})[0]
            ref = model(torch.from_numpy(feed).to(device)).cpu().numpy()
            rel = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
            worst = max(worst, rel)
            logger.info("Parität %s: rel=%.2e", name, rel)
    if worst > 1e-3:
        raise RuntimeError(
            f"ONNX-Parität verfehlt (rel={worst:.2e} > 1e-3, §III.9 (copilot-instructions.md)) — Export verworfen"
        )


def export_onnx(model: torch.nn.Module, out_path: Path, *, device: torch.device) -> None:
    """Statischer ONNX-Export im Wrapper-Vertrag + Checker + Parität."""
    import onnx

    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, N_FREQ, model_t_frames(), device=device)
    torch.onnx.export(
        model,
        (dummy,),
        f=str(out_path),
        input_names=["log_mag"],
        output_names=["mask"],
        opset_version=18,
    )
    onnx.checker.check_model(onnx.load(str(out_path)))
    _verify_onnx_parity(model, out_path, device)
    logger.info("ONNX exportiert: %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Train TF-GridNet-class full-band denoiser (SOTA-M9, D-K4-2)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Micro-Batch-Akkumulation (Speicherbudget 24 GiB); effektiv = batch_size × grad_accum",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=2026)  # §G5 (copilot-instructions.md)
    p.add_argument("--clean-dir", type=Path, default=None)
    p.add_argument("--noisy-dir", type=Path, default=None)
    p.add_argument("--synthetic", type=int, default=0, help="n synthetische Paare (Smoke/CI)")
    p.add_argument(
        "--exclude-pairs",
        type=Path,
        default=None,
        help="Textdatei mit auszuschließenden Paar-Dateinamen (je Zeile) — Leak-Schutz gegen Gate-Tracks",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Ziel-Ordner für Checkpoints/ONNX (Default: models/tf_gridnet) — Kandidaten-Läufe schreiben in einen eigenen Ordner",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--skip-onnx", action="store_true")
    args = p.parse_args()
    exclude_pairs: set[str] | None = None
    if args.exclude_pairs is not None:
        if not args.exclude_pairs.is_file():
            print(f"ABBRUCH: --exclude-pairs Datei fehlt: {args.exclude_pairs}")
            raise SystemExit(2)
        exclude_pairs = {
            zeile.strip() for zeile in args.exclude_pairs.read_text(encoding="utf-8").splitlines() if zeile.strip()
        }
        if not exclude_pairs:
            print(f"ABBRUCH: --exclude-pairs Datei ist leer: {args.exclude_pairs}")
            raise SystemExit(2)
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        steps_per_epoch=args.steps_per_epoch,
        resume=args.resume,
        seed=args.seed,
        clean_dir=args.clean_dir,
        noisy_dir=args.noisy_dir,
        synthetic=args.synthetic,
        device=args.device,
        hidden=args.hidden,
        blocks=args.blocks,
        skip_onnx=args.skip_onnx,
        grad_accum=args.grad_accum,
        exclude_pairs=exclude_pairs,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
