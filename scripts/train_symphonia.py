#!/usr/bin/env python3
"""Trainiert Symphonia ausschließlich auf Instrumental-Paaren.

Die Paare müssen aus ``generate_synthetic_degraded_instrumentals.py`` stammen:
die Referenz ist immer drums+bass+other.  Die Validation wird ausschließlich
aus MUSDBs ``test``-Split gelesen; eine zufällige Teilung derselben Songs wäre
Data-Leakage und ist für einen produktiven Checkpoint unzulässig.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.symphonia.symphonia_model import create_symphonia
from scripts.train_cantus import (
    ConditionExtractor,
    MelSpectralLoss,
    PairDataset,
    StftPhaseLoss,
    TemporalConsistencyLoss,
    _flow_pair,
    _set_seeds,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "models" / "symphonia" / "symphonia_config.json"
CHECKPOINT_DIR = ROOT / "models" / "symphonia"


class InstrumentalConditionExtractor(ConditionExtractor):
    """MERT/MuQ-Bedingung plus deterministischer Energie-/Onset-Track.

    ``PairDataset`` nennt den zweiten Frame-Stream aus Kompatibilitätsgründen
    ``pitch``.  Hier enthält er ausschließlich ``(frame_energy, onset)``;
    damit gelangen keine Vocal-Pitch-Features in Symphonia.
    """

    def __init__(self, sr: int) -> None:
        super().__init__(sr=sr)
        self._mert_cpu_session: Any | None = None

    def _extract_mert(self, mono: np.ndarray) -> np.ndarray | None:
        """Liest die CPU-Referenz, nie den bekannten fehlerhaften ORT-ROCm-EP.

        Der Trainings-Feature-Raum muss exakt der ONNX-CPU-Referenz entsprechen
        (§III.9 (copilot-instructions.md)); ROCm wird ausschließlich vom
        paritätsverifizierten Torch-Kern für die Restaurierung genutzt.
        """
        if self._mert_failed:
            return None
        try:
            if self._mert_cpu_session is None:
                import onnxruntime as ort

                path = ROOT / "models" / "mert" / "mert.onnx"
                self._mert_cpu_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            audio = np.asarray(mono, dtype=np.float32).reshape(1, -1)
            audio /= float(np.max(np.abs(audio))) + 1e-10
            feature = self._mert_cpu_session.run(None, {"input_values": audio})[0][0]
            return np.ascontiguousarray(feature, dtype=np.float32)
        except Exception as exc:
            self._mert_failed = True
            self._warn_once("MERT-CPU-Referenz", exc)
            return None

    def _extract_pitch(self, mono: np.ndarray, n_frames: int) -> np.ndarray:
        envelope = np.abs(np.asarray(mono, dtype=np.float32).reshape(-1))
        boundaries = np.linspace(0, envelope.size, n_frames + 1, dtype=np.int64)
        energy = np.zeros(n_frames, dtype=np.float32)
        for index in range(n_frames):
            frame = envelope[boundaries[index] : boundaries[index + 1]]
            energy[index] = float(np.mean(frame)) if frame.size else 0.0
        onset = np.maximum(np.diff(energy, prepend=energy[:1]), 0.0)
        rhythm = np.stack((energy, onset), axis=-1)
        rhythm /= np.maximum(np.max(rhythm, axis=0, keepdims=True), 1e-8)
        return np.nan_to_num(rhythm, nan=0.0, posinf=0.0, neginf=0.0)


class InstrumentalLoss(torch.nn.Module):
    """Flow-, Spektral-, Phasen-, Transienten- und Zeitverlust ohne Vocal-MOS."""

    def __init__(self, weights: dict[str, float], sr: int) -> None:
        super().__init__()
        self.weights = weights
        self.mel = MelSpectralLoss(sr=sr)
        self.phase = StftPhaseLoss()
        self.temporal = TemporalConsistencyLoss(sr=sr)

    @staticmethod
    def _onset_envelope(wave: torch.Tensor, frame: int = 1024, hop: int = 256) -> torch.Tensor:
        mono = wave.squeeze(1)
        frames = mono.unfold(-1, frame, hop)
        energy = torch.sqrt(torch.mean(frames.square(), dim=-1) + 1e-10)
        return functional.relu(torch.diff(energy, dim=-1, prepend=energy[..., :1]))

    def forward(
        self,
        velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        x_t: torch.Tensor,
        clean: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        estimate = x_t + (1.0 - t.reshape(-1, 1, 1)) * velocity
        flow = functional.mse_loss(velocity, target_velocity)
        mel = self.mel(estimate, clean)
        phase = self.phase(estimate, clean)
        onset = functional.l1_loss(self._onset_envelope(estimate), self._onset_envelope(clean))
        temporal = self.temporal(estimate, clean)
        parts = {
            "flow_matching": flow,
            "mel_spectral": mel,
            "stft_phase": phase,
            "onset_preservation": onset,
            "temporal": temporal,
        }
        total = sum(self.weights[name] * value for name, value in parts.items())
        return total, parts


def _collect_split(manifest: Path, split: str) -> Path:
    """Erzeugt eine deterministische Split-Ansicht ohne Songüberschneidung."""
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"Manifest {manifest} enthält keinen '{split}'-Split")
    view = manifest.with_name(f".{manifest.stem}_{split}.jsonl")
    view.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    return view


def _model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loss: InstrumentalLoss,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    max_batches: int,
) -> float:
    model.eval()
    values: list[float] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)
        t = torch.rand(clean.shape[0], generator=generator, device=device)
        x_t, target = _flow_pair(clean, degraded, t)
        velocity = model(
            x_t.transpose(1, 2),
            t,
            batch["mert"].to(device),
            batch["pitch"].to(device),
            batch["harm"].to(device),
            batch["use_cond"].to(device),
        ).transpose(1, 2)
        total, _ = loss(velocity, target, x_t, clean, t)
        if torch.isfinite(total):
            values.append(float(total))
    model.train()
    return float(np.mean(values)) if values else float("inf")


def _payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    val_loss: float,
    phase: str,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "phase": phase,
        "seed": seed,
        "config": config,
        "data_contract": "MUSDB18-HQ drums+bass+other; train/test song-isolated",
    }


def train(args: argparse.Namespace) -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    training = config["training"]
    seed = int(args.seed if args.seed is not None else training["seed"])
    _set_seeds(seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    manifests = [Path(value.strip()) for value in args.data.split(",") if value.strip()]
    if not manifests:
        raise ValueError("--data benötigt mindestens ein Manifest")
    train_views = [_collect_split(manifest, "train") for manifest in manifests]
    test_views = [_collect_split(manifest, "test") for manifest in manifests]
    samples = int(float(args.chunk_sec) * int(training["sr"]))
    extractor = InstrumentalConditionExtractor(sr=int(training["sr"]))
    train_data = PairDataset(train_views, chunk_samples=samples, extractor=extractor, seed=seed)
    test_data = PairDataset(test_views, chunk_samples=samples, extractor=extractor, seed=seed + 1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if not len(train_loader):
        raise ValueError("Zu wenige Trainingspaare für die gewählte Batch-Größe")
    model_config = config["model"][args.preset]
    model = create_symphonia(**_model_kwargs(model_config)).to(device)
    weights = training["loss_weights"]
    loss = InstrumentalLoss(weights, int(training["sr"])).to(device)
    learning_rate = float(args.lr if args.lr is not None else training[f"lr_{args.phase}"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=float(training["weight_decay"]), betas=(0.9, 0.999)
    )
    start_epoch = step = 0
    best = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch, step, best = int(checkpoint["epoch"]), int(checkpoint["step"]), float(checkpoint["val_loss"])
    checkpoint_dir = Path(args.output_dir) if args.output_dir else CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": args.phase,
        "preset": args.preset,
        "seed": seed,
        "device": str(device),
        "manifests": [str(manifest) for manifest in manifests],
        "checkpoints": [],
    }
    logger.info(
        "Symphonia: phase=%s epochs=%d device=%s params=%.1fM", args.phase, args.epochs, device, model.num_params
    )
    for epoch in range(start_epoch, args.epochs):
        train_data.set_epoch(epoch)
        model.train()
        for index, batch in enumerate(train_loader):
            if index >= args.steps_per_epoch:
                break
            clean, degraded = batch["clean"].to(device), batch["degraded"].to(device)
            t = torch.rand(clean.shape[0], device=device)
            x_t, target = _flow_pair(clean, degraded, t)
            use_cond = batch["use_cond"].to(device)
            if model_config["cond_dropout"]:
                use_cond *= (torch.rand(clean.shape[0], device=device) >= model_config["cond_dropout"]).float()
            velocity = model(
                x_t.transpose(1, 2),
                t,
                batch["mert"].to(device),
                batch["pitch"].to(device),
                batch["harm"].to(device),
                use_cond,
            ).transpose(1, 2)
            total, parts = loss(velocity, target, x_t, clean, t)
            if not torch.isfinite(total):
                logger.warning("§0a (copilot-instructions.md): nicht-finiter Symphonia-Loss, Schritt verworfen")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                logger.info(
                    "step=%d loss=%.5f flow=%.5f mel=%.5f phase=%.5f onset=%.5f",
                    step,
                    float(total),
                    float(parts["flow_matching"]),
                    float(parts["mel_spectral"]),
                    float(parts["stft_phase"]),
                    float(parts["onset_preservation"]),
                )
        validation = _evaluate(model, loss, test_loader, device, seed + epoch, args.max_validation_batches)
        payload = _payload(
            model,
            optimizer,
            epoch=epoch + 1,
            step=step,
            val_loss=validation,
            phase=args.phase,
            seed=seed,
            config=config,
        )
        torch.save(payload, checkpoint_dir / "checkpoint_latest.pt")
        if validation < best:
            best = validation
            torch.save(payload, checkpoint_dir / "checkpoint_best.pt")
        report["checkpoints"].append({"epoch": epoch + 1, "step": step, "val_loss": validation})
        logger.info("epoche=%d validierung=%.6f bestwert=%.6f", epoch + 1, validation, best)
    report["best_val_loss"] = best
    (checkpoint_dir / f"train_report_{args.phase}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="MUSDB-Instrumental-manifest.jsonl")
    parser.add_argument("--phase", choices=("pretrain", "fine_tune", "domain_adapt"), default="pretrain")
    parser.add_argument("--preset", choices=("full", "tiny"), default="full")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-sec", type=float, default=4.0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-validation-batches", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, help="Abweichendes Ziel nur für Trainings-Smokes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
