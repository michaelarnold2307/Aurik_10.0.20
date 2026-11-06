"""SOTA-M12 (D-K4-7): Training des WF-V4-Klassen-Warp-Head.

Trainingsziel (supervisiert, dokumentiert): Schätzung der Warp-Kurve
``warp_log2(t)`` (Rest-Wow/Flutter unter der Hörschwelle, |warp| ≤
``WARP_LIMIT_LOG2`` ≈ 50 Cents) aus beobachteter Pitch-Trajektorie,
Tracker-Konfidenz und Spektral-Features — der Challenger zur heuristischen
Phase-12-Warp-Schätzung.

Material (deterministisch synthetisch, §G5 (copilot-instructions.md)):
  Saubere harmonische Signale (6–12 Partialtöne, 55–330 Hz) werden mit
  BEKANNTEN Warp-Kurven degradiert (Wow 0,5–6 Hz + Flutter 8–60 Hz +
  Drift, Summe ≤ WARP_LIMIT_LOG2): y(t) = x(φ(t)) mit φ' = 2^warp
  (Capstan-Prinzip, inverse-geschwindigkeits-Warp — Muster
  backend/core/phases/phase_12_wow_flutter_fix.py). GT = Warp-Kurve.

  Modelleingaben folgen dem Inferenz-Vertrag
  ``build_wf4_warp_model.forward(x_f0, x_conf, x_feat)``:
    x_f0   — beobachtete Pitch-Abweichung (log2) inkl. FCPE/CRePE-artigem
             Tracker-Rauschen (GT bleibt die Warp-Kurve),
    x_conf — Tracker-Konfidenz (0..1),
    x_feat — ``compute_spectral_features`` der degradierten Waveform.
  ECHT-TRAINING (Musik, §III.11): ``--real-dir data/wf4_warp_pairs`` liest die
  deterministisch erzeugten MUSDB18-HQ-Paare (``scripts/prepare_wf4_warp_pairs.py``,
  Manifest ``aurik.wf4_warp_pairs/1``). Der Modelleingang ``x_f0`` ist dort die
  **gemessene** pYIN-Beobachtung (``pyin_fast``) des cleanen gegen das degradierte
  Segment — kein Zirkelschluss über die GT-Kurve. Fehlt Manifest/Beobachtung,
  bricht der Lader fail-closed ab (§V6 (copilot-instructions.md)) statt still
  synthetisch weiterzutrainieren. Der synthetische Pfad (ohne ``--real-dir``)
  bleibt nur Ersatz und wird als solcher geloggt + im Manifest markiert
  (``material: "synthetisch"``).

Determinismus (§G5 (copilot-instructions.md)): Seeds pro Session, Batch-RNG
als reine Funktion von (seed, epoch, step); kein ``time.time()`` in Logik.
Checkpoint/Manifest via ``save_guarded`` (§v10.16-Artefaktschutz,
backend/core/training_artifacts.py).

SMOKE-TEST: ``python scripts/train_wf4_warp_head.py --smoke`` — CPU < 3 Min,
KEIN Echt-Training (GPU belegt), endet mit Manifest + Checkpoint im
Ausgabeverzeichnis. KEIN automatischer Flag-Flip (§v10.802
(copilot-instructions.md)).

Aufruf (Echt-Training, bewusst NICHT Default):
  python scripts/train_wf4_warp_head.py --epochs 50 --steps-per-epoch 200 \\
      --real-dir data/wf4_warp_pairs --device cuda \\
      --out-dir models/wf4_warp_head
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

from backend.core.dsp.wf4_warp_head import (
    DEFAULT_DILATIONS,
    DEFAULT_HIDDEN,
    N_SPEC_FEAT,
    TARGET_FPS,
    WARP_LIMIT_LOG2,
    build_wf4_warp_model,
    compute_spectral_features,
)
from backend.core.training_artifacts import save_guarded

logger = logging.getLogger(__name__)

TARGET_SR = 48000
SEGMENT_S = 2.5
SEGMENT_FRAMES = int(SEGMENT_S * TARGET_FPS)
CHECKPOINT_SUBPATH = "wf4_warp_head_v1.pt"
DEFAULT_SEED = 20261027
LOSS_CONF_WEIGHT = 0.1


def _seed_for(seed: int, *parts: object) -> int:
    """Deterministischer Teil-Seed (§G5 (copilot-instructions.md)) — MUSTER train_tp_v2_phase._seed_for."""
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


# ── Synthetische Warp-Daten (Default) ───────────────────────────────────────


def make_warp_curve(rng: np.random.Generator, n_frames: int = SEGMENT_FRAMES) -> np.ndarray:
    """Bekannte Wow/Flutter-Warp-Kurve in log2-Stretch-Einheiten (GT).

    Wow 0,5–6 Hz (5–40 Cents) + Flutter 8–60 Hz (2–20 Cents) + langsame
    Drift-Komponente; Summe auf ``WARP_LIMIT_LOG2`` begrenzt (Vertrag des
    Heads). Deterministisch über den übergebenen RNG (§G5 (copilot-instructions.md)).
    """
    t = np.arange(n_frames, dtype=np.float64) / TARGET_FPS

    def _cents(c: float) -> float:
        return c / 1200.0

    wow_c = float(rng.uniform(5.0, 40.0))
    wow_f = float(rng.uniform(0.5, 6.0))
    flut_c = float(rng.uniform(2.0, 20.0))
    flut_f = float(rng.uniform(8.0, 60.0))
    drift_c = float(rng.uniform(-8.0, 8.0))
    warp = (
        _cents(wow_c) * np.sin(2.0 * np.pi * wow_f * t + float(rng.uniform(0, 2 * np.pi)))
        + _cents(flut_c) * np.sin(2.0 * np.pi * flut_f * t + float(rng.uniform(0, 2 * np.pi)))
        + _cents(drift_c) * (t / max(t[-1], 1e-9))
    )
    return np.clip(warp, -WARP_LIMIT_LOG2, WARP_LIMIT_LOG2).astype(np.float32)


def make_degraded(rng: np.random.Generator, warp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x_f0, x_conf, x_feat) des degradierten Signals zur Warp-Kurve ``warp``.

    x_f0 = Warp + FCPE/CRePE-artiges Tracker-Rauschen (σ ≈ 4 Cents,
    Ausreißer-Sprünge), x_conf = daraus abgeleitete Konfidenz (0..1).
    """
    n = int(SEGMENT_S * TARGET_SR)
    t = np.arange(n, dtype=np.float64) / TARGET_SR
    f0_base = float(rng.uniform(55.0, 330.0))
    n_partials = int(rng.integers(6, 13))

    # Sauberes Signal …
    x = np.zeros(n, dtype=np.float64)
    for k in range(1, n_partials + 1):
        amp = 1.0 / (k ** float(rng.uniform(0.8, 1.4)))
        x += amp * np.sin(2.0 * np.pi * f0_base * k * t + float(rng.uniform(0, 2 * np.pi)))
    x /= np.max(np.abs(x)) + 1e-12

    # … mit bekanntem Warp degradiert: φ(t) = ∫ 2^warp ds (Capstan-Prinzip).
    warp_dense = np.interp(np.linspace(0.0, 1.0, num=n), np.linspace(0.0, 1.0, num=warp.size), warp.astype(np.float64))
    phi = np.cumsum(np.power(2.0, warp_dense)) / TARGET_SR
    y = np.interp(phi, t, x).astype(np.float32)

    # Beobachtete Pitch-Trajektorie mit Tracker-Rauschen (GT bleibt ``warp``).
    noise = rng.standard_normal(warp.size) * (4.0 / 1200.0)
    jumps = rng.random(warp.size) < 0.01
    noise = noise + jumps * rng.standard_normal(warp.size) * (25.0 / 1200.0)
    x_f0 = (warp.astype(np.float64) + noise).astype(np.float32)
    x_conf = np.clip(1.0 - np.abs(noise) * 12.0, 0.0, 1.0).astype(np.float32)
    x_feat = compute_spectral_features(y, TARGET_SR, warp.size)
    return x_f0, x_conf, x_feat


def conf_target(warp: np.ndarray) -> np.ndarray:
    """Supervisions-Ziel des Konfidenz-Heads: Warp-Präsenz (0..1)."""
    return np.clip(np.abs(warp) / (0.25 * WARP_LIMIT_LOG2), 0.0, 1.0).astype(np.float32)


PAIRS_SCHEMA = "aurik.wf4_warp_pairs/1"


def lade_real_paare(real_dir: Path) -> list[dict[str, np.ndarray]]:
    """Lädt WF-V4-Paare **inkl. Tracker-Beobachtung** (fail-closed, §V6 (copilot-instructions.md)).

    Vertrag: ``data/wf4_warp_pairs`` aus ``scripts/prepare_wf4_warp_pairs.py``
    (Schema ``aurik.wf4_warp_pairs/1``) mit ``audio/`` (degradiert), ``warp/``
    (GT), ``f0/`` + ``conf/`` (pYIN-Beobachtung clean ↔ degradiert).

    Fehlt Manifest, Beobachtung oder eine Datei, wird **abgebrochen** — kein
    stiller Rückfall auf synthetische Sinus-Signale (§V6 (copilot-instructions.md)
    (copilot-instructions.md), §III.11: Trainingsdaten müssen Musik sein).
    """
    man_path = real_dir / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(
            f"WF-V4-Paare: Manifest fehlt ({man_path}) — §V6 (copilot-instructions.md) fail-closed, kein synthetischer Ersatz"
        )
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PAIRS_SCHEMA:
        raise ValueError(f"WF-V4-Paare: Schema {manifest.get('schema')!r} != {PAIRS_SCHEMA!r} ({man_path})")
    if manifest.get("f0_beobachtung") is None:
        raise ValueError(
            "WF-V4-Paare ohne Tracker-Beobachtung erzeugt (--ohne-f0) — Echt-Training braucht f0/ + conf/ (§III.11)"
        )
    eintraege = sorted(manifest.get("paare", []), key=lambda e: str(e.get("datei", "")))
    if not eintraege:
        raise ValueError(f"WF-V4-Paare: keine Paare im Manifest ({man_path})")

    import soundfile as sf

    geladen: list[dict[str, np.ndarray]] = []
    n_frames_erwartet: int | None = None
    for e in eintraege:
        name = str(e["datei"])
        pfade = {
            "wav": real_dir / "audio" / f"{name}.wav",
            "warp": real_dir / "warp" / f"{name}.npy",
            "f0": real_dir / "f0" / f"{name}.npy",
            "conf": real_dir / "conf" / f"{name}.npy",
        }
        for art, pfad in pfade.items():
            if not pfad.is_file():
                raise FileNotFoundError(
                    f"WF-V4-Paar {name}: {art}-Datei fehlt ({pfad}) — §V6 (copilot-instructions.md)"
                )
        # channels-last von soundfile ⇒ explizit auf Mono MIT Layout-Check
        # (AGENTS.md §3 Stereo-Layout-Invariante: nie blind axis=0 mitteln).
        y, sr = sf.read(str(pfade["wav"]), dtype="float32", always_2d=True)
        if y.ndim != 2 or y.shape[0] < 1:
            raise ValueError(f"WF-V4-Paar {name}: unerwartetes Layout {y.shape}")
        mono = y[:, 0] if y.shape[1] == 1 else y.mean(axis=1)
        warp = np.load(str(pfade["warp"])).astype(np.float32).ravel()
        x_f0 = np.load(str(pfade["f0"])).astype(np.float32).ravel()
        x_conf = np.load(str(pfade["conf"])).astype(np.float32).ravel()
        n = int(min(warp.size, x_f0.size, x_conf.size))
        if n <= 0:
            raise ValueError(f"WF-V4-Paar {name}: leere Arrays")
        if n_frames_erwartet is None:
            n_frames_erwartet = n
        elif n != n_frames_erwartet:
            raise ValueError(f"WF-V4-Paar {name}: {n} Frames != {n_frames_erwartet} — Batch-Stack nicht möglich")
        warp = warp[:n]
        geladen.append(
            {
                "x_f0": np.nan_to_num(x_f0[:n], nan=0.0, posinf=0.0, neginf=0.0),
                "x_conf": np.clip(np.nan_to_num(x_conf[:n], nan=0.0), 0.0, 1.0).astype(np.float32),
                "x_feat": compute_spectral_features(mono, int(sr), n),
                "y_warp": warp.astype(np.float32),
                "y_conf": conf_target(warp),
            }
        )
    logger.info("WF-V4-Echt-Paare geladen: %d (je %d Frames) aus %s", len(geladen), n_frames_erwartet or 0, real_dir)
    return geladen


def train(
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    synthetic_segments: int,
    seed: int,
    out_dir: Path,
    device: str,
    hidden: int,
    blocks: tuple[int, ...],
    smoke: bool,
    real_dir: Path | None = None,
) -> dict:
    import torch

    set_determinism(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.warning("CUDA angefordert — GPU ist belegt, Training startet auf %s", device)
    dev = torch.device(device)
    model = build_wf4_warp_model(n_feat=N_SPEC_FEAT, hidden=hidden, dilations=blocks).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_synth = 32 if smoke else synthetic_segments
    if smoke:
        epochs, steps_per_epoch = 1, 2

    real_paare = lade_real_paare(real_dir) if real_dir is not None else []
    if real_dir is not None and not real_paare:  # pragma: no cover — defensiv
        raise ValueError(f"--real-dir {real_dir} lieferte keine Paare (§V6 (copilot-instructions.md))")
    if real_paare:
        logger.info("Trainings-Material: %d ECHTE Musik-Paare aus %s", len(real_paare), real_dir)
    else:
        logger.warning(
            "Trainings-Material: synthetische Harmonische (kein --real-dir) — "
            "kein Musik-Training (§III.11), nur Ersatzpfad"
        )

    loss_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for step in range(steps_per_epoch):
            rng = np.random.default_rng(_seed_for(seed, "batch", epoch, step))
            xs_f0, xs_conf, xs_feat, ys_warp, ys_conf = [], [], [], [], []
            if real_paare:
                # Deterministische Permutation je (epoch, step) — §G5 (GEBOTE.md); kein time-Zugriff.
                for j in rng.permutation(len(real_paare))[:batch_size]:
                    p = real_paare[int(j)]
                    xs_f0.append(p["x_f0"])
                    xs_conf.append(p["x_conf"])
                    xs_feat.append(p["x_feat"])
                    ys_warp.append(p["y_warp"])
                    ys_conf.append(p["y_conf"])
            else:
                for _ in range(min(batch_size, n_synth)):
                    warp = make_warp_curve(rng)
                    x_f0, x_conf, x_feat = make_degraded(rng, warp)
                    xs_f0.append(x_f0)
                    xs_conf.append(x_conf)
                    xs_feat.append(x_feat)
                    ys_warp.append(warp)
                    ys_conf.append(conf_target(warp))
            xf = torch.from_numpy(np.stack(xs_f0)).float().to(dev)
            xc = torch.from_numpy(np.stack(xs_conf)).float().to(dev)
            xm = torch.from_numpy(np.stack(xs_feat)).float().to(dev)
            yw = torch.from_numpy(np.stack(ys_warp)).float().to(dev)
            yc = torch.from_numpy(np.stack(ys_conf)).float().to(dev)

            pred_warp, pred_conf = model(xf, xc, xm)
            loss = torch.nn.functional.mse_loss(pred_warp, yw) + LOSS_CONF_WEIGHT * torch.nn.functional.mse_loss(
                pred_conf, yc
            )
            if not torch.isfinite(loss):
                logger.warning("Epoch %d Step %d: nicht-endlicher Loss übersprungen", epoch, step)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            loss_acc = float(loss.detach().cpu())
        logger.info("Epoch %d/%d Loss %.6f", epoch + 1, epochs, loss_acc)

    ckpt = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "wf4_warp_config": {
            "n_feat": N_SPEC_FEAT,
            "hidden": hidden,
            "dilations": list(blocks),
            "target_fps": TARGET_FPS,
        },
        "seed": seed,
        "loss": loss_acc,
    }
    save_guarded(out_dir / CHECKPOINT_SUBPATH, ckpt)

    manifest = {
        "model": "wf4_warp_head",
        "flag": "use_wf4_warp_head",
        "training_goal": "Warp-Kurve warp_log2(t) aus beobachteter Pitch-Trajektorie + Spektral-Features (Capstan-Prinzip; GT aus bekannter Warp-Degradation, Eingang x_f0 als pYIN-Beobachtung clean↔degradiert)",
        "seed": seed,
        "epochs_run": epochs,
        "steps_per_epoch": steps_per_epoch,
        "synthetic_segments": n_synth,
        "real_dir": str(real_dir) if real_dir is not None else None,
        "real_pairs": len(real_paare),
        "material": "musik" if real_paare else "synthetisch",
        "checkpoint": CHECKPOINT_SUBPATH,
        "smoke": bool(smoke),
        "loss": loss_acc,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Fertig: %s, Manifest %s", out_dir / CHECKPOINT_SUBPATH, out_dir / "manifest.json")
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="SOTA-M12 (D-K4-7): WF-V4-Warp-Head-Training")
    p.add_argument("--smoke", action="store_true", help="CPU-Smoke-Test < 3 Min (KEIN Echt-Training)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--synthetic-segments", type=int, default=4096)
    p.add_argument(
        "--real-dir",
        type=Path,
        default=None,
        help="WF-V4-Paare (Manifest aurik.wf4_warp_pairs/1 inkl. f0/conf) — Echt-Training auf Musik",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-dir", type=Path, default=Path("models/wf4_warp_head"))
    p.add_argument("--device", type=str, default="cpu", choices=("auto", "cpu", "cuda"))
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--blocks", type=int, nargs="*", default=list(DEFAULT_DILATIONS))
    args = p.parse_args()
    if args.smoke:
        args.epochs = 1
        args.steps_per_epoch = 2
    train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        synthetic_segments=args.synthetic_segments,
        seed=args.seed,
        out_dir=args.out_dir,
        device=args.device,
        hidden=args.hidden,
        blocks=tuple(args.blocks),
        smoke=args.smoke,
        real_dir=args.real_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
