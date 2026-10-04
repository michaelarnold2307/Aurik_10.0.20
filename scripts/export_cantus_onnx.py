#!/usr/bin/env python3
"""Exportiert Cantus (Flow-Matching-DiT) nach ONNX mit Paritäts-Gate.

Signatur des Inferenzgraphs (models/cantus/cantus_dit.onnx):
  x        : [1, T, 1]  float32 — degradierter Vocal (48 kHz, mono)
  t        : [1]        float32 — Flow-Zeit (0..1, Inferenz: 0.5)
  mert     : [1, F, 768] float32 — MERT-v1-330M Features (frame-level)
  pitch    : [1, F, 2]  float32 — (log2-f0-normiert, voiced_prob)
  harm     : [1, 768]   float32 — MuQ-MuLan Embedding (L2-normiert)
  use_cond : [1]        float32 — 1.0 = konditioniert, 0.0 = Null-Tokens
  → v       : [1, T, 1] float32 — Flow-Matching-Geschwindigkeitsfeld
              Inferenz: ŷ = x + (1−t)·v

Paritäts-Gate (§III.9 (copilot-instructions.md)): Torch vs. ONNX-CPU auf
strukturierten Feeds, rel ≤ 1e-3, fail-closed (Exit 2 bei Verletzung).
Deterministische Feeds (§G5 (GEBOTE.md)) — gleicher Seed ⇒ identische Prüfung.

Usage:
    python3 -B scripts/export_cantus_onnx.py [--checkpoint models/cantus/checkpoint_best.pt]
        [--preset full|tiny] [--out models/cantus/cantus_dit.onnx]
        [--allow-random-init]   # NUR für CPU-Smoke/Tests ohne Gewichte
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cantus.cantus_model import (
    CantusExportWrapper,
    create_cantus,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "models" / "cantus" / "cantus_config.json"
DEFAULT_OUT = PROJECT_ROOT / "models" / "cantus" / "cantus_dit.onnx"
DEFAULT_CKPT = PROJECT_ROOT / "models" / "cantus" / "checkpoint_best.pt"
PARITY_REL_TOL = 1e-3  # §III.9 (copilot-instructions.md)


def _load_model(preset: str, checkpoint: Path, allow_random_init: bool) -> torch.nn.Module:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["model"][preset]
    kwargs = {k: v for k, v in cfg.items() if not k.startswith("_")}
    model = create_cantus(**kwargs)
    if checkpoint.is_file():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        logger.info(
            "Checkpoint geladen: %s (Trainingsabschnitt=%s seed=%s)", checkpoint, ckpt.get("phase"), ckpt.get("seed")
        )
    elif not allow_random_init:
        raise FileNotFoundError(
            f"Kein Checkpoint unter {checkpoint} — Training ausführen oder --allow-random-init (nur Tests)"
        )
    else:
        logger.warning("Zufällige Initialisierung (—allow-random-init): Ausgabeerzeugung NUR fuer Tests/Smoke")
    model.eval()
    return model


def make_feeds(seed: int = 7, t_samples: int = 8192, f_frames: int = 40) -> dict[str, torch.Tensor]:
    """Deterministische strukturierte Feeds (§G5 (GEBOTE.md)) — Sine + Rauschen + Harmonik."""
    g = torch.Generator().manual_seed(seed)
    tt = torch.arange(t_samples, dtype=torch.float32) / 48000.0
    sine = 0.4 * torch.sin(2.0 * torch.pi * 220.0 * tt) + 0.2 * torch.sin(2.0 * torch.pi * 440.0 * tt)
    noise = 0.02 * torch.randn(t_samples, generator=g)
    x = (sine + noise).reshape(1, t_samples, 1)
    t = torch.tensor([0.5], dtype=torch.float32)
    mert = torch.randn(1, f_frames, 1024, generator=g) * 0.1  # MERT-v1-330M (gemessene Breite)
    frames = torch.linspace(0.0, 1.0, f_frames)
    pitch = torch.stack([0.6 * frames, 0.9 * torch.ones(f_frames)], dim=0).reshape(1, f_frames, 2)
    harm = torch.randn(1, 768, generator=g) * 0.1
    harm = harm / (harm.norm(dim=-1, keepdim=True) + 1e-8)  # MuQ-MuLan-Vertrag: L2-normiert
    use_cond = torch.tensor([1.0], dtype=torch.float32)
    return {"x": x, "t": t, "mert": mert, "pitch": pitch, "harm": harm, "use_cond": use_cond}


def parity_check(wrapper: torch.nn.Module, onnx_path: Path, seed: int = 7) -> dict:
    """Torch vs. ONNX-CPU auf strukturierten Feeds (§III.9 (copilot-instructions.md)).

    Gate: max. rel. Abweichung ≤ 1e-3 gegen Torch-Referenz (fail-closed).
    Zwei Feeds: strukturierter Sine-Feed + unbedingter Pfad (use_cond=0).
    """
    import onnxruntime as ort  # lokaler Import: nur für das Gate benötigt

    feeds_a = make_feeds(seed=seed)
    feeds_b = make_feeds(seed=seed + 1, t_samples=4096, f_frames=1)
    feeds_b["use_cond"] = torch.tensor([0.0])  # Null-Tokens-Pfad

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in session.get_inputs()}
    report = {"tolerance": PARITY_REL_TOL, "feeds": [], "passed": True}
    for label, feeds in (("conditioned", feeds_a), ("unconditioned", feeds_b)):
        with torch.no_grad():
            ref = wrapper(**feeds).numpy()
        ort_out = session.run(None, {k: v.numpy() for k, v in feeds.items() if k in input_names})[0]
        denom = float(np.max(np.abs(ref))) + 1e-8
        rel = float(np.max(np.abs(ort_out - ref))) / denom
        entry = {"feed": label, "rel_error": rel, "ok": bool(rel <= PARITY_REL_TOL)}
        report["feeds"].append(entry)
        report["passed"] = report["passed"] and entry["ok"]
        logger.info("Parität [%s]: rel=%.2e (%s)", label, rel, "OK" if entry["ok"] else "FAIL")
    return report


def export_model(preset: str, checkpoint: Path, out_path: Path, allow_random_init: bool = False, seed: int = 7) -> int:
    """Exportiert + prüft; Exit 0 = OK, 2 = Paritäts-/Setup-Fehler (fail-closed)."""
    torch.manual_seed(seed)  # §G5 (GEBOTE.md): deterministische Feed-Generierung
    try:
        model = _load_model(preset, checkpoint, allow_random_init)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    wrapper = CantusExportWrapper(model).eval()

    feeds = make_feeds(seed=seed)
    dynamic = {
        "x": {0: "batch", 1: "time"},
        "t": {0: "batch"},
        "mert": {0: "batch", 1: "frames"},
        "pitch": {0: "batch", 1: "frames"},
        "harm": {0: "batch"},
        "use_cond": {0: "batch"},
        "v": {0: "batch", 1: "time"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (feeds["x"], feeds["t"], feeds["mert"], feeds["pitch"], feeds["harm"], feeds["use_cond"]),
        str(out_path),
        input_names=["x", "t", "mert", "pitch", "harm", "use_cond"],
        output_names=["v"],
        dynamic_axes=dynamic,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,  # Legacy-Exporter (Repo-Präzedenz export_sgmse_onnx.py): dynamic_axes-kompatibel
    )
    logger.info("ONNX-Ausgabe erzeugt: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    report = parity_check(wrapper, out_path, seed=seed)
    report["model"] = {"preset": preset, "checkpoint": str(checkpoint), "params_m": round(model.num_params, 2)}
    report_path = out_path.with_suffix(".parity.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Paritäts-Report: %s", report_path)
    if not report["passed"]:
        logger.error("§III.9 (copilot-instructions.md) Paritäts-Gate verfehlt — Ausgabe verworfen")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--preset", choices=("full", "tiny"), default="full")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Export OHNE Gewichte — nur CPU-Smoke/Tests, kein Produktions-Modell",
    )
    args = p.parse_args(argv)
    return export_model(args.preset, args.checkpoint, args.out, args.allow_random_init, args.seed)


if __name__ == "__main__":
    sys.exit(main())
