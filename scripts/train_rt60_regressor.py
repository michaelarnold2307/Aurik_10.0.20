#!/usr/bin/env python3
"""SOTA-M15 (KP-10): Training des RT60-Regressors auf RIR-Paaren.

Ground-Truth: bekannte Decay-Kurven der RIR-Simulationen (`rir_sim_*.wav`,
DFN-Assets). Trainings-Paar = (RIR-Waveform, GT-RT60) — bei den lokalen
Simulationen ist das GT-RT60 im Dateinamen kodiert
(`rir_sim_<id>_w<W>_l<L>_h<H>_rt60_<RT60>.wav`).

Datenlage (ehrlich, Muster D-K2-5)
----------------------------------
Das Repo enthält lokal **genau 1** RIR-Simulation
(`models/deepfilternet_v3_ii/assets/rir_sim_1001_w11.7_l2.6_h2.5_rt60_0.7919.wav`).
Echte RIR-Aufnahmen mit gemessenem RT60 fehlen komplett. Ein belastbares
Training auf echtem Material ist damit **nicht möglich** — die Datenlücke wird
hier dokumentiert, nicht durch synthetische Daten kaschiert
(§V6 (copilot-instructions.md): kein stiller Fallback auf erfundene GT).

Was dieses Skript kann:
- **Inventar**: alle `rir_sim_*.wav` im Repo inventarisieren (Anzahl, SR,
  Channels, Duration, GT-RT60 aus Dateiname parsbar?, GT-Decay direkt
  ableitbar?). Manifest als JSON.
- **Synth-RIR-Smoke-Test** (`--smoke`): deterministische synthetische
  RIRs (exponentieller Abkling-Noise, GT-RT60 bekannt) — CPU, < 3 Min,
  validiert den vollen Trainings-Pfad (Feature-Extraktion → Ridge-Fit →
  Checkpoint) ohne GPU. KEIN Echt-Training (GPU belegt).
- **Checkpoint**: `models/rt60/rt60_regressor_v1.npz` (weights, bias,
  manifest, seed). Deterministisch (§G5 (copilot-instructions.md)): Seed pro
  Session, kein `time.time()` in der Logik.

Bestehende Artefakte: `scripts/train_basicpitch.py`,
`scripts/train_aspade_declipper.py` (Muster für Seed/Checkpoint/Resume).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.dsp.rt60_regressor import (
    DEFAULT_SEED,
    N_FEATURES,
    RT60_MAX_S,
    RT60_MIN_S,
    _decay_features,
    _heuristic_weights,
    schroeder_edc_db,
)

logger = logging.getLogger(__name__)

# GT-RT60-Pattern im Dateinamen: rir_sim_<id>_w<W>_l<L>_h<H>_rt60_<RT60>.wav
_RIR_NAME_PATTERN = re.compile(
    r"rir_sim_(?P<id>\d+)_w(?P<w>[\d.]+)_l(?P<l>[\d.]+)_h(?P<h>[\d.]+)_rt60_(?P<rt60>[\d.]+)\.wav$",
    re.IGNORECASE,
)

MANIFEST_PATH = _PROJECT_ROOT / "models" / "rt60" / "rir_manifest.json"
CHECKPOINT_PATH = _PROJECT_ROOT / "models" / "rt60" / "rt60_regressor_v1.npz"


# ── Daten-Inventar ───────────────────────────────────────────────────────────


@dataclass
class RirEntry:
    """Ein RIR-Datensatz-Eintrag im Manifest."""

    path: str
    filename: str
    sample_rate: int | None = None
    channels: int | None = None
    n_frames: int | None = None
    duration_s: float | None = None
    gt_rt60_from_name: float | None = None
    gt_decay_derivable: bool = False
    parse_error: str | None = None
    notes: list[str] = field(default_factory=list)


def _parse_gt_from_name(filename: str) -> float | None:
    """GT-RT60 aus dem Dateinamen parsen (DFN-Simulations-Konvention)."""
    m = _RIR_NAME_PATTERN.search(filename)
    if not m:
        return None
    try:
        val = float(m.group("rt60"))
    except ValueError:
        return None
    if not np.isfinite(val):
        return None
    return val


def _inspect_wav(path: Path) -> RirEntry:
    """Eine WAV-Datei inventarisieren (SR, Channels, Duration, GT-Decay)."""
    entry = RirEntry(path=str(path), filename=path.name)
    entry.gt_rt60_from_name = _parse_gt_from_name(path.name)
    try:
        with wave.open(str(path), "rb") as w:
            entry.sample_rate = w.getframerate()
            entry.channels = w.getnchannels()
            entry.n_frames = w.getnframes()
            if entry.sample_rate and entry.n_frames:
                entry.duration_s = entry.n_frames / float(entry.sample_rate)
        # GT-Decay ist ableitbar, wenn GT-RT60 aus dem Namen bekannt ist
        # (Dann ist die Decay-Kurve der Simulation die Referenz).
        entry.gt_decay_derivable = entry.gt_rt60_from_name is not None
        if entry.gt_rt60_from_name is None:
            entry.notes.append("GT-RT60 nicht im Dateinamen kodiert — kein GT-Decay ableitbar")
    except Exception as exc:  # pylint: disable=broad-except
        entry.parse_error = str(exc)
        entry.notes.append(f"WAV nicht lesbar: {exc}")
    return entry


def inventory_rirs(root: Path) -> list[RirEntry]:
    """Alle `rir_sim_*.wav` unter `root` inventarisieren (rekursiv)."""
    entries: list[RirEntry] = []
    for path in sorted(root.rglob("rir_sim_*.wav")):
        if ".venv" in path.parts or "node_modules" in path.parts:
            continue
        entries.append(_inspect_wav(path))
    return entries


def write_manifest(entries: list[RirEntry], out_path: Path) -> None:
    """Manifest als JSON schreiben (Determinismus: sortierte Pfade)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "scripts/train_rt60_regressor.py",
        "sota_measure": "SOTA-M15 (KP-10)",
        "n_rirs": len(entries),
        "n_with_gt_rt60": sum(1 for e in entries if e.gt_rt60_from_name is not None),
        "n_gt_decay_derivable": sum(1 for e in entries if e.gt_decay_derivable),
        "entries": [asdict(e) for e in entries],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifest geschrieben: %s (%d Einträge)", out_path, len(entries))


# ── Datensatz (RIR-Paare + synthetische Paare) ───────────────────────────────


def _synth_rir(rt60: float, sr: int, seed: int, dur_s: float = 1.5) -> np.ndarray:
    """Deterministisches synthetisches RIR: exponentiell abklingender Noise.

    GT-RT60 = `rt60` (Amplituden-Envelope exp(−6.9078·t/rt60) ⇒ Energie −60 dB
    bei rt60). Wird nur im Smoke-Test und für Unit-Tests verwendet — **kein**
    Ersatz für echte RIR-Daten (§V6 (copilot-instructions.md)).
    """
    rng = np.random.default_rng(seed)
    n = int(sr * dur_s)
    t = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-6.9078 * t / rt60)
    x = rng.standard_normal(n) * env
    return x.astype(np.float32)


def build_synth_pairs(n_samples: int, sr: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetische (Feature-Matrix, GT-RT60-Vektor) für den Smoke-Test.

    Deterministisch (§G5 (copilot-instructions.md)): Seed pro Session, kein
    `np.random` ohne Seed. GT-RT60 logarithmisch zwischen 0.15 s und 3.0 s.
    """
    rng = np.random.default_rng(seed)
    rt60_targets = np.exp(rng.uniform(np.log(0.15), np.log(3.0), size=n_samples))
    feats = np.zeros((n_samples, N_FEATURES), dtype=np.float64)
    for i, rt in enumerate(rt60_targets):
        x = _synth_rir(float(rt), sr, seed=seed + i, dur_s=max(1.2, float(rt) * 1.5))
        edc = schroeder_edc_db(x)
        feats[i] = _decay_features(edc, sr)
    return feats, rt60_targets.astype(np.float64)


# ── Training (Ridge-Regression, deterministisch) ─────────────────────────────


def train_ridge(
    feats: np.ndarray,
    targets: np.ndarray,
    *,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """Deterministisches Ridge-Fitting (Closed-Form, kein SGD, kein Zufall).

    Args:
        feats:   (N, N_FEATURES) Feature-Matrix.
        targets: (N,) GT-RT60 in s.
        l2:      L2-Regularisierung (stabilisiert kleine Datensätze).

    Returns:
        (weights, bias) — `weights.shape == (N_FEATURES,)`.
    """
    if feats.ndim != 2 or feats.shape[1] != N_FEATURES:
        raise ValueError(f"feats muss (N, {N_FEATURES}) sein, got {feats.shape}")
    if targets.ndim != 1 or targets.shape[0] != feats.shape[0]:
        raise ValueError(f"targets muss (N,) mit N={feats.shape[0]} sein, got {targets.shape}")

    # Design-Matrix mit Bias-Spalte
    x = np.hstack([feats, np.ones((feats.shape[0], 1), dtype=np.float64)])
    # Ridge: (XᵀX + λI)⁻¹ Xᵀy — deterministisch, keine Iteration.
    xtx = x.T @ x + l2 * np.eye(x.shape[1], dtype=np.float64)
    xty = x.T @ targets
    try:
        sol = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        logger.warning(
            "Ridge-Fit: Singuläre Matrix — Fallback auf deterministische Heuristik-Gewichte "
            "(§V6 (copilot-instructions.md))"
        )
        return _heuristic_weights()
    weights = sol[:N_FEATURES]
    bias = float(sol[N_FEATURES])
    return weights, bias


# ── Checkpoint ────────────────────────────────────────────────────────────────


def save_checkpoint(
    out_path: Path,
    weights: np.ndarray,
    bias: float,
    *,
    seed: int,
    manifest: dict,
) -> None:
    """Checkpoint speichern (npz, allow_pickle=False — deterministisch)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        weights=np.asarray(weights, dtype=np.float64),
        bias=np.float64(bias),
        seed=np.int64(seed),
        manifest_json=json.dumps(manifest, ensure_ascii=False),
    )
    logger.info("Checkpoint gespeichert: %s (seed=%d)", out_path, seed)


# ── Hauptprogramm ────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="SOTA-M15 (KP-10): RT60-Regressor-Training auf RIR-Paaren")
    p.add_argument("--smoke", action="store_true", help="Synth-RIR-Smoke-Test auf CPU (< 3 Min)")
    p.add_argument("--n-synth", type=int, default=64, help="Anzahl synthetischer Paare (Smoke)")
    p.add_argument("--sr", type=int, default=48000, help="Abtastrate der synthetischen RIRs")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Determinismus §G5 (copilot-instructions.md)")
    p.add_argument("--l2", type=float, default=1e-3, help="L2-Regularisierung (Ridge)")
    p.add_argument("--out", type=Path, default=CHECKPOINT_PATH, help="Checkpoint-Pfad")
    p.add_argument("--manifest-out", type=Path, default=MANIFEST_PATH, help="Manifest-Pfad")
    p.add_argument("--inventory-only", action="store_true", help="Nur Inventar, kein Training")
    args = p.parse_args()

    # 1) Inventar der lokalen RIR-Simulationen
    entries = inventory_rirs(_PROJECT_ROOT)
    n_gt = sum(1 for e in entries if e.gt_rt60_from_name is not None)
    logger.info(
        "Daten-Inventar: %d RIR-Simulationen gefunden, davon %d mit GT-RT60 aus Dateiname",
        len(entries),
        n_gt,
    )
    if n_gt == 0:
        logger.warning(
            "Datenlücke (D-K2-5-Muster): keine RIR-Simulationen mit GT-RT60 im Dateinamen — "
            "Echt-Training nicht möglich. Nur Synth-Smoke-Test. (§V6 (copilot-instructions.md))"
        )
    write_manifest(entries, args.manifest_out)

    if args.inventory_only:
        return

    # 2) Training
    if args.smoke:
        logger.info(
            "Smoke-Test: %d synthetische RIR-Paare (SR=%d, seed=%d) — KEIN Echt-Training (GPU belegt)",
            args.n_synth,
            args.sr,
            args.seed,
        )
        feats, targets = build_synth_pairs(args.n_synth, args.sr, args.seed)
    else:
        if n_gt == 0:
            logger.warning(
                "Kein --smoke und keine GT-RIRs — abgebrochen (§V6 (copilot-instructions.md)). "
                "Nutze --smoke für den deterministischen CPU-Smoke-Test."
            )
            return
        # Echt-Training auf lokalen RIRs (hier: nur 1 Datei — dokumentierte Datenlücke)
        feats_list = []
        targets_list = []
        for e in entries:
            if e.gt_rt60_from_name is None:
                continue
            try:
                with wave.open(e.path, "rb") as w:
                    sr = w.getframerate()
                    raw = w.readframes(w.getnframes())
                x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                edc = schroeder_edc_db(x)
                feats_list.append(_decay_features(edc, sr))
                targets_list.append(e.gt_rt60_from_name)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("RIR %s nicht lesbar (%s) — übersprungen (§V6 (copilot-instructions.md))", e.path, exc)
        if not feats_list:
            logger.warning("Keine lesbaren GT-RIRs — abgebrochen (§V6 (copilot-instructions.md))")
            return
        feats = np.stack(feats_list)
        targets = np.asarray(targets_list, dtype=np.float64)
        logger.warning(
            "Echt-Training auf nur %d RIR(s) — statistisch nicht belastbar (Datenlücke D-K2-5-Muster). "
            "Checkpoint nur als Platzhalter markieren, kein Flag-Flip ohne Gate.",
            len(targets),
        )

    weights, bias = train_ridge(feats, targets, l2=args.l2)

    # Plausibilitäts-Check (Bestimmtheitsmaß, deterministisch)
    pred = feats @ weights + bias
    ss_res = float(np.sum((pred - targets) ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    logger.info("R² auf dem Trainingssatz: %.4f (Bestimmtheitsmaß, informativ)", r2)

    manifest = {
        "sota_measure": "SOTA-M15 (KP-10)",
        "n_samples": int(feats.shape[0]),
        "n_features": N_FEATURES,
        "l2": args.l2,
        "seed": args.seed,
        "r2_train": r2,
        "rt60_min_s": RT60_MIN_S,
        "rt60_max_s": RT60_MAX_S,
        "data_note": (
            "Datenlücke dokumentiert (Muster D-K2-5): nur 1 lokale RIR-Simulation im Repo; "
            "keine echten RIR-Aufnahmen. Smoke-Training auf synthetischen RIRs (§V6 (copilot-instructions.md))."
        ),
    }
    save_checkpoint(args.out, weights, bias, seed=args.seed, manifest=manifest)


if __name__ == "__main__":
    main()
