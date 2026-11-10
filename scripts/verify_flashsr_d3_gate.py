"""scripts/verify_flashsr_d3_gate.py — D3-3-Gate: MUSDB-ΔSDR des FlashSR-Finetunes.

Der Trainer `scripts/train_flashsr_f4.py` schreibt sein eigenes Gate als
Auftrag in den Report (``gate_hint``): **„MUSDB-ΔSDR ≥ +2 dB
(Bandbreiten-Extension) NACH dem Lauf messen"**. Genau das macht dieses
Skript — und nur das. Es ist **keine** Verdrahtung und **kein** Deploy:
das Ergebnis ist ein Messwert, die Aktivierung liefe über §4.

Warum ein eigenes Skript und nicht `benchmark_bwe_candidates.py`?
Der Benchmark vergleicht die *deployten ONNX-Kandidaten* (AERO vs.
FlashSR) gegeneinander. Hier geht es um den *trainierten Torch-Checkpoint*
(`models/flashsr/best.pt`), der (noch) keinen ONNX-Export hat, und um die
Frage „schlägt der Finetune die 16-kHz-Basis?". Die Messgrößen (`_sdr`,
`_segmentwise`, `_bandlimit`, `_load_mixture`) werden aus dem Benchmark
**importiert** — eine Wahrheit für SDR und Segmentierung.

Vertrag (bewusst streng, Lehre aus D-K3-85):

- **CPU-only, fail-closed.** Ist eine GPU sichtbar, bricht das Skript mit
  Exit 2 ab, statt sie zu belegen (Kette 2 trainiert). Aufruf:
  ``CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" python3 -B scripts/verify_flashsr_d3_gate.py``
- **Diagnose gegen Mess-Artefakte.** Peilung (beste Verzögerung), Pegel
  (optimaler Gain, RMS-Verhältnis) und die Band-Aufteilung
  (Passband-/HF-SDR) werden mitgeschrieben — ein negatives ΔSDR wird erst
  dann als Qualitätsurteil gebucht, wenn diese Artefakte ausgeschlossen
  sind.
- **Zwei Varianten.** ``raw`` ist der Konsum-Vertrag (Modellausgabe, wie
  das Plugin sie weiterreicht). ``residual`` ist eine reine
  **Mess-Option**: Basis-Passband bit-nah erhalten, nur das modellierte
  Hochband addieren (konstruktives Never-worsen) — sie ist **nicht**
  verdrahtet.
- **Determinismus (§G5 (copilot-instructions.md)).** Seed 42, feste Track-Auswahl, keine Zeitquelle
  in der Entscheidung; Checkpoint- und Basis-sha256 stehen im Report.

Exit-Codes: 0 = Gate bestanden, 1 = Gate nicht bestanden (Messung gültig),
2 = Messung nicht möglich (Setup/GPU/fehlende Dateien).

Aufruf:
    CUDA_VISIBLE_DEVICES="" python3 -B scripts/verify_flashsr_d3_gate.py \
        [--tracks 3] [--seconds 20] [--mit-pretrained]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.benchmark_bwe_candidates import (
    _TARGET_SR,
    _bandlimit,
    _load_mixture,
    _sdr,
    _segmentwise,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("verify_flashsr_d3_gate")

SCHEMA = "aurik.flashsr_d3_gate/1"
MUSDB_TEST = _ROOT / "data" / "musdb18hq" / "test"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
SR_LO = 16000
SCHWELLE_DB = 2.0
GRENZE_HF = 16000.0


def _sha256(pfad: Path) -> str:
    h = hashlib.sha256()
    with pfad.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _band(x: np.ndarray, lo: float | None = None, hi: float | None = None, sr: int = _TARGET_SR) -> np.ndarray:
    """Bandbegrenzung per FFT (zero-phase, deterministisch)."""
    spektrum = np.fft.rfft(np.asarray(x, dtype=np.float64))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    if lo is not None:
        spektrum[f < lo] = 0.0
    if hi is not None:
        spektrum[f >= hi] = 0.0
    return np.fft.irfft(spektrum, len(x))


def _residual(est: np.ndarray, basis: np.ndarray, sr: int = _TARGET_SR) -> np.ndarray:
    """Basis-Passband + modelliertes Hochband (Mess-Option, nicht verdrahtet)."""
    hf = np.asarray(est, dtype=np.float64) - _band(est, hi=GRENZE_HF, sr=sr)
    return (np.asarray(basis, dtype=np.float64) + hf).astype(np.float32)


def _beste_verzoegerung(est: np.ndarray, ref: np.ndarray, max_lag: int = 240) -> int:
    """Verzögerung von ``est`` gegen ``ref`` in Samples (±``max_lag``)."""
    a = np.asarray(est[: 2 * max_lag + 24000], dtype=np.float64)
    b = np.asarray(ref[: len(a)], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    kreuz = np.correlate(a, b, mode="full")
    return int(np.argmax(kreuz)) - (len(b) - 1)


def _optimaler_gain(est: np.ndarray, ref: np.ndarray) -> float:
    nenner = float(np.dot(est, est))
    return float(np.dot(est, ref) / nenner) if nenner > 1e-20 else 1.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _gate_urteil(mean_dsdr: float, min_dsdr: float, schwelle: float = SCHWELLE_DB) -> dict:
    """Gate: Mittel ≥ Schwelle UND kein Segment schlechter als die Basis."""
    bestanden = bool(mean_dsdr >= schwelle and min_dsdr >= 0.0)
    if mean_dsdr < schwelle:
        grund = f"Mittel {mean_dsdr:+.2f} dB < Schwelle {schwelle:+.1f} dB"
    elif min_dsdr < 0.0:
        grund = f"Never-worsen verletzt (min {min_dsdr:+.2f} dB < 0)"
    else:
        grund = "Schwelle und Never-worsen erfüllt"
    return {"bestanden": bestanden, "schwelle_db": schwelle, "grund": grund}


def _waehle_tracks(anzahl: int, seed: int) -> list[Path]:
    tracks = sorted(p for p in MUSDB_TEST.glob("*") if p.is_dir())
    if not tracks:
        return []
    rng = np.random.default_rng(seed)
    gewaehlt = [tracks[i] for i in rng.choice(len(tracks), size=min(anzahl, len(tracks)), replace=False)]
    return sorted(gewaehlt)


def _cpu_guard() -> str | None:
    """None = CPU-only messbar; sonst Begründung (fail-closed)."""
    import torch

    if torch.cuda.is_available():
        return (
            "GPU ist sichtbar — dieser Lauf ist CPU-only (Kette 2 trainiert). "
            "Starte mit CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES=''."
        )
    return None


def _lade_modell(basis_checkpoint: Path, finetune: Path | None):
    import torch

    sys.path.insert(0, str(_ROOT / "models" / "flashsr"))
    from FastAudioSR import FASR

    fasr = FASR(str(basis_checkpoint))
    if finetune is not None:
        zustand = torch.load(str(finetune), map_location="cpu")
        fasr.model.load_state_dict(zustand["model"])
        logger.info("Finetune geladen: %s (Epoche %s)", finetune, zustand.get("epoch"))
    fasr.model = fasr.model.to("cpu")
    fasr.device = torch.device("cpu")
    fasr.model.eval()
    return fasr.model


def _inferenz(modell, ref: np.ndarray) -> np.ndarray:
    """48 kHz → 16 kHz (librosa, wie im Trainer) → Modell → 48 kHz-Ausgabe."""
    import librosa
    import torch

    lo = librosa.resample(np.asarray(ref, dtype=np.float32), orig_sr=_TARGET_SR, target_sr=SR_LO)
    with torch.no_grad():
        y = modell(torch.from_numpy(lo).unsqueeze(0).unsqueeze(1)).squeeze(1).reshape(-1).numpy()
    n = min(len(y), len(ref))
    if len(y) < len(ref):
        y = np.pad(y, (0, len(ref) - len(y)))
    return np.asarray(y[:n], dtype=np.float32)


def _messe(ref: np.ndarray, est: np.ndarray, basis: np.ndarray) -> dict:
    """ΔSDR je 1-s-Segment + Diagnose (Peilung, Pegel, Bänder)."""
    n = min(len(ref), len(est), len(basis))
    ref, est, basis = ref[:n], est[:n], basis[:n]
    d_segmente = [a - b for a, b in zip(_segmentwise(_sdr, est, ref), _segmentwise(_sdr, basis, ref))]
    return {
        "mean_dSDR": float(np.mean(d_segmente)),
        "min_dSDR": float(np.min(d_segmente)),
        "max_dSDR": float(np.max(d_segmente)),
        "n_segmente": len(d_segmente),
        "sdr_est_db": float(_sdr(est, ref)),
        "sdr_basis_db": float(_sdr(basis, ref)),
        "passband_sdr_est_db": float(_sdr(_band(est, hi=GRENZE_HF), _band(ref, hi=GRENZE_HF))),
        "passband_sdr_basis_db": float(_sdr(_band(basis, hi=GRENZE_HF), _band(ref, hi=GRENZE_HF))),
        "hf_sdr_est_db": float(_sdr(_band(est, lo=GRENZE_HF), _band(ref, lo=GRENZE_HF))),
        "rms_est": _rms(est),
        "rms_ref": _rms(ref),
        "rms_verhaeltnis": _rms(est) / max(_rms(ref), 1e-12),
        "optimale_verstaerkung": _optimaler_gain(est, ref),
        "beste_verzoegerung_samples": _beste_verzoegerung(est, ref),
        "gate": _gate_urteil(float(np.mean(d_segmente)), float(np.min(d_segmente))),
    }


def _zielpfad(ziel: Path, inhalt: str) -> Path:
    """Nie einen vorhandenen Beleg ueberschreiben (Lehre D-K3-85: Belege sind Beweise).

    Eine identische Wiederholung ist idempotent (derselbe Pfad), ein abweichender
    Lauf bekommt einen numerischen Nebennamen ``_2``, ``_3``, ...
    """
    if not ziel.exists():
        return ziel
    if ziel.read_text(encoding="utf-8") == inhalt:
        return ziel
    for nummer in range(2, 100):
        kandidat = ziel.with_name(f"{ziel.stem}_{nummer}{ziel.suffix}")
        if not kandidat.exists():
            return kandidat
        if kandidat.read_text(encoding="utf-8") == inhalt:
            return kandidat
    raise RuntimeError(f"kein freier Beleg-Pfad neben {ziel}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="D3-3-Gate: MUSDB-ΔSDR des FlashSR-Finetunes (CPU, §G5 (copilot-instructions.md))"
    )
    ap.add_argument("--checkpoint", default="models/flashsr/best.pt", help="Trainer-Checkpoint (state dict)")
    ap.add_argument(
        "--basis-checkpoint",
        default="models/flashsr/models/upsampler.pth",
        help="Architektur-/Basis-Gewichte zum Aufbau (Trainer-Default)",
    )
    ap.add_argument("--tracks", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mit-pretrained", action="store_true", help="Vortrainings-Modell als Referenzzeile mitmessen")
    ap.add_argument(
        "--out", default="", help="Report-Pfad (Default: docs/reports/current/<datum>_flashsr_d3_gate.json)"
    )
    args = ap.parse_args(argv)

    grund = _cpu_guard()
    if grund:
        print(f"ABBRUCH: {grund}")
        return 2

    finetune = (_ROOT / args.checkpoint).resolve()
    basis_ckpt = (_ROOT / args.basis_checkpoint).resolve()
    for pfad in (finetune, basis_ckpt):
        if not pfad.is_file():
            print(f"ABBRUCH: fehlt: {pfad}")
            return 2

    tracks = _waehle_tracks(args.tracks, args.seed)
    if not tracks:
        print(f"ABBRUCH: MUSDB18HQ-Testset fehlt: {MUSDB_TEST}")
        return 2

    logger.info("FlashSR-D3-Gate — %d Track(s) × %d s, Seed %d, CPU", len(tracks), args.seconds, args.seed)
    modell = _lade_modell(basis_ckpt, finetune)
    vorher = _lade_modell(basis_ckpt, None) if args.mit_pretrained else None

    ergebnis: dict = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": args.seed,
        "sekunden": args.seconds,
        "schwelle_db": SCHWELLE_DB,
        "checkpoint": {"pfad": str(finetune.relative_to(_ROOT)), "sha256": _sha256(finetune)},
        "basis_checkpoint": {"pfad": str(basis_ckpt.relative_to(_ROOT)), "sha256": _sha256(basis_ckpt)},
        "tracks": [],
        "varianten": {},
    }

    for track in tracks:
        ref = _load_mixture(track, args.seconds, args.seed)
        basis = _bandlimit(ref, SR_LO)
        est_raw = _inferenz(modell, ref)
        est_res = _residual(est_raw, basis)
        zeile = {
            "track": track.name,
            "raw": _messe(ref, est_raw, basis),
            "residual": _messe(ref, est_res, basis),
        }
        if vorher is not None:
            zeile["pretrained"] = _messe(ref, _inferenz(vorher, ref), basis)
        ergebnis["tracks"].append(zeile)
        for variante in ("raw", "residual", "pretrained"):
            if variante in zeile:
                m = zeile[variante]
                print(
                    f"  {track.name[:34]:34s} {variante:10s} meanΔSDR={m['mean_dSDR']:+.2f} "
                    f"minΔSDR={m['min_dSDR']:+.2f} Passband={m['passband_sdr_est_db']:.2f} "
                    f"HF={m['hf_sdr_est_db']:+.2f} Lag={m['beste_verzoegerung_samples']:+d} "
                    f"Gain={m['optimale_verstaerkung']:.2f}"
                )

    for variante in ("raw", "residual", "pretrained"):
        werte = [t[variante] for t in ergebnis["tracks"] if variante in t]
        if not werte:
            continue
        mean = float(np.mean([w["mean_dSDR"] for w in werte]))
        minimum = float(np.min([w["min_dSDR"] for w in werte]))
        ergebnis["varianten"][variante] = {
            "mean_dSDR_ueber_tracks": mean,
            "min_dSDR_ueber_tracks": minimum,
            "gate": _gate_urteil(mean, minimum),
        }

    primaer = ergebnis["varianten"]["raw"]
    ziel = Path(args.out) if args.out else REPORT_DIR / f"{date.today().isoformat()}_flashsr_d3_gate.json"
    if not ziel.is_absolute():
        ziel = _ROOT / ziel
    ziel.parent.mkdir(parents=True, exist_ok=True)
    inhalt = json.dumps(ergebnis, indent=2, ensure_ascii=False) + "\n"
    ziel = _zielpfad(ziel, inhalt)
    ziel.write_text(inhalt, encoding="utf-8")

    print(
        f"\nVariante raw      : meanΔSDR={primaer['mean_dSDR_ueber_tracks']:+.2f} dB, "
        f"minΔSDR={primaer['min_dSDR_ueber_tracks']:+.2f} dB ⇒ "
        f"{'BESTANDEN' if primaer['gate']['bestanden'] else 'NICHT BESTANDEN'} "
        f"({primaer['gate']['grund']})"
    )
    if "residual" in ergebnis["varianten"]:
        res = ergebnis["varianten"]["residual"]
        print(
            f"Variante residual : meanΔSDR={res['mean_dSDR_ueber_tracks']:+.2f} dB, "
            f"minΔSDR={res['min_dSDR_ueber_tracks']:+.2f} dB ⇒ "
            f"{'BESTANDEN' if res['gate']['bestanden'] else 'NICHT BESTANDEN'} "
            f"(Mess-Option, NICHT verdrahtet)"
        )
    if "pretrained" in ergebnis["varianten"]:
        vor = ergebnis["varianten"]["pretrained"]
        print(f"Referenz pretrained: meanΔSDR={vor['mean_dSDR_ueber_tracks']:+.2f} dB")
    print(f"Report: {ziel.relative_to(_ROOT)}")
    return 0 if primaer["gate"]["bestanden"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
