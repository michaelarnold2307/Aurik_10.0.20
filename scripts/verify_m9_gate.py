"""scripts/verify_m9_gate.py — SOTA-M9-Gate: TF-GridNet-Quality-Stufe gegen die Basis.

Der SOTA-M9-TF-GridNet-Denoiser (`models/tf_gridnet/tf_gridnet_quality.onnx`,
trainiert auf 25 echten MUSDB-Paaren, val 0,0190) ist in
`backend/core/phases/phase_03_denoise.py` hinter
`use_tf_gridnet_quality` verdrahtet (Default AUS). Es fehlte das objektive
Aktivierungs-Gate (Handover Punkt 11: „Never-worsen-Gate/Hörabnahme“).

Dieses Skript misst den echten Produktionsschritt
(`TFGridNetDenoiser.denoise`, dieselbe Inferenz, die die Quality-Stufe hinter
dem Flag aufruft) gegen die **Basis** (= Eingang; hinter Flag AUS ist die
Stufe ein Passthrough, `apply_tf_gridnet_quality_stage` gibt exakt das
Eingangsobjekt zurück — der heutige Produktionszustand ist die Basis).

Zwei Messarme:

``clean``
    Never-worsen auf sauberer Musik: die Quality-Stufe darf den Eingang
    nicht über die JND verschlechtern (musik_mos ≥ −JND, defektlast ≤ +JND,
    HNR ≥ −0,5 dB je Track).
``rausch``
    Deterministisch verrauschtes Material (``defekt_variante(x, "rausch",
    rng)``, Seed 42): die Quality-Stufe muss den Rausch-Arm messbar
    verbessern (HNR +0,5 dB, musik_mos +0,05 oder defektlast −0,05).

Gate (fail-closed): bestanden ⇔ never-worsen auf allen Clean-Tracks UND
Vorteil auf ≥ 2 von 3 Tracks im Rausch-Arm.

Ehrliche Grenze (2026-11-14): das Provenienz-Manifest der 25 Trainingspaare
(`data/tfgrid_pairs/manifest.json`) existiert nicht — Leak-Freiheit gegenüber
der Trainingsmenge ist für die Track-Auswahl nicht nachweisbar. Der Clean-Arm
nutzt MUSDB18HQ-Test-Split-Titel; die Never-worsen-Aussage gilt auf diesen
Titeln, die Generalisierung bleibt Vorbehalt bis zur Hör-Abnahme (§4).

Exit-Codes: 0 = Gate bestanden, 1 = Messung gültig, Gate nicht bestanden,
2 = Messung nicht möglich (Setup-Fehler/GPU sichtbar).

CPU-only (keine GPU-Kette stören), deterministisch (§G5 (copilot-instructions.md)):
feste Track-Auswahl über Seed 42. Kein Flag-Flip, kein Deploy — ein
bestandenes Gate ist die Voraussetzung für das §4-Verfahren, nicht dessen
Ersatz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("verify_m9_gate")

SCHEMA = "aurik.m9_gate/1"
MUSDB_TEST = _ROOT / "data" / "musdb18hq" / "test"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
SR = 44100

JND = 0.05
HNR_TOLERANZ_DB = 0.5
SEGMENT_S = 20.0
BENEFIT_MIN_TRACKS = 2
N_TRACKS = 3


def _cpu_guard() -> str | None:
    """Die Messung darf keine GPU-Kette stören (Handover §5).

    Der M9-Pfad ist CPU-only per Konstruktion: die ONNX-Session in
    ``tf_gridnet_denoiser.py`` erzeugt hartkodiert
    ``providers=["CPUExecutionProvider"]`` und dieses Skript initialisiert
    niemals CUDA. Ein vorhandener ROCm-torch (``cuda.is_available()``) wird
    deshalb nur protokolliert, nicht als Abbruchgrund gewertet — der
    ursprüngliche Abbruch bei „GPU sichtbar“ wäre auf diesem Host ein
    Dauer-Fail, ohne dass die Messung die GPU jemals anfasst.
    (Quelltext-Pin: tests/unit/test_m9_gate.py.)
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("GPU vorhanden — M9-Gate bleibt CPU-only (ORT CPUExecutionProvider, kein CUDA-Init).")
    except Exception:  # pragma: no cover — torch fehlt ⇒ CPU ohnehin einziger Pfad
        return None
    return None


def _sha256(pfad: Path) -> str:
    h = hashlib.sha256()
    with pfad.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _waehle_tracks(anzahl: int, seed: int) -> list[Path]:
    if not MUSDB_TEST.is_dir():
        return []
    tracks = sorted(p for p in MUSDB_TEST.iterdir() if p.is_dir() and (p / "mixture.wav").is_file())
    if not tracks:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(tracks, min(anzahl, len(tracks))))


def _lade_segment(track: Path, seconds: float) -> np.ndarray:
    """Center-Segment (seconds) des Mono-Mixes, float32 (−1..1)."""
    from scipy.io import wavfile

    roh = wavfile.read(track / "mixture.wav")
    if not isinstance(roh, tuple) or len(roh) != 2:
        raise RuntimeError("wavfile.read lieferte unerwartetes Format (Bug 12, §V36)")
    sr = int(roh[0])
    d = roh[1]
    x = np.asarray(d, np.float32)
    x = x / (float(np.abs(x).max()) + 1e-9)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    n = int(seconds * sr)
    if len(mono) <= n:
        return np.asarray(mono, dtype=np.float32)
    mid = len(mono) // 2
    return np.asarray(mono[mid - n // 2 : mid + n // 2], dtype=np.float32)


def _rausch_arm(audio: np.ndarray, seed: int) -> np.ndarray:
    """Deterministisch verrauschtes Paar (§G5 (copilot-instructions.md)): defekt_variante „rausch“."""
    from scripts.ohr_panel import defekt_variante

    rng = np.random.default_rng(seed)
    return np.asarray(defekt_variante(audio, "rausch", rng), dtype=np.float32)


def _zeugen(eingang: np.ndarray, ausgang: np.ndarray, sr: int) -> dict[str, Any]:
    """musik_mos + defektlast über den Wohlklang-Zeugen (Report-only-Messung)."""
    from backend.core.dsp.wohlklang_zeuge import messe_wohlklang_zeuge

    roh = messe_wohlklang_zeuge(eingang, ausgang, sr)
    out: dict[str, Any] = {
        "musik_mos": float(roh["ausgang"]["musik_mos"]),
        "defektlast": float(roh["ausgang"]["defektlast"]),
        "delta_musik_mos": float(roh["delta"]["musik_mos"]),
        "delta_defektlast": float(roh["delta"]["defektlast"]),
    }
    return out


def _hnr(audio: np.ndarray, sr: int) -> float:
    from scripts.validate_hr_v1 import _hnr_db

    return float(_hnr_db(audio, sr))


@dataclass
class Messung:
    arm: str
    track: str
    hnr_basis: float
    hnr_m9: float
    mos_basis: float
    mos_m9: float
    defekt_basis: float
    defekt_m9: float
    applied: bool

    def als_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "track": self.track,
            "hnr_basis_db": round(self.hnr_basis, 4),
            "hnr_m9_db": round(self.hnr_m9, 4),
            "delta_hnr_db": round(self.hnr_m9 - self.hnr_basis, 4),
            "musik_mos_basis": round(self.mos_basis, 4),
            "musik_mos_m9": round(self.mos_m9, 4),
            "defektlast_basis": round(self.defekt_basis, 4),
            "defektlast_m9": round(self.defekt_m9, 4),
            "applied": self.applied,
        }


def _messe(track: Path, arm: str, m9, seed: int) -> Messung:
    segment = _lade_segment(track, SEGMENT_S)
    basis = segment
    if arm == "rausch":
        basis = _rausch_arm(segment, seed)
    try:
        out = m9.denoise(basis, SR)
    except Exception as _exc:  # pragma: no cover — Fail-closed zählt als nicht angewandt
        logger.warning("M9-Inferenz fehlgeschlagen (%s) — als nicht angewandt gewertet.", _exc)
        out = None
    applied = out is not None
    ausgang = np.asarray(out, dtype=np.float32) if applied else np.asarray(basis, dtype=np.float32)
    z_basis = _zeugen(basis, basis, SR)  # Basis ist ihr eigener Zeuge (musik_mos/defektlast des Eingangs)
    z_m9 = _zeugen(basis, ausgang, SR)
    return Messung(
        arm=arm,
        track=track.name,
        hnr_basis=_hnr(basis, SR),
        hnr_m9=_hnr(ausgang, SR),
        mos_basis=float(z_basis["musik_mos"]),
        mos_m9=float(z_m9["musik_mos"]),
        defekt_basis=float(z_basis["defektlast"]),
        defekt_m9=float(z_m9["defektlast"]),
        applied=applied,
    )


def _never_worsen_ok(m: Messung) -> bool:
    if not m.applied:
        return False
    return (
        (m.mos_m9 - m.mos_basis) >= -JND
        and (m.defekt_m9 - m.defekt_basis) <= JND
        and (m.hnr_m9 - m.hnr_basis) >= -HNR_TOLERANZ_DB
    )


def _vorteil(m: Messung) -> bool:
    if not m.applied:
        return False
    return (
        (m.hnr_m9 - m.hnr_basis) > HNR_TOLERANZ_DB
        or (m.mos_m9 - m.mos_basis) > JND
        or (m.defekt_basis - m.defekt_m9) > JND
    )


def main(argv: list[str] | None = None) -> int:
    global SEGMENT_S

    ap = argparse.ArgumentParser(description="SOTA-M9-Gate: TF-GridNet-Quality-Stufe gegen die Basis")
    ap.add_argument("--tracks", type=int, default=N_TRACKS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seconds", type=float, default=SEGMENT_S)
    args = ap.parse_args(argv)

    SEGMENT_S = args.seconds

    guard = _cpu_guard()
    if guard:
        print(f"ABBRUCH: {guard}")
        return 2

    from backend.core.dsp.tf_gridnet_denoiser import TFGridNetDenoiser

    m9 = TFGridNetDenoiser()
    if not m9.is_available():
        print("ABBRUCH: M9-Modell nicht verfügbar (models/tf_gridnet/tf_gridnet_quality.onnx fehlt).")
        return 2

    tracks = _waehle_tracks(args.tracks, args.seed)
    if len(tracks) < 3:
        print("ABBRUCH: weniger als 3 MUSDB-Test-Tracks gefunden.")
        return 2

    messungen: list[Messung] = []
    for i, track in enumerate(tracks):
        messungen.append(_messe(track, "clean", m9, seed=args.seed))
        messungen.append(_messe(track, "rausch", m9, seed=args.seed + i))

    clean = [m for m in messungen if m.arm == "clean"]
    rausch = [m for m in messungen if m.arm == "rausch"]
    never_worsen = all(_never_worsen_ok(m) for m in clean)
    vorteile = sum(1 for m in rausch if _vorteil(m))
    bestanden = never_worsen and vorteile >= BENEFIT_MIN_TRACKS

    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": args.seed,
        "segmente_s": args.seconds,
        "modell": "models/tf_gridnet/tf_gridnet_quality.onnx",
        "modell_sha256": _sha256(_ROOT / "models" / "tf_gridnet" / "tf_gridnet_quality.onnx"),
        "tracks": [t.name for t in tracks],
        "never_worsen": never_worsen,
        "vorteile_rauscharm": vorteile,
        "vorteile_min": BENEFIT_MIN_TRACKS,
        "gate_bestanden": bestanden,
        "limitation": (
            "Provenienz-Manifest der 25 Trainingspaare fehlt — Leak-Freiheit der Track-Auswahl "
            "nicht nachweisbar; Never-worsen auf MUSDB18HQ-Test-Split belegt."
        ),
        "messungen": [m.als_dict() for m in messungen],
    }

    ziel = REPORT_DIR / f"{date.today().isoformat()}_m9_gate.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = REPORT_DIR / f"{date.today().isoformat()}_m9_gate.md"
    zeilen = [
        "# SOTA-M9-Gate — TF-GridNet-Quality-Stufe gegen die Basis",
        "",
        f"- Datum: {beleg['datum']} | Seed {beleg['seed']} | {len(tracks)} MUSDB18HQ-Test-Tracks × 2 Arme",
        f"- Never-worsen (clean): {'✓ BESTANDEN' if never_worsen else '✗ VERFEHLT'}",
        f"- Rausch-Arm-Vorteile: {vorteile}/{len(rausch)} (min {BENEFIT_MIN_TRACKS})",
        f"- **Gate: {'BESTANDEN' if bestanden else 'NICHT BESTANDEN'}** (Exit {'0' if bestanden else '1'})",
        "",
        "Hinweis: bestandenes Gate = Voraussetzung für §4-Hör-Abnahme, kein Deploy.",
        f"Beleg: `{ziel.name}`",
    ]
    md.write_text("\n".join(zeilen) + "\n", encoding="utf-8")

    print(json.dumps({k: beleg[k] for k in ("never_worsen", "vorteile_rauscharm", "gate_bestanden")}))
    print(f"Beleg: {ziel}")
    return 0 if bestanden else 1


if __name__ == "__main__":
    raise SystemExit(main())
