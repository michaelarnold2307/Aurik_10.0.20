"""scripts/verify_cantus_gate.py — SOTA-M14-Gate: Cantus-Vocal-Restorer (v3) gegen die Basis.

Hintergrund (SOTA-HANDOVER §2c Punkt 4, Leiter-Schluss 2026-11-19): Die
Cantus-Leiter v2 (pretrain 50 → fine_tune 30 → domain_adapt v3, 20 Epochen
NUR CC0-Vinyl auf `data/cantus_real`) ist abgeschlossen. Vertrag: **kein
Deploy ohne Never-worsen-Gate + Hörabnahme** — dieses Skript ist das
objektive Gate; die menschliche §4-Hör-Abnahme folgt separat.

Produktions-Wahrheit: Der Cantus-Kern ist im Modell-Zoo NICHT freigegeben
(`status="training"` in `backend/core/model_zoo_registry.py`), daher ist der
heutige Produktionspfad des Cantus-Schritts der **Wiener-DSP-Fallback**
(`CantusPlugin` → `dsp_fallback`, §V6 (copilot-instructions.md)). Die Basis
dieses Gates ist genau dieser Fallback — nicht ein früheres ONNX-Artefakt.

Zwei Messarme (M9-Muster, `scripts/verify_m9_gate.py`):

``clean``
    Never-worsen auf sauberem Gesang: MUSDB18HQ-Test-Split-`vocals.wav`
    (nicht Teil von `data/cantus_real`; Leak-Kautelen unten). Der
    v3-Kandidat darf den Eingang nicht über die JND verschlechtern
    (musik_mos ≥ −JND, defektlast ≤ +JND, HNR ≥ −0,5 dB je Track).

``degradiert``
    Echte degradierte Vocal-Paare (`data/cantus_real/*_degraded.wav`,
    deterministische Seed-42-Auswahl): der Kandidat muss die Wiener-Basis
    messbar schlagen (HNR > +0,5 dB ODER musik_mos > +JND ODER
    defektlast < −JND) auf ≥ 2 von 3 Tracks.

Gate (fail-closed): bestanden ⇔ never-worsen auf allen Clean-Tracks UND
Vorteil gegenüber der Wiener-Basis auf ≥ 2 von 3 degradierten Tracks.

Ehrliche Grenze: `data/cantus_real` ist die Trainings-Domäne von
domain_adapt v3 — Leak-Freiheit des Gewinn-Arms ist nicht nachweisbar
(dieselbe dokumentierte Kautel wie das M9-Gate bei den 25 Trainingspaaren).
Die Never-worsen-Aussage gilt auf den MUSDB18HQ-Test-Vocals; die
Generalisierung des Gewinns bleibt Vorbehalt bis zur §4-Hör-Abnahme.

Exit-Codes: 0 = Gate bestanden, 1 = Messung gültig, Gate nicht bestanden,
2 = Messung nicht möglich (Setup-Fehler/Artefakt fehlt).

CPU-only (keine GPU-Kette stören, Handover §5), deterministisch
(§G5 (copilot-instructions.md)): feste Track-/Paar-Auswahl über Seed 42.
Kein Flag-Flip, kein Deploy, keine Zoo-Freigabe — ein bestandenes Gate ist
die Voraussetzung für das §4-Verfahren, nicht dessen Ersatz.
"""

from __future__ import annotations

import argparse
import gc
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

logger = logging.getLogger("verify_cantus_gate")

SCHEMA = "aurik.cantus_gate/1"
MUSDB_TEST = _ROOT / "data" / "musdb18hq" / "test"
CANTUS_REAL = _ROOT / "data" / "cantus_real"
KANDIDAT_DEFAULT = _ROOT / "models" / "cantus" / "cantus_dit_v3_kandidat.onnx"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
SR = 48000  # Cantus-intern immer 48 kHz (§V6-Fallback sonst)

JND = 0.05
HNR_TOLERANZ_DB = 0.5
SEGMENT_S = 20.0
BENEFIT_MIN_TRACKS = 2
N_TRACKS = 3

MATERIAL = "mp3_low"  # in _TARGET_MATERIALS des Plugins — öffnet das Material-Gate
RESTORABILITY = 10.0  # weit unter der 30er-Schwelle — der Schritt wird geplant


def _cpu_guard() -> str | None:
    """Die Messung darf keine GPU-Kette stören (Handover §5).

    Der Kandidaten-Pfad ist CPU-only per Konstruktion: ein expliziter
    `model_path` setzt `_checkpoint_path = None`, womit der Torch-ROCm-Kern
    ausgeschlossen bleibt und die ONNX-Session hartkodiert
    `providers=["CPUExecutionProvider"]` läuft. Ein vorhandener ROCm-torch
    wird nur protokolliert, nicht als Abbruchgrund gewertet (M9-Vorbild).
    """
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("GPU vorhanden — Cantus-Gate bleibt CPU-only (expliziter ONNX-Pfad, kein CUDA-Init).")
    except Exception:  # pragma: no cover — torch fehlt ⇒ CPU ohnehin einziger Pfad
        return None
    return None


def _sha256(pfad: Path) -> str:
    h = hashlib.sha256()
    with pfad.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _lade_mono(pfad: Path, seconds: float | None) -> tuple[np.ndarray, int]:
    """Mono-Segment (float32, −1..1) + Quellrate; `seconds=None` = ganze Datei."""
    from scipy.io import wavfile

    roh = wavfile.read(pfad)
    if not isinstance(roh, tuple) or len(roh) != 2:
        raise RuntimeError("wavfile.read lieferte unerwartetes Format (Bug 12, §V36)")
    sr_src = int(roh[0])
    d = roh[1]
    x = np.asarray(d, np.float32)
    x = x / (float(np.abs(x).max()) + 1e-9)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if seconds is not None:
        n = int(seconds * sr_src)
        if len(mono) > n:
            mid = len(mono) // 2
            mono = mono[mid - n // 2 : mid + n // 2]
    return np.asarray(mono, dtype=np.float32), sr_src


def _auf_48k(mono: np.ndarray, sr_src: int) -> np.ndarray:
    """Deterministisches Resampling auf die Pipeline-Rate (§G5 (copilot-instructions.md))."""
    if sr_src == SR:
        return np.asarray(mono, dtype=np.float32)
    from scipy.signal import resample_poly

    gcd = int(np.gcd(SR, sr_src))
    return np.asarray(resample_poly(mono.astype(np.float64), SR // gcd, sr_src // gcd), dtype=np.float32)


def _waehle_vocal_tracks(anzahl: int, seed: int) -> list[Path]:
    if not MUSDB_TEST.is_dir():
        return []
    tracks = sorted(p for p in MUSDB_TEST.iterdir() if p.is_dir() and (p / "vocals.wav").is_file())
    if not tracks:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(tracks, min(anzahl, len(tracks))))


def _waehle_degradierte(anzahl: int, seed: int) -> list[Path]:
    """Deterministische Auswahl echter degradierter Vocals (§G5 (copilot-instructions.md)).

    Je Titel/Quelle wird genau ein Paar genommen — die Seed-42-Auswahl
    streut über Titel und Degradations-Arten (vinyl/rir/vinyl_rir).
    """
    if not CANTUS_REAL.is_dir():
        return []
    alle = sorted(CANTUS_REAL.glob("*_degraded.wav"))
    rng = random.Random(seed)
    return sorted(rng.sample(alle, min(anzahl, len(alle))))


def _zeugen(eingang: np.ndarray, ausgang: np.ndarray, sr: int) -> dict[str, Any]:
    """musik_mos + defektlast über den Wohlklang-Zeugen (Report-only-Messung)."""
    from backend.core.dsp.wohlklang_zeuge import messe_wohlklang_zeuge

    roh = messe_wohlklang_zeuge(eingang, ausgang, sr)
    return {
        "musik_mos": float(roh["ausgang"]["musik_mos"]),
        "defektlast": float(roh["ausgang"]["defektlast"]),
        "delta_musik_mos": float(roh["delta"]["musik_mos"]),
        "delta_defektlast": float(roh["delta"]["defektlast"]),
    }


def _hnr(audio: np.ndarray) -> float:
    from scripts.validate_hr_v1 import _hnr_db

    return float(_hnr_db(audio))


@dataclass
class CantusMessung:
    arm: str
    track: str
    hnr_eingang: float
    hnr_basis: float
    hnr_kandidat: float
    mos_eingang: float
    mos_basis: float
    mos_kandidat: float
    defekt_eingang: float
    defekt_basis: float
    defekt_kandidat: float
    basis_applied: bool
    kandidat_applied: bool

    def als_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "track": self.track,
            "hnr_eingang_db": round(self.hnr_eingang, 4),
            "hnr_basis_db": round(self.hnr_basis, 4),
            "hnr_kandidat_db": round(self.hnr_kandidat, 4),
            "delta_hnr_basis_db": round(self.hnr_basis - self.hnr_eingang, 4),
            "delta_hnr_kandidat_db": round(self.hnr_kandidat - self.hnr_eingang, 4),
            "musik_mos_eingang": round(self.mos_eingang, 4),
            "musik_mos_basis": round(self.mos_basis, 4),
            "musik_mos_kandidat": round(self.mos_kandidat, 4),
            "defektlast_eingang": round(self.defekt_eingang, 4),
            "defektlast_basis": round(self.defekt_basis, 4),
            "defektlast_kandidat": round(self.defekt_kandidat, 4),
            "basis_applied": self.basis_applied,
            "kandidat_applied": self.kandidat_applied,
        }


def _enhance_mit(plugin: Any, audio: np.ndarray) -> np.ndarray:
    """Produktions-Semantik: `enhance(material, restorability)` — echter Schritt inkl. Guards."""
    result = plugin.enhance(audio, SR, MATERIAL, restorability_score=RESTORABILITY)
    return np.asarray(result.audio, dtype=np.float32)


def _messe_clean(track: Path, kandidat_plugin: Any) -> CantusMessung:
    """Clean-Arm: Never-worsen des Kandidaten auf sauberem Gesang (kein Wiener-Vergleich nötig).

    Die Wiener-Basis wird auf clean ebenfalls gemessen — sie ist Teil des
    heutigen Produktionspfads; beide müssen den Eingang nicht verschlechtern.
    """
    segment, sr_src = _lade_mono(track / "vocals.wav", SEGMENT_S)
    basis_in = _auf_48k(segment, sr_src)

    from plugins.cantus_plugin import CantusPlugin

    # Basis-Instanz: kanonischer Pfad ⇒ Wiener-DSP-Fallback (kein ML, kein RAM-Budget-Fresser).
    basis_plugin = CantusPlugin()
    basis_out = _enhance_mit(basis_plugin, basis_in)
    basis_applied = bool(getattr(basis_plugin, "_fallback_active", False))
    del basis_plugin
    gc.collect()

    kandidat_out = _enhance_mit(kandidat_plugin, basis_in)

    z_in = _zeugen(basis_in, basis_in, SR)
    z_basis = _zeugen(basis_in, basis_out, SR)
    z_kand = _zeugen(basis_in, kandidat_out, SR)
    return CantusMessung(
        arm="clean",
        track=track.name,
        hnr_eingang=_hnr(basis_in),
        hnr_basis=_hnr(basis_out),
        hnr_kandidat=_hnr(kandidat_out),
        mos_eingang=float(z_in["musik_mos"]),
        mos_basis=float(z_basis["musik_mos"]),
        mos_kandidat=float(z_kand["musik_mos"]),
        defekt_eingang=float(z_in["defektlast"]),
        defekt_basis=float(z_basis["defektlast"]),
        defekt_kandidat=float(z_kand["defektlast"]),
        basis_applied=basis_applied,
        kandidat_applied=True,
    )


def _messe_degradiert(pfad: Path, kandidat_plugin: Any) -> CantusMessung:
    """Gewinn-Arm: Kandidat gegen die Wiener-Basis auf echtem degradiertem Gesang."""
    segment, sr_src = _lade_mono(pfad, None)
    eingang = _auf_48k(segment, sr_src)

    from plugins.cantus_plugin import CantusPlugin

    basis_plugin = CantusPlugin()
    basis_out = _enhance_mit(basis_plugin, eingang)
    basis_applied = bool(getattr(basis_plugin, "_fallback_active", False))
    del basis_plugin
    gc.collect()

    kandidat_out = _enhance_mit(kandidat_plugin, eingang)

    z_in = _zeugen(eingang, eingang, SR)
    z_basis = _zeugen(eingang, basis_out, SR)
    z_kand = _zeugen(eingang, kandidat_out, SR)
    return CantusMessung(
        arm="degradiert",
        track=pfad.name,
        hnr_eingang=_hnr(eingang),
        hnr_basis=_hnr(basis_out),
        hnr_kandidat=_hnr(kandidat_out),
        mos_eingang=float(z_in["musik_mos"]),
        mos_basis=float(z_basis["musik_mos"]),
        mos_kandidat=float(z_kand["musik_mos"]),
        defekt_eingang=float(z_in["defektlast"]),
        defekt_basis=float(z_basis["defektlast"]),
        defekt_kandidat=float(z_kand["defektlast"]),
        basis_applied=basis_applied,
        kandidat_applied=True,
    )


def _never_worsen_ok(m: CantusMessung) -> bool:
    if not m.kandidat_applied:
        return False
    return (
        (m.mos_kandidat - m.mos_eingang) >= -JND
        and (m.defekt_kandidat - m.defekt_eingang) <= JND
        and (m.hnr_kandidat - m.hnr_eingang) >= -HNR_TOLERANZ_DB
    )


def _vorteil(m: CantusMessung) -> bool:
    """Kandidat schlägt die Wiener-Basis messbar (M9-Schwellen)."""
    if not m.kandidat_applied:
        return False
    return (
        (m.hnr_kandidat - m.hnr_basis) > HNR_TOLERANZ_DB
        or (m.mos_kandidat - m.mos_basis) > JND
        or (m.defekt_basis - m.defekt_kandidat) > JND
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SOTA-M14-Gate: Cantus v3 gegen die Wiener-Basis")
    ap.add_argument("--tracks", type=int, default=N_TRACKS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--modell",
        type=Path,
        default=KANDIDAT_DEFAULT,
        help="ONNX-Pfad des v3-Kandidaten (Default: models/cantus/cantus_dit_v3_kandidat.onnx)",
    )
    args = ap.parse_args(argv)

    guard = _cpu_guard()
    if guard:
        print(f"ABBRUCH: {guard}")
        return 2

    modell_pfad = Path(args.modell)
    if not modell_pfad.is_file():
        print(f"ABBRUCH: Kandidaten-ONNX fehlt: {modell_pfad}")
        return 2

    from plugins.cantus_plugin import CantusPlugin

    kandidat_plugin = CantusPlugin(model_path=modell_pfad)
    if not getattr(kandidat_plugin, "_model_loaded", False):
        print(
            "ABBRUCH: Kandidat nicht geladen — "
            f"Fallback-Grund: {getattr(kandidat_plugin, '_fallback_reason', 'unbekannt')}"
        )
        return 2

    clean_tracks = _waehle_vocal_tracks(args.tracks, args.seed)
    degradierte = _waehle_degradierte(args.tracks, args.seed)
    if len(clean_tracks) < 3 or len(degradierte) < 3:
        print(
            f"ABBRUCH: zu wenig Messmaterial (clean={len(clean_tracks)}, degradiert={len(degradierte)}; je ≥3 nötig)."
        )
        return 2

    messungen: list[CantusMessung] = []
    for track in clean_tracks:
        messungen.append(_messe_clean(track, kandidat_plugin))
    for pfad in degradierte:
        messungen.append(_messe_degradiert(pfad, kandidat_plugin))

    clean = [m for m in messungen if m.arm == "clean"]
    degradiert = [m for m in messungen if m.arm == "degradiert"]
    never_worsen = all(_never_worsen_ok(m) for m in clean)
    vorteile = sum(1 for m in degradiert if _vorteil(m))
    bestanden = never_worsen and vorteile >= BENEFIT_MIN_TRACKS

    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": args.seed,
        "modell": str(modell_pfad.relative_to(_ROOT)) if modell_pfad.is_relative_to(_ROOT) else str(modell_pfad),
        "modell_sha256": _sha256(modell_pfad),
        "basis": "Wiener-DSP-Fallback (Cantus im Modell-Zoo status=training — heutiger Produktionspfad)",
        "material": MATERIAL,
        "restorability_score": RESTORABILITY,
        "clean_tracks": [t.name for t in clean_tracks],
        "degradierte_paare": [p.name for p in degradierte],
        "never_worsen": never_worsen,
        "vorteile_degradiert": vorteile,
        "vorteile_min": BENEFIT_MIN_TRACKS,
        "gate_bestanden": bestanden,
        "limitation": (
            "data/cantus_real ist die Trainings-Domäne von domain_adapt v3 — Leak-Freiheit des "
            "Gewinn-Arms nicht nachweisbar (M9-Kautel-Muster); Never-worsen auf MUSDB18HQ-Test-Vocals belegt."
        ),
        "messungen": [m.als_dict() for m in messungen],
    }

    ziel = REPORT_DIR / f"{date.today().isoformat()}_cantus_gate.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = REPORT_DIR / f"{date.today().isoformat()}_cantus_gate.md"
    zeilen = [
        "# SOTA-M14-Cantus-Gate — v3-Kandidat gegen die Wiener-Basis",
        "",
        f"- Datum: {beleg['datum']} | Seed {beleg['seed']} | "
        f"{len(clean_tracks)} MUSDB18HQ-Test-Vocals + {len(degradierte)} echte degradierte Paare",
        f"- Kandidat: `{beleg['modell']}` (sha256 `{beleg['modell_sha256'][:12]}…`)",
        f"- Never-worsen (clean): {'✓ BESTANDEN' if never_worsen else '✗ VERFEHLT'}",
        f"- Vorteile gegen die Wiener-Basis: {vorteile}/{len(degradiert)} (min {BENEFIT_MIN_TRACKS})",
        f"- **Gate: {'BESTANDEN' if bestanden else 'NICHT BESTANDEN'}** (Exit {'0' if bestanden else '1'})",
        "",
        "Hinweis: bestandenes Gate = Voraussetzung für §4-Hör-Abnahme, kein Deploy, keine Zoo-Freigabe.",
        f"Beleg: `{ziel.name}`",
    ]
    md.write_text("\n".join(zeilen) + "\n", encoding="utf-8")

    print(json.dumps({k: beleg[k] for k in ("never_worsen", "vorteile_degradiert", "gate_bestanden")}))
    print(f"Beleg: {ziel}")
    return 0 if bestanden else 1


if __name__ == "__main__":
    raise SystemExit(main())
