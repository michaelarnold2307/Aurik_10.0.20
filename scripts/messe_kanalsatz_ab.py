"""scripts/messe_kanalsatz_ab.py — Ganz-Titel-A/B: Witness-Kanalsatz AN vs. Bestand AUS (DRT-12).

Hintergrund (Spec 26 DRT-12, 2026-11-19): Die Ganz-Titel-Sinn-Beweise
(„0/12“, „6/45“) wurden mit **deaktivierten** Witness-Kanälen gemessen —
sie belegen den Bestand, nicht den Kandidaten-Zustand. Dieses Skript misst
die fehlende Seite: dieselben Schadens-Paare, dieselben Metriken, aber mit
allen gate-bestandenen Witness-Kanälen AKTIV (prozess-lokal).

Vertrag:
- **Mess-only:** die Kanal-Flags werden nur in DIESEM Prozess gesetzt
  (`backend.core.music_model_flags`-Attribute); keine Flag-Datei wird
  geschrieben, keine Produktions-Defaults werden verändert (§G1
  (copilot-instructions.md)-Bit-Identität bleibt unangetastet).
- **Produktions-Pfad:** die Restaurierung läuft über
  `scripts.erstelle_schadens_proben.restauriere_schnipsel` (Pre-Analysis →
  Denker) — dieselbe Semantik wie die Bestandsmessung; die Kanal-Module
  behalten ihre Trigger-Masken, Never-worsen-Guards und die DRT-11-
  Kontext-Sperre (kein Freibrief).
- **A-Seite:** falls eine Bestand-Bewertung (`--bestand`) übergeben wird,
  werden deren Δ-Werte als A-Seite verglichen; sonst wird nur die
  Kanalsatz-Seite gemessen (report-only).
- **Determinismus (§G5 (copilot-instructions.md)):** feste Paar-Auswahl über
  Seed 42; CPU-only via AURIK_FORCE_CPU (stört die GPU-Kette nicht).

Beleg: `docs/reports/current/<datum>_kanalsatz_ab.json`, Schema
`aurik.kanalsatz_ab/1`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SCHEMA = "aurik.kanalsatz_ab/1"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
JND = 0.05

# Gate-bestandene Witness-Kanäle (10 Alt + 2 Neu; groove_echo trägt die
# mos-Variante über denselben Kanal, DRT-8/10-Erstbeleg 2026-11-19).
KANALSAETZ_FLAGS = [
    "use_witness_kopf_groove_echo",
    "use_witness_kopf_tape_head_level_dip",
    "use_witness_kopf_clicks",
    "use_witness_kopf_compression_artifacts",
    "use_witness_kopf_jitter_artifacts",
    "use_witness_kopf_tape_head_clog",
    "use_witness_kopf_phase_rotation",
    "use_witness_kopf_sibilance",
    "use_witness_kopf_sticky_shed_residue",
    "use_witness_kopf_transport_bump",
    "use_witness_kopf_dynamic_compression_excess",
    "use_witness_kopf_dropout_head_contact",
]


def _aktiviere_kanalsatz() -> None:
    """Setzt die Kanal-Flags NUR in diesem Prozess (mess-only, DRT-12)."""
    import backend.core.music_model_flags as flags

    gesetzt = []
    for name in KANALSAETZ_FLAGS:
        if not hasattr(flags, name):
            raise RuntimeError(f"Kanal-Flag fehlt im music_model_flags-Modul: {name}")
        setattr(flags, name, True)
        gesetzt.append(name)
    print(f"Kanalsatz aktiviert ({len(gesetzt)} Flags, prozess-lokal).")


def _zeugen(schaden: np.ndarray, restauriert: np.ndarray, sr: int) -> dict[str, Any]:
    from backend.core.dsp.wohlklang_zeuge import messe_wohlklang_zeuge

    roh = messe_wohlklang_zeuge(schaden, restauriert, sr)
    return {
        "delta_musik_mos": float(roh["delta"]["musik_mos"]),
        "delta_defektlast": float(roh["delta"]["defektlast"]),
    }


def _paare_auswaehlen(verzeichnis: Path, anzahl: int, seed: int) -> list[tuple[str, Path]]:
    """Schadens-Paare (ohne _restauriert) deterministisch auswählen (Seed 42)."""
    alle = sorted(p for p in verzeichnis.glob("*.wav") if "_restauriert" not in p.name and "original" not in p.name)
    rng = np.random.default_rng(seed)
    auswahl = [alle[i] for i in rng.permutation(len(alle))][:anzahl]
    return [(p.stem, p) for p in auswahl]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DRT-12: Witness-Kanalsatz-A/B auf identischen Paaren")
    ap.add_argument("--verzeichnis", required=True, help="Ordner mit <titel> (<rezept>).wav-Schadenspaaren")
    ap.add_argument("--paare", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bestand", type=Path, default=None, help="bewertung.json der AUS-Messung (A-Seite)")
    ap.add_argument("--ausgang", type=Path, default=None, help="Optional: Kanalsatz-Restaurate hier ablegen")
    ap.add_argument("--beleg", type=Path, default=None, help="Optional: Beleg-Pfad überschreiben (sonst datumskanonisch)")
    args = ap.parse_args(argv)

    # CPU-Vertrag VOR allen backend-Importen (GPU-Kette läuft parallel).
    os.environ.setdefault("AURIK_FORCE_CPU", "1")
    os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

    from scripts.erstelle_schadens_proben import restauriere_schnipsel  # pylint: disable=import-outside-toplevel

    _aktiviere_kanalsatz()

    bestand = {}
    if args.bestand is not None and Path(args.bestand).is_file():
        roh = json.loads(Path(args.bestand).read_text(encoding="utf-8"))
        for z in roh.get("zeilen", []):
            song = z.get("song") or z.get("stamm") or ""
            rezept = z.get("rezept")
            key = f"{song} ({rezept})" if rezept else song
            bestand[key] = {
                "delta_musik_mos": z.get("delta_musik_mos"),
                "delta_defektlast": z.get("delta_defektlast"),
                "besser": bool(z.get("besser")),
            }

    verzeichnis = Path(args.verzeichnis)
    paare = _paare_auswaehlen(verzeichnis, args.paare, args.seed)
    if not paare:
        print("ABBRUCH: keine Schadens-Paare gefunden.")
        return 2

    zeilen = []
    for i, (stamm, pfad) in enumerate(paare):
        import soundfile as sf
        from scipy.io import wavfile

        roh = wavfile.read(pfad)
        if not isinstance(roh, tuple) or len(roh) != 2:
            print(f"ÜBERSPRUNGEN (Format): {pfad.name}")
            continue
        sr = int(roh[0])
        d = roh[1]
        x = np.asarray(d, np.float32) / 32768.0 if np.asarray(d).dtype != np.float32 else np.asarray(d, np.float32)
        schaden = x.mean(axis=1) if x.ndim == 2 else x

        restauriert = restauriere_schnipsel(schaden, sr, str(pfad))
        z = _zeugen(schaden, restauriert, sr)

        eintrag: dict[str, Any] = {
            "stamm": stamm,
            "kanalsatz_delta_musik_mos": round(z["delta_musik_mos"], 4),
            "kanalsatz_delta_defektlast": round(z["delta_defektlast"], 4),
        }
        b = bestand.get(stamm)
        if b is not None and b.get("delta_musik_mos") is not None:
            eintrag["bestand_delta_musik_mos"] = round(float(b["delta_musik_mos"]), 4)
            eintrag["bestand_delta_defektlast"] = round(float(b["delta_defektlast"]), 4)
            eintrag["delta_zu_bestand_mos"] = round(z["delta_musik_mos"] - float(b["delta_musik_mos"]), 4)
            eintrag["delta_zu_bestand_defektlast"] = round(z["delta_defektlast"] - float(b["delta_defektlast"]), 4)
        zeilen.append(eintrag)

        if args.ausgang is not None:
            args.ausgang.mkdir(parents=True, exist_ok=True)
            sf.write(str(args.ausgang / f"{stamm}_kanalsatz.wav"), restauriert, sr)
        print(f"[{i + 1}/{len(paare)}] {stamm} — ΔMOS {z['delta_musik_mos']:+.3f}, Δdefektlast {z['delta_defektlast']:+.3f}")

    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": args.seed,
        "verzeichnis": str(verzeichnis.relative_to(_ROOT)) if verzeichnis.is_relative_to(_ROOT) else str(verzeichnis),
        "kanalsatz_flags": KANALSAETZ_FLAGS,
        "n_paare": len(zeilen),
        "bestand_beleg": str(args.bestand) if args.bestand else None,
        "zeilen": zeilen,
    }
    ziel = args.beleg if args.beleg is not None else REPORT_DIR / f"{date.today().isoformat()}_kanalsatz_ab.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Beleg: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
