"""scripts/erstelle_vocal_harshness_expansion.py — Vocal-Harshness-Trainingsdaten über viele Titel.

Hintergrund (2026-11-19, SOTA-HANDOVER 37c): Die Witness-Klasse
`vocal_harshness` hat nur 4 Regionen im Haupt-Manifest (3 Titel,
clipping/kassette) — der Holdout-Leak-Schutz lässt damit kaum
Trainingsregionen übrig. Die 4 Bestandszeilen entstanden aus den Rezepten
`clipping` und `kassette` auf gesangsstarken Titeln — genau dort erzeugt
der Defekt-Detektor VOCAL_HARSHNESS-Regionen.

Dieses Skript erzeugt den fehlenden Daten-Nachschub deterministisch
(§G5 (copilot-instructions.md), Seed 42) nach dem Crackle-Muster:

  1. N gesangsstarke Titel aus `--quellen` (Default: MUSDB-Train-Mixe,
     `*/mixture.wav` — 100 Titel, alle mit Gesang),
  2. je Titel die Rezepte `clipping` und `kassette` (`schaden_rezept`),
  3. Produktions-Restaurierung (`restauriere_schnipsel`, Pre-Analysis →
     Denker, CPU-only via AURIK_FORCE_CPU) ⇒ `(<titel>) (<rezept>)_restauriert.wav`,
  4. Clean-Symlinks mit „original“ im Namen für die Paar-Generierung
     (`erstelle_klassen_paare.py`-Konvention).

Danach (extern, in der Watchdog-Kette):
`erstelle_klassen_paare.py --schaden-dir <ausgang> --clean-dir <clean-links>
--rezepte clipping,kassette --out data/klassen_paare_vh` und
`haenge_klassen_manifest_an.py --klasse vocal_harshness` ans Haupt-Manifest.

Resume-sicher: fertige (Titel, Rezept)-Paare werden übersprungen.
Report-only, kein Flag-Flip, kein Deploy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SCHEMA = "aurik.vocal_harshness_expansion/1"
DEFAULT_QUELLEN = _ROOT / "data" / "musdb18hq" / "train"
DEFAULT_AUSGANG = _ROOT / "data" / "vocal_harshness_expand_src"
DEFAULT_CLEAN = _ROOT / "data" / "vocal_harshness_expand_clean"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
REZEPTE = ("clipping", "kassette")
SR = 48000


def _lade_mono_48k(pfad: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile

    roh = wavfile.read(pfad)
    if not isinstance(roh, tuple) or len(roh) != 2:
        raise RuntimeError("wavfile.read lieferte unerwartetes Format (Bug 12, §V36)")
    sr_src = int(roh[0])
    d = roh[1]
    x = np.asarray(d, np.float32)
    x = x / (float(np.abs(x).max()) + 1e-9)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if sr_src != SR:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(SR, sr_src))
        mono = np.asarray(resample_poly(mono.astype(np.float64), SR // gcd, sr_src // gcd), dtype=np.float32)
    return mono, SR


def main(argv: list[str] | None = None) -> int:
    # Crash-Diagnose (2026-11-19): nativer Crash (SIGSEGV etc.) landet im
    # faulthandler-Log statt spurlos zu verschwinden.
    import faulthandler as _fh_mod

    _fh_log = _ROOT / "logs" / "python_faulthandler_ketten.log"
    _fh_log.parent.mkdir(parents=True, exist_ok=True)
    _fh_handle = open(_fh_log, "a", encoding="utf-8")
    _fh_mod.enable(file=_fh_handle, all_threads=True)

    ap = argparse.ArgumentParser(description="Vocal-Harshness-Trainingsdaten über viele Titel")
    ap.add_argument("--titel", type=int, default=20, help="Anzahl sauberer Titel (≥3 für Leak-Schutz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quellen", type=Path, default=DEFAULT_QUELLEN)
    ap.add_argument("--ausgang", type=Path, default=DEFAULT_AUSGANG)
    ap.add_argument("--clean-links", type=Path, default=DEFAULT_CLEAN)
    args = ap.parse_args(argv)

    for k, v in {"AURIK_FORCE_CPU": "1", "AURIK_NO_RT_LIMIT": "1"}.items():
        os.environ.setdefault(k, v)

    from scripts.erstelle_schadens_proben import (  # pylint: disable=import-outside-toplevel
        restauriere_schnipsel,
        schaden_rezept,
    )

    quellen = sorted(args.quellen.glob("*/mixture.wav")) if args.quellen.is_dir() else []
    if not quellen:
        print(f"ABBRUCH: keine `*/mixture.wav`-Quellen unter {args.quellen}.")
        return 2
    rng = np.random.default_rng(args.seed)
    auswahl = [quellen[i] for i in rng.permutation(len(quellen))][: args.titel]
    if len(auswahl) < 3:
        print(f"ABBRUCH: weniger als 3 Quellen-Titel gefunden ({len(auswahl)}).")
        return 2

    args.ausgang.mkdir(parents=True, exist_ok=True)
    args.clean_links.mkdir(parents=True, exist_ok=True)

    zeilen = []
    gesamt = len(auswahl) * len(REZEPTE)
    lauf = 0
    for quelle in auswahl:
        stamm = quelle.parent.name  # MUSDB-Mixe heißen alle `mixture.wav` — Stamm = Titelordner
        import soundfile as sf

        clean = None
        for rezept in REZEPTE:
            lauf += 1
            schaden_pfad = args.ausgang / f"{stamm} ({rezept}).wav"
            restauriert_pfad = args.ausgang / f"{stamm} ({rezept})_restauriert.wav"
            eintrag = {
                "titel": stamm,
                "quelle": str(quelle.relative_to(_ROOT)),
                "rezept": rezept,
                "schaden": str(schaden_pfad.relative_to(_ROOT)),
                "restauriert": str(restauriert_pfad.relative_to(_ROOT)),
            }
            if (
                schaden_pfad.is_file()
                and restauriert_pfad.is_file()
                and schaden_pfad.stat().st_size > 0
                and restauriert_pfad.stat().st_size > 0
            ):
                zeilen.append(eintrag)
                print(f"[{lauf}/{gesamt}] {stamm} ({rezept}) — bereits fertig, übersprungen (Resume)")
                continue
            if clean is None:
                clean, _ = _lade_mono_48k(quelle)
            rng_titel = np.random.default_rng(args.seed + lauf * 7919)
            schaden = schaden_rezept(clean, rezept, rng_titel)
            sf.write(str(schaden_pfad), np.asarray(schaden, dtype=np.float32), SR)
            restauriert = restauriere_schnipsel(np.asarray(schaden, dtype=np.float32), SR, str(schaden_pfad))
            sf.write(str(restauriert_pfad), np.asarray(restauriert, dtype=np.float32), SR)
            zeilen.append(eintrag)
            print(f"[{lauf}/{gesamt}] {stamm} ({rezept}) — Schaden + Produktions-Restaurierung fertig")
        # Clean-Symlink mit „original“ im Namen (erstelle_klassen_paare-Konvention).
        link = args.clean_links / f"{stamm} original.wav"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(quelle.resolve())

    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": args.seed,
        "titel_anzahl": len(auswahl),
        "rezepte": list(REZEPTE),
        "quellen": str(args.quellen.relative_to(_ROOT)),
        "eintraege": zeilen,
    }
    ziel = REPORT_DIR / f"{date.today().isoformat()}_vocal_harshness_expansion.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Beleg: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
