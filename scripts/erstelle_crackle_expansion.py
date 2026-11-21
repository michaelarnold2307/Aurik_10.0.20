"""scripts/erstelle_crackle_expansion.py — Crackle-Trainingsdaten über mehrere Titel.

Hintergrund (2026-11-19, SOTA-HANDOVER §2c/(37c)): Die k398-Klasse `crackle`
hat 391 Regionen, aber alle von EINEM Titel („Elke Best - Boogie Woogie
Mama“) — der Holdout-Leak-Schutz des Witness-Trainers lässt damit keine
Trainingsregionen übrig (holdout = der einzige Titel). Decrackle ist nach
Denoise der größte Register-Hebel; der Witness-Kopf braucht ≥2 Titel.

Dieses Skript erzeugt den fehlenden Daten-Nachschub deterministisch (§G5 (copilot-instructions.md), Seed 42):

  1. N saubere Titel aus `--quellen` (data/schadens_quellen, MUSDB-Mixe),
  2. Vinyl-Rezept mit CRACKLE/Surface-Noise (`erstelle_corpus_vinyl._vinyl_rezept`
     — CC0-Vinyl-Leerrillen-Bank data/vinyl_noise_bank, kein Schellack),
  3. Produktions-Restaurierung (`restauriere_schnipsel`, Pre-Analysis → Denker,
     CPU-only via AURIK_FORCE_CPU) ⇒ `(<titel>) (vinyl)_restauriert.wav`,
  4. Clean-Symlinks mit „original“ im Namen für die Paar-Generierung
     (`erstelle_klassen_paare.py`-Konvention).

Danach (extern): `erstelle_klassen_paare.py --schaden-dir <ausgang>
--clean-dir <clean-links> --rezepte vinyl --out data/klassen_paare_crackle_expansion`
und die crackle-Zeilen ins Haupt-Manifest `data/klassen_paare/manifest.jsonl`
anhängen. Report-only, kein Flag-Flip, kein Deploy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SCHEMA = "aurik.crackle_expansion/1"
DEFAULT_QUELLEN = _ROOT / "data" / "schadens_quellen"
DEFAULT_AUSGANG = _ROOT / "data" / "klassen_paare_expand_src"
DEFAULT_CLEAN = _ROOT / "data" / "klassen_paare_expand_clean"
DEFAULT_BANK = _ROOT / "data" / "vinyl_noise_bank"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
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

    ap = argparse.ArgumentParser(description="Crackle-Trainingsdaten über mehrere Titel")
    ap.add_argument("--titel", type=int, default=6, help="Anzahl sauberer Titel (≥3 für Leak-Schutz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quellen", type=Path, default=DEFAULT_QUELLEN)
    ap.add_argument("--ausgang", type=Path, default=DEFAULT_AUSGANG)
    ap.add_argument("--clean-links", type=Path, default=DEFAULT_CLEAN)
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    args = ap.parse_args(argv)

    os_env_force_cpu = {"AURIK_FORCE_CPU": "1"}
    import os

    for k, v in os_env_force_cpu.items():
        os.environ.setdefault(k, v)

    from scripts.erstelle_corpus_vinyl import _vinyl_rezept  # pylint: disable=import-outside-toplevel
    from scripts.erstelle_schadens_proben import restauriere_schnipsel  # pylint: disable=import-outside-toplevel

    bank = sorted(p for p in args.bank.glob("*.wav") if p.is_file())
    if len(bank) < 2:
        print(f"ABBRUCH: Vinyl-Rauschbank unvollständig ({len(bank)} Dateien in {args.bank}).")
        return 2

    quellen = sorted(p for p in args.quellen.glob("*.wav") if p.is_file())
    rng = np.random.default_rng(args.seed)
    auswahl = [quellen[i] for i in rng.permutation(len(quellen))][: args.titel]
    if len(auswahl) < 3:
        print(f"ABBRUCH: weniger als 3 Quellen-Titel gefunden ({len(auswahl)}).")
        return 2

    args.ausgang.mkdir(parents=True, exist_ok=True)
    args.clean_links.mkdir(parents=True, exist_ok=True)

    zeilen = []
    for i, quelle in enumerate(auswahl):
        stamm = quelle.stem
        import soundfile as sf

        # Resume-Sicherheit (2026-11-19, Crash-Härtung): fertige Titel werden
        # nicht neu gerechnet — ein abgebrochener Lauf setzt ohne Zeitverlust fort.
        schaden_pfad = args.ausgang / f"{stamm} (vinyl).wav"
        restauriert_pfad = args.ausgang / f"{stamm} (vinyl)_restauriert.wav"
        fertig = (
            schaden_pfad.is_file()
            and restauriert_pfad.is_file()
            and schaden_pfad.stat().st_size > 0
            and restauriert_pfad.stat().st_size > 0
        )
        # Der Rezeptlauf rekonstruiert auch beim Resume die Parameter für den Abschlussbeleg.
        clean, _ = _lade_mono_48k(quelle)
        rng_titel = np.random.default_rng(args.seed + i)
        schaden, params = _vinyl_rezept(clean, bank, rng_titel)
        link = args.clean_links / f"{stamm} original.wav"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(quelle.resolve())
        eintrag = {
            "titel": stamm,
            "quelle": str(quelle.relative_to(_ROOT)),
            "schaden": str(schaden_pfad.relative_to(_ROOT)),
            "restauriert": str(restauriert_pfad.relative_to(_ROOT)),
            "rezept_params": {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in params.items()
            },
        }
        if fertig:
            zeilen.append(eintrag)
            print(f"[{i + 1}/{len(auswahl)}] {stamm} — bereits fertig, übersprungen (Resume)")
            continue

        sf.write(str(schaden_pfad), np.asarray(schaden, dtype=np.float32), SR)
        restauriert = restauriere_schnipsel(np.asarray(schaden, dtype=np.float32), SR, str(schaden_pfad))
        sf.write(str(restauriert_pfad), np.asarray(restauriert, dtype=np.float32), SR)
        zeilen.append(eintrag)
        print(f"[{i + 1}/{len(auswahl)}] {stamm} — Schaden + Produktions-Restaurierung fertig")

    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": args.seed,
        "titel_anzahl": len(auswahl),
        "bank": str(args.bank.relative_to(_ROOT)),
        "eintraege": zeilen,
    }
    ziel = REPORT_DIR / f"{date.today().isoformat()}_crackle_expansion.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Beleg: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
