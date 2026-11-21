"""scripts/messe_kanalsatz_ab.py — Ganz-Titel-A/B: Witness-Kanalsatz AN vs. Bestand AUS (DRT-12).

Hintergrund (Spec 26 DRT-12, 2026-11-19): Die Ganz-Titel-Sinn-Beweise
(„0/12“, „6/45“) wurden mit **deaktivierten** Witness-Kanälen gemessen —
sie belegen den Bestand, nicht den Kandidaten-Zustand. Dieses Skript misst
die fehlende Seite: dieselben Schadens-Paare, dieselben Metriken, aber mit
allen gate-bestandenen Witness-Kanälen AKTIV (prozess-lokal).

**Crash-Härtung Runde 2 (2026-11-19, Smoke-SIGABRT-Befund):** der 4-Paar-Smoke
endete mit `Fatal Python error: Aborted` (Faulthandler: phase_10_compression →
perceptual_loudness_cap → temporal_loudness) — der Prozess starb MITTEN im
Lauf, ohne Beleg. Für die 45-Paar-Evidenz ist das untragbar, deshalb:

- **Subprozess-Isolation:** jedes Paar wird in einem EIGENEN Kindprozess
  gerechnet (`--einzel N`). Ein nativer Crash des Kinds kostet nur dieses
  eine Paar; der Elternprozess schreibt `status: "abgestuerzt"` + Returncode
  und rechnet die übrigen Paare weiter.
- **Inkrementeller Beleg:** nach JEDEM Paar wird der Beleg neu geschrieben
  (abgeschlossene Paare gehen nie verloren).
- **Resume:** erfolgreiche Paar-Zeilendateien werden übersprungen; abgestürzte
  Paare werden erneut gerechnet. Nicht erfolgreiche Endergebnisse halten die
  Kette mit einem Fehlerstatus an, statt als vollständige Evidenz zu gelten.

Vertrag (unverändert):
- **Mess-only:** die Kanal-Flags werden nur in DIESEM Prozess gesetzt
  (`backend.core.music_model_flags`-Attribute); keine Flag-Datei wird
  geschrieben, keine Produktions-Defaults werden verändert (§G1 (copilot-instructions.md)-Bit-Identität bleibt unangetastet).
- **Produktions-Pfad:** die Restaurierung läuft über
  `scripts.erstelle_schadens_proben.restauriere_schnipsel` (Pre-Analysis →
  Denker) — dieselbe Semantik wie die Bestandsmessung; die Kanal-Module
  behalten ihre Trigger-Masken, Never-worsen-Guards und die DRT-11-
  Kontext-Sperre (kein Freibrief).
- **A-Seite:** falls eine Bestand-Bewertung (`--bestand`) übergeben wird,
  werden deren Δ-Werte als A-Seite verglichen; sonst wird nur die
  Kanalsatz-Seite gemessen (report-only).
- **Isolierte Ablation:** `--kanal <flag>` aktiviert genau einen Witness-Kanal.
  Flag-Satz und Zeilen werden im Beleg verankert; Resume über andere
  Flag-Konfigurationen wird abgewiesen.
- **Determinismus (§G5 (copilot-instructions.md)):** feste Paar-Auswahl über
  Seed 42; CPU-only via AURIK_FORCE_CPU (stört die GPU-Kette nicht).

Beleg: `docs/reports/current/<datum>_kanalsatz_ab[_voll].json`, Schema
`aurik.kanalsatz_ab/1` (additiv um `status`/`returncode` je Zeile +
`modus: subprozess-isoliert` und die kanonische Überrestaurations-Klassifikation).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def _aktiviere_kanalsatz(kanal: str | None = None) -> None:
    """Aktiviert den vollständigen Satz oder genau einen Kanal, nur prozess-lokal."""
    import backend.core.music_model_flags as flags

    if kanal is not None and kanal not in KANALSAETZ_FLAGS:
        raise ValueError(f"Unbekannter DRT-12-Kanal: {kanal}")
    aktiviert = KANALSAETZ_FLAGS if kanal is None else [kanal]
    for name in KANALSAETZ_FLAGS:
        if not hasattr(flags, name):
            raise RuntimeError(f"Kanal-Flag fehlt im music_model_flags-Modul: {name}")
        setattr(flags, name, name in aktiviert)
    print(f"Kanalsatz aktiviert ({len(aktiviert)} Flags, prozess-lokal).")


def _zeugen(schaden: np.ndarray, restauriert: np.ndarray, sr: int) -> dict[str, Any]:
    from backend.core.dsp.wohlklang_zeuge import messe_wohlklang_zeuge

    roh = messe_wohlklang_zeuge(schaden, restauriert, sr)
    return {
        "delta_musik_mos": float(roh["delta"]["musik_mos"]),
        "delta_defektlast": float(roh["delta"]["defektlast"]),
        "ueberrestauration": bool(roh["ueberrestauration"]),
    }


def _paare_auswaehlen(verzeichnis: Path, anzahl: int, seed: int) -> list[tuple[str, Path]]:
    """Schadens-Paare (ohne _restauriert) deterministisch auswählen (Seed 42)."""
    alle = sorted(p for p in verzeichnis.glob("*.wav") if "_restauriert" not in p.name and "original" not in p.name)
    rng = np.random.default_rng(seed)
    auswahl = [alle[i] for i in rng.permutation(len(alle))][:anzahl]
    return [(p.stem, p) for p in auswahl]


def _bestand_laden(bestand_pfad: Path | None) -> dict[str, dict[str, Any]]:
    bestand: dict[str, dict[str, Any]] = {}
    if bestand_pfad is not None and bestand_pfad.is_file():
        roh = json.loads(bestand_pfad.read_text(encoding="utf-8"))
        for z in roh.get("zeilen", []):
            song = z.get("song") or z.get("stamm") or ""
            rezept = z.get("rezept")
            key = f"{song} ({rezept})" if rezept else song
            bestand[key] = {
                "delta_musik_mos": z.get("delta_musik_mos"),
                "delta_defektlast": z.get("delta_defektlast"),
                "besser": bool(z.get("besser")),
                "ueberrestauration": z.get("ueberrestauration"),
            }
    return bestand


def _zeilen_dir(beleg: Path) -> Path:
    return beleg.parent / (beleg.stem + "_zeilen")


def _unvollstaendigen_beleg_finden(
    paare: list[tuple[str, Path]],
    verzeichnis: Path,
    seed: int,
    bestand_pfad: Path | None,
    berichts_ordner: Path = REPORT_DIR,
    kanalsatz_flags: list[str] | None = None,
) -> Path | None:
    """Findet den jüngsten passenden, noch nicht erfolgreichen Voll-Lauf."""
    verzeichnis_label = str(verzeichnis.relative_to(_ROOT)) if verzeichnis.is_relative_to(_ROOT) else str(verzeichnis)
    bestand_label = str(bestand_pfad) if bestand_pfad else None
    erwartete_flags = KANALSAETZ_FLAGS if kanalsatz_flags is None else kanalsatz_flags
    erwartete_staemme = [stamm for stamm, _ in paare]
    for kandidat in sorted(berichts_ordner.glob("*_kanalsatz_ab*_voll.json"), reverse=True):
        try:
            daten = json.loads(kandidat.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Resume-Beleg übersprungen ({kandidat}): {exc}")
            continue
        if not isinstance(daten, dict) or daten.get("schema") != SCHEMA:
            continue
        if (
            daten.get("seed") != seed
            or daten.get("verzeichnis") != verzeichnis_label
            or daten.get("bestand_beleg") != bestand_label
            or daten.get("n_geplant") != len(paare)
            or daten.get("kanalsatz_flags", KANALSAETZ_FLAGS) != erwartete_flags
        ):
            continue
        zeilen = daten.get("zeilen")
        if not isinstance(zeilen, list) or len(zeilen) > len(paare):
            continue
        staemme = [zeile.get("stamm") if isinstance(zeile, dict) else None for zeile in zeilen]
        if staemme != erwartete_staemme[: len(zeilen)]:
            continue
        if len(zeilen) < len(paare) or any(zeile.get("status") != "ok" for zeile in zeilen):
            return kandidat
    return None


def _beleg_schreiben(
    beleg: Path,
    paare: list[tuple[str, Path]],
    verzeichnis: Path,
    seed: int,
    bestand_pfad: Path | None,
    zeilen: list[dict[str, Any]],
    kanalsatz_flags: list[str] | None = None,
) -> None:
    beleg_pfad = beleg
    daten = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": seed,
        "modus": "subprozess-isoliert",
        "verzeichnis": str(verzeichnis.relative_to(_ROOT)) if verzeichnis.is_relative_to(_ROOT) else str(verzeichnis),
        "kanalsatz_flags": KANALSAETZ_FLAGS if kanalsatz_flags is None else kanalsatz_flags,
        "n_paare": len(zeilen),
        "n_geplant": len(paare),
        "bestand_beleg": str(bestand_pfad) if bestand_pfad else None,
        "zeilen": zeilen,
    }
    beleg_pfad.write_text(json.dumps(daten, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _kind_einzel(argv: list[str]) -> int:
    """Kind-Modus: EIN Paar rechnen und als Zeilen-Datei ablegen (crash-isoliert)."""
    import faulthandler as _fh_mod

    _fh_log = _ROOT / "logs" / "python_faulthandler_ketten.log"
    _fh_log.parent.mkdir(parents=True, exist_ok=True)
    _fh_handle = open(_fh_log, "a", encoding="utf-8")
    _fh_mod.enable(file=_fh_handle, all_threads=True)

    ap = argparse.ArgumentParser(description="DRT-12-Kanalsatz: Einzelpaar-Kindprozess")
    ap.add_argument("--verzeichnis", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bestand", type=Path, default=None)
    ap.add_argument("--ausgang", type=Path, default=None)
    ap.add_argument("--kanal", choices=KANALSAETZ_FLAGS, default=None)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--paare", type=int, required=True)
    ap.add_argument("--zeile-datei", type=Path, required=True)
    args = ap.parse_args(argv)

    os.environ.setdefault("AURIK_FORCE_CPU", "1")
    os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

    from scripts.erstelle_schadens_proben import restauriere_schnipsel  # pylint: disable=import-outside-toplevel

    kanal_flags = KANALSAETZ_FLAGS if args.kanal is None else [args.kanal]
    _aktiviere_kanalsatz(args.kanal)
    bestand = _bestand_laden(args.bestand)
    paare = _paare_auswaehlen(Path(args.verzeichnis), args.paare, args.seed)
    stamm, pfad = paare[args.index]

    from scipy.io import wavfile

    roh = wavfile.read(pfad)
    if not isinstance(roh, tuple) or len(roh) != 2:
        ergebnis: dict[str, Any] = {"stamm": stamm, "status": "format_fehler"}
    else:
        sr = int(roh[0])
        d = roh[1]
        x = np.asarray(d, np.float32) / 32768.0 if np.asarray(d).dtype != np.float32 else np.asarray(d, np.float32)
        schaden = x.mean(axis=1) if x.ndim == 2 else x

        restauriert = restauriere_schnipsel(schaden, sr, str(pfad))
        z = _zeugen(schaden, restauriert, sr)

        ergebnis = {
            "stamm": stamm,
            "status": "ok",
            "kanalsatz_delta_musik_mos": round(z["delta_musik_mos"], 4),
            "kanalsatz_delta_defektlast": round(z["delta_defektlast"], 4),
            "kanalsatz_ueberrestauration": z["ueberrestauration"],
        }
        b = bestand.get(stamm)
        if b is not None and b.get("delta_musik_mos") is not None:
            ergebnis["bestand_delta_musik_mos"] = round(float(b["delta_musik_mos"]), 4)
            ergebnis["bestand_delta_defektlast"] = round(float(b["delta_defektlast"]), 4)
            ergebnis["bestand_ueberrestauration"] = b.get("ueberrestauration")
            ergebnis["delta_zu_bestand_mos"] = round(z["delta_musik_mos"] - float(b["delta_musik_mos"]), 4)
            ergebnis["delta_zu_bestand_defektlast"] = round(z["delta_defektlast"] - float(b["delta_defektlast"]), 4)

        if args.ausgang is not None:
            import soundfile as sf

            args.ausgang.mkdir(parents=True, exist_ok=True)
            sf.write(str(args.ausgang / f"{stamm}_kanalsatz.wav"), restauriert, sr)
        print(
            f"[{args.index + 1}/{args.paare}] {stamm} — ΔMOS {z['delta_musik_mos']:+.3f}, Δdefektlast {z['delta_defektlast']:+.3f}"
        )

    args.zeile_datei.parent.mkdir(parents=True, exist_ok=True)
    ergebnis["kanalsatz_flags"] = kanal_flags
    tmp = args.zeile_datei.with_suffix(".tmp")
    tmp.write_text(json.dumps(ergebnis, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(args.zeile_datei)
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--index" in argv or "--einzel" in argv:
        argv = [a for a in argv if a != "--einzel"]
        if "--index" not in argv:
            raise SystemExit("Kind-Modus braucht --index")
        return _kind_einzel(argv)

    ap = argparse.ArgumentParser(description="DRT-12: Witness-Kanalsatz-A/B auf identischen Paaren")
    ap.add_argument("--verzeichnis", required=True, help="Ordner mit <titel> (<rezept>).wav-Schadenspaaren")
    ap.add_argument("--paare", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bestand", type=Path, default=None, help="bewertung.json der AUS-Messung (A-Seite)")
    ap.add_argument("--ausgang", type=Path, default=None, help="Optional: Kanalsatz-Restaurate hier ablegen")
    ap.add_argument(
        "--beleg", type=Path, default=None, help="Optional: Beleg-Pfad überschreiben (sonst datumskanonisch)"
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Jüngsten passenden unvollständigen Voll-Beleg fortsetzen, auch über einen Tageswechsel hinweg.",
    )
    ap.add_argument(
        "--kanal",
        choices=KANALSAETZ_FLAGS,
        default=None,
        help="Nur diesen Witness-Kanal testen; ohne Angabe läuft der vollständige 12-Flag-Satz.",
    )
    args = ap.parse_args(argv)
    kanal_flags = KANALSAETZ_FLAGS if args.kanal is None else [args.kanal]
    kanal_suffix = f"_{args.kanal}" if args.kanal is not None else ""

    os.environ.setdefault("AURIK_FORCE_CPU", "1")
    os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

    verzeichnis = Path(args.verzeichnis)
    paare = _paare_auswaehlen(verzeichnis, args.paare, args.seed)
    if not paare:
        print("ABBRUCH: keine Schadens-Paare gefunden.")
        return 2
    verzeichnis_label = str(verzeichnis.relative_to(_ROOT)) if verzeichnis.is_relative_to(_ROOT) else str(verzeichnis)
    bestand_label = str(args.bestand) if args.bestand else None

    if args.beleg is not None:
        ziel = args.beleg
    elif args.resume:
        ziel = _unvollstaendigen_beleg_finden(
            paare,
            verzeichnis,
            args.seed,
            args.bestand,
            kanalsatz_flags=kanal_flags,
        ) or (REPORT_DIR / f"{date.today().isoformat()}_kanalsatz_ab{kanal_suffix}_voll.json")
        print(f"Voll-Lauf-Beleg für Resume: {ziel}")
    else:
        ziel = REPORT_DIR / f"{date.today().isoformat()}_kanalsatz_ab{kanal_suffix}.json"

    if ziel.is_file():
        try:
            vorheriger_beleg = json.loads(ziel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ABBRUCH: vorhandener Beleg ist nicht lesbar ({ziel}): {exc}")
            return 2
        if not isinstance(vorheriger_beleg, dict):
            print(f"ABBRUCH: vorhandener Beleg hat kein Objektformat ({ziel}).")
            return 2
        erwartete_identitaet = {
            "schema": SCHEMA,
            "seed": args.seed,
            "verzeichnis": verzeichnis_label,
            "n_geplant": len(paare),
            "bestand_beleg": bestand_label,
            "kanalsatz_flags": kanal_flags,
        }
        vorhandene_identitaet = {
            "schema": vorheriger_beleg.get("schema"),
            "seed": vorheriger_beleg.get("seed"),
            "verzeichnis": vorheriger_beleg.get("verzeichnis"),
            "n_geplant": vorheriger_beleg.get("n_geplant"),
            "bestand_beleg": vorheriger_beleg.get("bestand_beleg"),
            "kanalsatz_flags": vorheriger_beleg.get("kanalsatz_flags", KANALSAETZ_FLAGS),
        }
        if vorhandene_identitaet != erwartete_identitaet:
            print(
                f"ABBRUCH: Beleg {ziel} gehört zu einer anderen Messkonfiguration; bitte einen neuen Beleg-Pfad verwenden."
            )
            return 2

    zdir = _zeilen_dir(ziel)
    zdir.mkdir(parents=True, exist_ok=True)

    zeilen: list[dict[str, Any]] = []
    for i, (stamm, _pfad) in enumerate(paare):
        zeile_pfad = zdir / f"{i:03d}_{stamm[:40].replace('/', '_')}.json"
        if zeile_pfad.is_file():
            fertig = json.loads(zeile_pfad.read_text(encoding="utf-8"))
            if not isinstance(fertig, dict):
                print(f"ABBRUCH: Zeile {zeile_pfad} hat kein Objektformat.")
                return 2
            vorhandene_flags = fertig.get("kanalsatz_flags", KANALSAETZ_FLAGS)
            if vorhandene_flags != kanal_flags:
                print(
                    f"ABBRUCH: Zeile {zeile_pfad} wurde mit einem anderen Kanalsatz erzeugt; "
                    "bitte einen neuen Beleg-Pfad verwenden."
                )
                return 2
            status = fertig.get("status")
            if status == "ok":
                zeilen.append(fertig)
                _beleg_schreiben(ziel, paare, verzeichnis, args.seed, args.bestand, zeilen, kanal_flags)
                print(f"[{i + 1}/{len(paare)}] {stamm} — vorhanden (Resume): ok")
                continue
            if status != "abgestuerzt":
                zeilen.append(fertig)
                _beleg_schreiben(ziel, paare, verzeichnis, args.seed, args.bestand, zeilen, kanal_flags)
                print(f"[{i + 1}/{len(paare)}] {stamm} — nicht erfolgreich ({status}), bleibt offen.")
                continue
            print(f"[{i + 1}/{len(paare)}] {stamm} — vorheriger Absturz, erneuter Versuch.")

        kind_argv = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--verzeichnis",
            str(verzeichnis),
            "--seed",
            str(args.seed),
            "--paare",
            str(args.paare),
            "--index",
            str(i),
            "--zeile-datei",
            str(zeile_pfad),
        ]
        if args.kanal is not None:
            kind_argv += ["--kanal", args.kanal]
        if args.bestand is not None:
            kind_argv += ["--bestand", str(args.bestand)]
        if args.ausgang is not None:
            kind_argv += ["--ausgang", str(args.ausgang)]

        print(f"[{i + 1}/{len(paare)}] {stamm} — Kindprozess gestartet …")
        abgeschlossen = subprocess.run(kind_argv, check=False, cwd=str(_ROOT))
        if abgeschlossen.returncode == 0 and zeile_pfad.is_file():
            fertig = json.loads(zeile_pfad.read_text(encoding="utf-8"))
        else:
            fertig = {
                "stamm": stamm,
                "status": "abgestuerzt",
                "returncode": abgeschlossen.returncode,
                "kanalsatz_flags": kanal_flags,
            }
            zeile_pfad.write_text(json.dumps(fertig, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[{i + 1}/{len(paare)}] {stamm} — Kind abgestürzt (rc={abgeschlossen.returncode}), weiter.")
        zeilen.append(fertig)
        _beleg_schreiben(ziel, paare, verzeichnis, args.seed, args.bestand, zeilen, kanal_flags)

    _beleg_schreiben(ziel, paare, verzeichnis, args.seed, args.bestand, zeilen, kanal_flags)
    n_ok = sum(1 for z in zeilen if z.get("status") == "ok")
    n_fehler = len(zeilen) - n_ok
    print(f"Beleg: {ziel} ({n_ok} ok, {n_fehler} nicht erfolgreich, {len(zeilen)} gesamt)")
    return 0 if len(zeilen) == len(paare) and n_fehler == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
