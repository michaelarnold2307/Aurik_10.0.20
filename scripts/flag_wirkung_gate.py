#!/usr/bin/env python3
"""flag_wirkung_gate — jeder `use_*`-Schalter muss im Produktionspfad wirken.

Hintergrund (D-K3-85/D-K3-90/D-K3-91, Klasse „Phantom-Verdrahtung"): Ein
Schalter in `backend/core/music_model_flags.py` wurde mehrfach als „Default
AUS, bit-identisch" dokumentiert, ohne dass ihn irgendein Produktionsmodul
las — die B3-Messung verglich dann den Pfad mit sich selbst (No-Op) und
lieferte ein Mess-Artefakt statt eines Qualitätsurteils. Der manuelle
Suchlauf (2026-11-10, 24 Flags) findet solche Fälle, vergisst sie aber beim
nächsten Mal. Dieses Gate macht daraus eine maschinelle Invariante.

Drei Stufen (alle fail-closed):

A. **Lesestelle** — jeder Flag wird außerhalb von
   `backend/core/music_model_flags.py` mindestens einmal gelesen.
   Als Lesestelle zählt ein Name-Load, ein Attribut-Zugriff (`mmf.use_x`)
   oder ein Import-Alias, dessen lokaler Name danach auch geladen wird
   (ein toter `import ... as` zählt ausdrücklich NICHT).
B. **Produktionslesestelle** — dieselbe Definition, beschränkt auf
   `backend/`, `plugins/`, `denker/`, `Aurik10/`, `cli/`, `forensics/`.
   Ein nur in `scripts/` gelesener Flag ist im Betrieb wirkungslos; er ist
   nur mit einer begründeten Ausnahme (Register-ID) zulässig.
C. **B3-/Release-Kandidaten** — jeder Flag, der in einem Mess-/Freigabe-
   Skript (`scripts/b3_*.py`, `scripts/release_candidate_pipeline.py`) als
   Kandidat genannt wird, MUSS Stufe B erfüllen. Sonst misst die Stufe
   nichts (genau die M12-WF4/TP-V2-Lehre).

`tests/` zählt in keiner Stufe als Lesestelle: ein Flag, den nur ein Test
liest, ist im Produktionspfad tot.

Aufruf:
    python3 -B scripts/flag_wirkung_gate.py            # Report, Exit 0/1
    python3 -B scripts/flag_wirkung_gate.py --verbose  # alle Flags
    python3 -B scripts/flag_wirkung_gate.py --ci       # kompakt für Pre-Commit
    python3 -B scripts/flag_wirkung_gate.py --json     # maschinenlesbar
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FLAG_DATEI = Path("backend/core/music_model_flags.py")

# Verzeichnisse, die den Produktionspfad bilden (Stufe B).
PRODUKTIONS_VERZEICHNISSE: tuple[str, ...] = (
    "backend",
    "plugins",
    "denker",
    "Aurik10",
    "cli",
    "forensics",
)

# Mess-/Freigabe-Skripte, deren Kandidaten Stufe B erfüllen müssen (Stufe C).
MESS_SKRIPT_MUSTER: tuple[str, ...] = ("b3_*.py", "release_candidate_pipeline.py")

# Nicht-Quelltext: Modelle, Daten, Ausgaben, virtuelle Umgebungen.
AUSGESCHLOSSEN: frozenset[str] = frozenset(
    {
        ".git",
        ".venv_aurik",
        "venv",
        "models",
        "data",
        "output",
        "output_audio",
        "study_stimuli",
        "node_modules",
        "build",
        "dist",
        "__pycache__",
    }
)

# Bewusste, begründete Ausnahmen (fail-closed: ohne Eintrag = Verstoß).
# Jede Ausnahme nennt die Register-ID und den Grund; sie erscheint im Report.
AUSNAHMEN: dict[str, str] = {
    "use_bw_v5": (
        "D-K3-92: Der BW-v5-Pfad läuft in phase_06 über die NVSR-Route "
        "(`_use_nvsr`), nicht über diesen Schalter — es gibt keine "
        "Produktions-Lesestelle. W-8 ist damit kein messbarer B3-Kandidat, "
        "bis der Schalter verdrahtet ist (klangwirksam ⇒ nur über §4)."
    ),
}


@dataclass(frozen=True)
class Befund:
    """Ergebnis je Schalter."""

    flag: str
    produktionsstellen: tuple[str, ...]
    script_stellen: tuple[str, ...]
    ausnahme: str | None

    @property
    def hat_lesestelle(self) -> bool:
        return bool(self.produktionsstellen or self.script_stellen)

    @property
    def hat_produktionswirkung(self) -> bool:
        return bool(self.produktionsstellen)


def definiere_flags(datei: Path) -> list[str]:
    """Liest die `use_*`-Schalter aus der Flag-Datei (Reihenfolge bleibt stabil)."""
    baum = ast.parse(datei.read_text(encoding="utf-8"))
    flags: list[str] = []
    for knoten in baum.body:
        ziele: list[ast.expr] = []
        if isinstance(knoten, ast.AnnAssign):
            ziele = [knoten.target]
        elif isinstance(knoten, ast.Assign):
            ziele = list(knoten.targets)
        for ziel in ziele:
            if isinstance(ziel, ast.Name) and ziel.id.startswith("use_"):
                flags.append(ziel.id)
    return flags


def _lesestellen(datei: Path) -> set[str]:
    """Sammelt die in einer Datei tatsächlich gelesenen Flag-Namen.

    Gezählt werden Name-Loads, Attribut-Zugriffe, `getattr`/`hasattr`-Aufrufe mit
    Flag-Namen als String-Literal und Import-Aliase, deren lokaler Name
    anschließend geladen wird. Ein `setattr`-Aufruf ist eine Schreibstelle und
    zählt ausdrücklich nicht.
    """
    try:
        text = datei.read_text(encoding="utf-8", errors="replace")
        if "use_" not in text:
            return set()
        baum = ast.parse(text)
    except (SyntaxError, OSError) as exc:
        logger.warning("Überspringe nicht lesbare Datei %s (%s)", datei, type(exc).__name__)
        return set()

    gelesen: set[str] = set()
    for knoten in ast.walk(baum):
        if isinstance(knoten, ast.Name) and isinstance(knoten.ctx, ast.Load):
            if knoten.id.startswith("use_"):
                gelesen.add(knoten.id)
        elif isinstance(knoten, ast.Attribute) and knoten.attr.startswith("use_"):
            gelesen.add(knoten.attr)
        elif isinstance(knoten, ast.Call) and isinstance(knoten.func, ast.Name):
            # `getattr(modul, "use_x", default)` ist die häufigste Lesestelle im Repo;
            # `setattr` ist eine Schreibstelle und zählt ausdrücklich nicht.
            if knoten.func.id in {"getattr", "hasattr"} and len(knoten.args) >= 2:
                arg = knoten.args[1]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("use_"):
                    gelesen.add(arg.value)

    # Import-Alias: der Alias muss im Modul auch benutzt werden, sonst ist der
    # Import tot und der Schalter wird nie ausgewertet (D-K3-91-Muster).
    geladen = {
        knoten.id for knoten in ast.walk(baum) if isinstance(knoten, ast.Name) and isinstance(knoten.ctx, ast.Load)
    }
    for knoten in ast.walk(baum):
        if isinstance(knoten, ast.ImportFrom):
            for alias in knoten.names:
                if alias.name.startswith("use_") and (alias.asname or alias.name) in geladen:
                    gelesen.add(alias.name)
    return gelesen


def _quelltext_dateien(wurzel: Path) -> list[Path]:
    """Alle Python-Dateien außer Tests, Modellen, Daten und Umgebungen (sortiert)."""
    dateien = [
        pfad for pfad in wurzel.rglob("*.py") if not (set(pfad.parts) & AUSGESCHLOSSEN) and "tests" not in pfad.parts
    ]
    return sorted(dateien)


def _mess_kandidaten(wurzel: Path) -> dict[str, set[str]]:
    """Flag-Namen, die Mess-/Freigabe-Skripte als Kandidaten nennen (Stufe C)."""
    kandidaten: dict[str, set[str]] = {}
    skripte: list[Path] = []
    for muster in MESS_SKRIPT_MUSTER:
        skripte.extend(sorted((wurzel / "scripts").glob(muster)))
    for datei in sorted(set(skripte)):
        try:
            baum = ast.parse(datei.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError) as exc:
            logger.warning("Überspringe nicht lesbares Mess-Skript %s (%s)", datei, type(exc).__name__)
            continue
        for knoten in ast.walk(baum):
            if isinstance(knoten, ast.Constant) and isinstance(knoten.value, str):
                if knoten.value.startswith("use_"):
                    kandidaten.setdefault(knoten.value, set()).add(str(datei.relative_to(wurzel)))
    return kandidaten


def pruefe(
    wurzel: Path = ROOT,
    *,
    ausnahmen: dict[str, str] | None = None,
) -> list[Befund]:
    """Prüft alle Schalter und liefert die Befunde in Definitions-Reihenfolge."""
    ausnahmen = AUSNAHMEN if ausnahmen is None else ausnahmen
    flag_datei = wurzel / FLAG_DATEI
    if not flag_datei.is_file():
        raise FileNotFoundError(f"Flag-Datei fehlt: {flag_datei}")

    flags = definiere_flags(flag_datei)
    produktiv: dict[str, set[str]] = {f: set() for f in flags}
    messend: dict[str, set[str]] = {f: set() for f in flags}

    for datei in _quelltext_dateien(wurzel):
        if datei == flag_datei:
            continue
        gelesen = _lesestellen(datei)
        treffer = gelesen & set(flags)
        if not treffer:
            continue
        relativ = datei.relative_to(wurzel)
        ziel = produktiv if relativ.parts[0] in PRODUKTIONS_VERZEICHNISSE else messend
        for flag in treffer:
            ziel[flag].add(str(relativ))

    return [
        Befund(
            flag=flag,
            produktionsstellen=tuple(sorted(produktiv[flag])),
            script_stellen=tuple(sorted(messend[flag])),
            ausnahme=ausnahmen.get(flag),
        )
        for flag in flags
    ]


def verstoesse(
    befunde: list[Befund],
    *,
    kandidaten: dict[str, set[str]] | None = None,
) -> list[str]:
    """Formuliert die Verstöße (leer = Gate grün)."""
    meldungen: list[str] = []
    for befund in befunde:
        if befund.ausnahme:
            continue
        if not befund.hat_lesestelle:
            meldungen.append(
                f"{befund.flag}: keine Lesestelle außerhalb der Flag-Datei — der Schalter ist ein stiller "
                "No-Op (§V7 (copilot-instructions.md))"
            )
        elif not befund.hat_produktionswirkung:
            meldungen.append(
                f"{befund.flag}: nur in Mess-/Hilfsskripten gelesen "
                f"({', '.join(befund.script_stellen)}) — im Produktionspfad wirkungslos"
            )
    for flag, skripte in sorted((kandidaten or {}).items()):
        befund = next((b for b in befunde if b.flag == flag), None)
        if befund is None:
            continue
        if befund.ausnahme or befund.hat_produktionswirkung:
            continue
        meldungen.append(
            f"{flag}: als Mess-Kandidat genannt ({', '.join(sorted(skripte))}), "
            "hat aber keine Produktionswirkung — die Messung wäre ein No-Op-Artefakt"
        )
    return meldungen


def _report(befunde: list[Befund], *, verbose: bool) -> str:
    zeilen = [f"Flag-Wirkungs-Gate — {len(befunde)} Schalter geprüft"]
    for befund in befunde:
        if not verbose and befund.hat_produktionswirkung and not befund.ausnahme:
            continue
        status = "AUSNAHME" if befund.ausnahme else ("OK" if befund.hat_produktionswirkung else "TOT")
        prod = ", ".join(befund.produktionsstellen) or "—"
        skript = ", ".join(befund.script_stellen) or "—"
        zeilen.append(f"  [{status}] {befund.flag}: Produktion={prod} | Skripte={skript}")
        if befund.ausnahme:
            zeilen.append(f"           Grund: {befund.ausnahme}")
    return "\n".join(zeilen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prüft die Wirkung aller use_*-Schalter.")
    parser.add_argument("--verbose", action="store_true", help="alle Schalter auflisten")
    parser.add_argument("--ci", action="store_true", help="kompakte Ausgabe für Pre-Commit")
    parser.add_argument("--json", action="store_true", help="maschinenlesbarer Report")
    parser.add_argument("--wurzel", default=str(ROOT), help="Repo-Wurzel (Tests/Diagnose)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    wurzel = Path(args.wurzel)

    try:
        befunde = pruefe(wurzel)
        kandidaten = _mess_kandidaten(wurzel)
    except FileNotFoundError as exc:
        logger.error("Gate nicht ausführbar: %s", exc)
        return 2

    meldungen = verstoesse(befunde, kandidaten=kandidaten)

    if args.json:
        print(
            json.dumps(
                {
                    "schema": "aurik.flag_wirkung/1",
                    "flags": [
                        {
                            "flag": b.flag,
                            "produktionsstellen": list(b.produktionsstellen),
                            "script_stellen": list(b.script_stellen),
                            "ausnahme": b.ausnahme,
                        }
                        for b in befunde
                    ],
                    "kandidaten": {k: sorted(v) for k, v in sorted(kandidaten.items())},
                    "verstoesse": meldungen,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif not args.ci:
        print(_report(befunde, verbose=args.verbose))

    if meldungen:
        print("❌ Flag-Wirkungs-Gate: " + str(len(meldungen)) + " Verstoß/Verstöße", file=sys.stderr)
        for meldung in meldungen:
            print(f"   - {meldung}", file=sys.stderr)
        print(
            "   → Schalter verdrahten oder begründete Ausnahme (Register-ID) in AUSNAHMEN eintragen.",
            file=sys.stderr,
        )
        return 1

    if args.ci:
        print(f"Flag-Wirkungs-Gate: {len(befunde)} Schalter, alle wirksam (0 Verstöße)")
    else:
        print(f"\n✅ {len(befunde)} Schalter, alle wirksam (0 Verstöße)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
