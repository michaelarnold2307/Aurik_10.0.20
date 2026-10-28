#!/usr/bin/env python3
"""Flag-Freigabe-Kette — Verdikt → Register-Buchung → Flag-Flip → Commit (Pfeiler 1).

Roadmap: „Verdikt → Flag-Flip in einem Kommando ... Ziel: ≤ 10 min vom Hören
zum Commit" (docs/TODOS_SOTA_ROADMAP.md). Diese Kette ist das Werkzeug — die
HÖR-ENTSCHEIDUNG selbst bleibt beim Menschen (§v10.802 (copilot-instructions.md):
Aktivierung nur bei Hör-Sign-off). Ohne existierende Sign-off-Datei verweigert
die Kette die Buchung (fail-closed).

Ablauf (alles oder nichts, bei Fehlschlag Worktree-Rücksetzung):
  1. Verdikt validieren: ``fit.json`` (scripts/mushra_harness.py thresholds-fit)
     oder ab_harness-Statistik (``aurik.ab_stats/1``). Nur belegte Vorteile
     werden gebucht; ZU_WENIGE_TRIALS/UNENTSCHEIDEN sind ehrliche Ablehnungen.
  2. Hör-Sign-off-Datei verlangen (§v10.802 (copilot-instructions.md)).
  3. Register-Buchung in ``.github/SOTA_DEFICIT_REGISTER.md``: D-*-Zeile zum
     Flag → Status ``geschlossen`` + Beleg-Pfade (8 Zellen bleiben 8 Zellen).
  4. Flag-Flip in ``backend/core/music_model_flags.py`` (False → True + Beleg).
  5. Test-Angleichung: ``use_xxx is False``-Assertions werden mitgezogen.
  6. Nach-Gates: ``scripts/sota_deficit_gate.py`` + ``scripts/wohlklang_gate.py``.
  7. ``--commit`` erzeugt den Conventional-Commit (Hooks laufen alle Gates).

Aufruf: python3 -B scripts/flag_freigabe.py --flag use_x \\
    --verdikt <fit.json|ab-stats.json> --signoff <datei.md> [--commit]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

REGISTER = PROJECT_ROOT / ".github" / "SOTA_DEFICIT_REGISTER.md"
FLAGS_PY = PROJECT_ROOT / "backend" / "core" / "music_model_flags.py"

BELEGTE_VERDIKTE = ("HOERBAR_BELEGT", "PRAEFERENZ_BELEGT")


def verdikt_ok(data: dict) -> tuple[bool, str]:
    """Nur belegte Vorteile sind buchbar — alles andere ist eine ehrliche Ablehnung."""
    if "verdict" in data:
        verdict = str(data.get("verdict"))
        pooled = data.get("pooled", {}) or {}
        if verdict not in BELEGTE_VERDIKTE:
            return False, f"Verdikt {verdict!r} ist kein Beleg (erlaubt: {BELEGTE_VERDIKTE})"
        if not pooled.get("above_chance_ci95"):
            return False, "95 %-CI überschreitet den Zufallspegel nicht (above_chance_ci95=False)"
        if not pooled.get("trials_sufficient"):
            return False, "Mindestzahl Trials nicht erreicht (trials_sufficient=False)"
        return True, f"fit.json: {verdict} (n={pooled.get('trials')})"
    if str(data.get("schema", "")).startswith("aurik.ab_stats"):
        if not data.get("signifikant"):
            return False, "ab_harness: nicht signifikant (signifikant=False)"
        if (data.get("siege_a") or 0) == (data.get("siege_b") or 0):
            return False, "ab_harness: ausgeglichenes Bild — kein Vorteil"
        return True, f"ab_harness: signifikant (p={data.get('p_wert_vorzeichen_test')})"
    return False, "unbekanntes Verdikt-Schema (weder fit.json noch aurik.ab_stats/1)"


def objektivbeschluss_ok(data: dict, flag: str) -> tuple[bool, str]:
    """Formeller Objektivmetrik-Beschluss (Wohlklang-Vertrag, Abschnitt B3) — fail-closed."""
    if str(data.get("schema")) != "aurik.objektivbeschluss/1":
        return False, "Schema unbekannt (erwartet aurik.objektivbeschluss/1)"
    if str(data.get("flag")) != flag:
        return False, f"Beschluss gilt fuer {data.get('flag')}, nicht fuer {flag}"
    if int(data.get("referenz_songs") or 0) < 3:
        return False, "referenz_songs < 3 (Vertrag: mindestens 3 echte Songs)"
    if data.get("never_worsen") is not True:
        return False, "never_worsen nicht bestaetigt (G-4)"
    if data.get("signifikant") is not True:
        return False, "Vorteil nicht signifikant (G-1/G-2/G-3)"
    if int(data.get("defekt_fallbacks", -1)) != 0:
        return False, "defekt_fallbacks != 0 (B1)"
    if data.get("blind_ausgewertet") is not True:
        return False, "Auswertung nicht blind (Vertragsbedingung 5)"
    messung = str(data.get("messung") or "")
    if not messung:
        return False, "Mess-Report nicht benannt (Provenienz)"
    mp = Path(messung) if Path(messung).is_absolute() else PROJECT_ROOT / messung
    if not mp.is_file():
        return False, f"Mess-Report fehlt: {messung}"
    ziel = (data.get("zielmetrik") or {}).get("name", "?")
    return True, f"Objektivbeschluss gueltig: {ziel} auf {data.get('referenz_songs')} Songs (Hoer-Vorbehalt)"


def finde_register_zeile(text: str, flag: str) -> tuple[int, str] | None:
    """Die D-*-Zeile, die dieses Flag trägt (genau eine offene erwartet)."""
    treffer = [
        (i, line)
        for i, line in enumerate(text.splitlines())
        if line.startswith("| D-") and flag in line and "geschlossen" not in line.split("|")[6]
    ]
    if len(treffer) != 1:
        return None
    return treffer[0]


def buche_register(text: str, flag: str, verdikt: Path, signoff: Path, verfahren: str = "H\u00f6r-Sign-off") -> str:
    """Status → geschlossen, Beleg → Verdikt + Sign-off. 8 Zellen bleiben 8 Zellen."""
    ziel = finde_register_zeile(text, flag)
    if ziel is None:
        raise ValueError(f"keine eindeutige offene Register-Zeile für {flag}")
    idx, line = ziel
    cells = line.strip().strip("|").split("|")
    if len(cells) != 8:
        raise ValueError(f"Register-Zeile hat {len(cells)} statt 8 Zellen: {line[:80]}")
    cells[5] = " geschlossen "
    cells[7] = f" `{verdikt}` + `{signoff}` ({verfahren}) "
    zeilen = text.splitlines()
    zeilen[idx] = "|" + "|".join(cells) + "|"
    return "\n".join(zeilen) + ("\n" if text.endswith("\n") else "")


def flippe_flag(text: str, flag: str, beleg_name: str, verfahren: str = "H\u00f6r-Sign-off") -> str:
    """use_x: bool = False → True; Kommentar trägt den Beleg (kein stiller Wandel)."""
    muster = re.compile(rf"^({re.escape(flag)}: bool = )False(\s*#.*)$", re.MULTILINE)
    if not muster.search(text):
        raise ValueError(f"{flag}: bool = False nicht gefunden (schon aktiv oder anders benannt)")
    heute = _dt.date.today().isoformat()
    ersatz = rf"\g<1>True\g<2> — AKTIV nach {verfahren} (Flag-Flip {heute}, Beleg: {beleg_name})"
    return muster.sub(ersatz, text, count=1)


def gleiche_tests_an(text: str, flag: str) -> tuple[str, int]:
    """``use_x is False``-Assertions werden mitgezogen (Regeländerung ⇒ Test nachziehen)."""
    muster = re.compile(rf"(\b\w+\.)?{re.escape(flag)}(\s+is\s+)False\b")
    n = len(muster.findall(text))
    return muster.sub(rf"\g<1>{flag}\g<2>True", text), n


def _p(repo_pfad: str) -> Path:
    p = Path(repo_pfad)
    return p if p.is_absolute() else PROJECT_ROOT / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verdikt → Register-Buchung → Flag-Flip → Commit")
    ap.add_argument("--flag", required=True, help="Flag-Name, z.B. use_aspade_declipper")
    ap.add_argument("--verdikt", required=True, help="fit.json (mushra_harness) oder ab_harness-Statistik")
    ap.add_argument("--signoff", default=None, help="Hör-Sign-off-Datei")
    ap.add_argument(
        "--objektivbeschluss", default=None, help="Formeller Objektivmetrik-Beschluss (JSON, Wohlklang-Vertrag B3)"
    )
    ap.add_argument("--commit", action="store_true", help="anschließend committen (sonst nur vorbereitet)")
    args = ap.parse_args(argv)

    flag = args.flag
    verdikt = _p(args.verdikt)
    signoff = _p(args.signoff) if args.signoff else Path("/dev/null")

    if not verdikt.is_file():
        print(f"ABBRUCH: Verdikt-Datei fehlt: {verdikt}")
        return 2
    if args.objektivbeschluss is not None:
        beschluss = _p(args.objektivbeschluss)
        if not beschluss.is_file():
            print(f"ABBRUCH: Objektivbeschluss fehlt: {beschluss}")
            return 2
        ok_b, grund_b = objektivbeschluss_ok(json.loads(beschluss.read_text(encoding="utf-8")), flag)
        if not ok_b:
            print(f"ABBRUCH: Objektivbeschluss ungueltig — {grund_b}")
            return 2
        verfahren = "Objektivbeschluss B3, Hör-Vorbehalt"
        signoff = beschluss
        print(f"Objektivbeschluss: {grund_b}")
    elif signoff is not None and signoff.is_file():
        verfahren = "Hör-Sign-off"
    else:
        print("ABBRUCH: entweder --signoff (Hör-Abnahme) oder --objektivbeschluss (Vertrag B3) erforderlich.")
        return 2
    ok, grund = verdikt_ok(json.loads(verdikt.read_text(encoding="utf-8")))
    if not ok:
        print(f"ABBRUCH: Verdikt nicht buchbar — {grund}")
        return 2

    originals = {
        REGISTER: REGISTER.read_text(encoding="utf-8"),
        FLAGS_PY: FLAGS_PY.read_text(encoding="utf-8"),
    }
    test_dateien: dict[Path, str] = {}
    for tp in sorted((PROJECT_ROOT / "tests").rglob("*.py")):
        t = tp.read_text(encoding="utf-8")
        if f"{flag} is False" in t or f"{flag}  is False" in t:
            test_dateien[tp] = t

    try:
        neu_reg = buche_register(originals[REGISTER], flag, verdikt, signoff, verfahren=verfahren)
        neu_flags = flippe_flag(originals[FLAGS_PY], flag, verdikt.name, verfahren=verfahren)
        n_tests = 0
        neue_tests: dict[Path, str] = {}
        for tp, t in test_dateien.items():
            nt, n = gleiche_tests_an(t, flag)
            neue_tests[tp] = nt
            n_tests += n

        REGISTER.write_text(neu_reg, encoding="utf-8")
        FLAGS_PY.write_text(neu_flags, encoding="utf-8")
        for tp, nt in neue_tests.items():
            tp.write_text(nt, encoding="utf-8")
        print(f"Register gebucht, Flag geflippt, {n_tests} Test-Assertion(s) angeglichen.")
        print(f"Beleg: {grund} | Sign-off: {signoff}")

        for gate in ("scripts/sota_deficit_gate.py", "scripts/wohlklang_gate.py"):
            r = subprocess.run([sys.executable, str(PROJECT_ROOT / gate)], cwd=PROJECT_ROOT, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"Nach-Gate fehlgeschlagen: {gate}")
        print("Nach-Gates grün (sota_deficit_gate, wohlklang_gate).")
    except Exception as exc:
        log.warning("Flag-Freigabe abgebrochen (%s) — Worktree wird zurückgesetzt.", exc)
        REGISTER.write_text(originals[REGISTER], encoding="utf-8")
        FLAGS_PY.write_text(originals[FLAGS_PY], encoding="utf-8")
        for tp, t in test_dateien.items():
            tp.write_text(t, encoding="utf-8")
        print(f"ABBRUCH (Worktree zurückgesetzt): {exc}")
        return 1

    if args.commit:
        r = subprocess.run(
            ["git", "add", "-A"],
            cwd=PROJECT_ROOT,
            timeout=120,
        )
        if r.returncode != 0:
            print("ABBRUCH: git add fehlgeschlagen")
            return 1
        msg = f"feat(ml-sota): Flag-Flip {flag} — Hör-Sign-off gebucht ({verdikt.name})"
        body = (
            f"Verdikt: {grund}\nBeleg: {verdikt}\nHör-Sign-off (§v10.802 (copilot-instructions.md)): {signoff}\n"
            f"Register: Status geschlossen, Beleg-Pfade gebucht.\n"
        )
        r = subprocess.run(["git", "commit", "-m", msg, "-m", body], cwd=PROJECT_ROOT, timeout=900)
        if r.returncode != 0:
            print("ABBRUCH: git commit fehlgeschlagen (Hooks) — Änderungen bleiben im Worktree.")
            return 1
        print("Commit erzeugt.")
    else:
        print("Vorbereitet (ohne --commit): Änderungen im Worktree prüfen und committen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
