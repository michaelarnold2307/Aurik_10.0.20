#!/usr/bin/env python3
"""Kandidaten-Archivierung — „schwache Kandidaten archivieren statt pflegen" (Pfeiler 3).

Roadmap Pfeiler 3: Whisper-Denoiser (D-K0-4, §9/5 der A2-Liste) und AERO
(D-K2-5, §9/6 dort) — **Archivierung ist ein SOTA-Urteil** und laut Katalog
`docs/reports/current/2026-10-25_sota_massnahmen_katalog.md` eine
**Sign-off-Entscheidung des Users/Maintainers**. Diese Kette ist die Mechanik:
ohne existierende Sign-off-Datei passiert nichts (fail-closed); im Trockenlauf
sowieso nichts.

Was die Ausführung tut (alles oder nichts, bei Fehlschlag Rollback):
  1. Register-Zeile zur ID auflösen; Artefakt-Pfade (Backticks) aufnehmen.
  2. Snapshot nach `models/_archive_<datum>/<id>_<name>/` (Kopie — das
     Deployte bleibt stehen; Archiv ist die sichere Variante, reversibel).
  3. Register-Buchung: Status ``geschlossen`` + Beleg (Archivpfad + Sign-off).
  4. Nach-Gates: `scripts/sota_deficit_gate.py` + `scripts/wohlklang_gate.py`.

Aufruf:
    python3 -B scripts/kandidaten_archiv.py --id D-K0-4            # Trockenlauf
    python3 -B scripts/kandidaten_archiv.py --id D-K0-4 --ausfuehren \\
        --signoff docs/reports/current/<signoff>.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

REGISTER = PROJECT_ROOT / ".github" / "SOTA_DEFICIT_REGISTER.md"


def finde_zeile(text: str, eintrag_id: str) -> tuple[int, str] | None:
    """Register-Zeile zur ID (genau eine)."""
    treffer = [(i, l) for i, l in enumerate(text.splitlines()) if l.startswith(f"| {eintrag_id} ")]
    return treffer[0] if len(treffer) == 1 else None


def artefakt_pfade(zeile: str, projekt_root: Path) -> list[Path]:
    """Existierende Pfade aus den Backticks der Zeile (Artefakte, keine Zitate)."""
    out: list[Path] = []
    for token in re.findall(r"`([^`]+)`", zeile):
        p = projekt_root / token
        if p.exists() and ("models/" in token or "scripts/" in token):
            out.append(p)
    return out


def archiv_plan(eintrag_id: str, *, projekt_root: Path) -> dict:
    """Trockener Plan: Quellen, Ziel, Register-Zeile. Schreibt nichts."""
    reg = projekt_root / ".github" / "SOTA_DEFICIT_REGISTER.md"
    text = reg.read_text(encoding="utf-8")
    ziel_row = finde_zeile(text, eintrag_id)
    if ziel_row is None:
        raise ValueError(f"keine eindeutige Register-Zeile für {eintrag_id}")
    idx, zeile = ziel_row
    cells = zeile.strip().strip("|").split("|")
    if len(cells) != 8:
        raise ValueError(f"Register-Zeile hat {len(cells)} statt 8 Zellen")
    if cells[5].strip() == "geschlossen":
        raise ValueError(f"{eintrag_id} ist bereits geschlossen")
    quellen = artefakt_pfade(zeile, projekt_root)
    if not quellen:
        raise ValueError(f"keine existierenden Artefakt-Pfade in der Zeile {eintrag_id}")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", cells[1].strip().strip("`")[:40])
    ziel = (
        projekt_root / "models" / f"_archive_{_dt.date.today().isoformat().replace('-', '')}" / f"{eintrag_id}_{name}"
    )
    return {
        "id": eintrag_id,
        "zeilen_index": idx,
        "zeile": zeile,
        "status_alt": cells[5].strip(),
        "quellen": [str(p.relative_to(projekt_root)) for p in quellen],
        "ziel": str(ziel.relative_to(projekt_root)),
    }


def fuehre_aus(
    plan: dict,
    *,
    projekt_root: Path,
    signoff: Path,
    gates: Callable[[], bool] | None = None,
) -> tuple[int, dict]:
    """Snapshot + Register-Buchung; Rollback bei Gate-Fehlschlag. Liefert (exit, plan)."""
    if not signoff.is_file():
        plan["fehler"] = f"Sign-off-Datei fehlt: {signoff} (§9/5–6 der A2-Liste, User)"
        return 2, plan
    reg = projekt_root / ".github" / "SOTA_DEFICIT_REGISTER.md"
    text_alt = reg.read_text(encoding="utf-8")
    ziel = projekt_root / plan["ziel"]
    ziel.mkdir(parents=True, exist_ok=True)
    try:
        for rel in plan["quellen"]:
            src = projekt_root / rel
            dst = ziel / src.name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        cells = plan["zeile"].strip().strip("|").split("|")
        cells[5] = " geschlossen "
        cells[7] = f" `{plan['ziel']}` + `{signoff}` (SOTA-Urteil §9/5–6 der A2-Liste, User-Sign-off) "
        zeilen = text_alt.splitlines()
        zeilen[plan["zeilen_index"]] = "|" + "|".join(cells) + "|"
        reg.write_text("\n".join(zeilen) + ("\n" if text_alt.endswith("\n") else ""), encoding="utf-8")

        ok = (gates or _gates)()
        if not ok:
            raise RuntimeError("Nach-Gate fehlgeschlagen")
        plan["ergebnis"] = "archiviert"
        return 0, plan
    except Exception as exc:
        log.warning("Archivierung abgebrochen (%s) — Rollback.", exc)
        reg.write_text(text_alt, encoding="utf-8")
        if ziel.exists():
            shutil.rmtree(ziel)
        plan["fehler"] = str(exc)
        return 1, plan


def _gates() -> bool:
    for gate in ("scripts/sota_deficit_gate.py", "scripts/wohlklang_gate.py"):
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / gate)], cwd=PROJECT_ROOT, timeout=600)
        if r.returncode != 0:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Schwache Kandidaten archivieren statt pflegen (Pfeiler 3)")
    ap.add_argument("--id", required=True, help="Register-ID, z.B. D-K0-4")
    ap.add_argument("--ausfuehren", action="store_true", help="sonst nur Trockenlauf")
    ap.add_argument("--signoff", type=Path, default=None, help="User-Sign-off (§9/5–6 der A2-Liste)")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    try:
        plan = archiv_plan(args.id, projekt_root=PROJECT_ROOT)
    except ValueError as exc:
        print(f"ABBRUCH: {exc}")
        return 2

    if not args.ausfuehren:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print("Trockenlauf — nichts verändert. Für die Ausführung: --ausfuehren --signoff <datei>")
        return 0

    if args.signoff is None:
        print("ABBRUCH: --ausfuehren verlangt --signoff (SOTA-Urteil ist User/ Maintainer, §9/5–6 der A2-Liste).")
        return 2
    code, plan = fuehre_aus(plan, projekt_root=PROJECT_ROOT, signoff=Path(args.signoff))
    report = args.report or (
        PROJECT_ROOT / "docs" / "reports" / "current" / f"kandidaten_archiv_{args.id.replace('-', '_')}.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Ergebnis: {plan.get('ergebnis') or plan.get('fehler')} — Report: {report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
