"""scripts/compliance/wettbewerbs_paritaet_gate.py — Pre-Commit-Gate (fail-closed).

Mission (2026-11-17): Aurik muss für das menschliche Gehör in jeder Disziplin
mindestens die Stufe von iZotope RX, Cedar und Acon erreichen. Dieses Gate
stellt sicher, dass die Paritäts-Verpflichtung **nie aus den Augen verloren**
wird — maschinell, bei jedem Commit:

  A) Das Paritäts-Register (.github/WETTBEWERBS_PARITAET.md) existiert und
     trägt ALLE Disziplin-Zeilen (Marker-Pin).
  B) Jede Disziplin-Zeile hat einen gültigen Status-Token
     (PARITAET/GESICHERT/LUECKE/OFFEN/NICHT_MESSBAR).
  C) GESICHERT-Zeilen verweisen auf ein Hör-Runbook; LUECKE-Zeilen tragen
     einen konkreten nächsten Schritt.
  D) Die Mission ist in SOTA_HANDOVER §0, AGENTS.md und Spec 01 verankert
     (Marker-Pin: „Wettbewerbs-Parität“).

Kein Mutator — nur Prüfung. Exit 1 bei jedem Verstoß.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

LEDGER = _ROOT / ".github" / "WETTBEWERBS_PARITAET.md"
HANDOVER = _ROOT / "docs" / "SOTA_HANDOVER.md"
AGENTS = _ROOT / "AGENTS.md"
SPEC01 = _ROOT / ".github" / "specs" / "01_musical_goals.md"

DISCIPLINEN = (
    "Declick",
    "Decrackle",
    "Denoise",
    "Declip",
    "Dehum",
    "Wow/Flutter",
    "HF-/Bandbreiten-Rekonstruktion",
    "Stereo-Imaging/Azimuth",
    "Dereverb",
    "Pitch/Speed-Korrektur",
    "Dynamik-Rückbau",
    "Phasen-/Kammfilter-Artefakte",
    "Dropout-/Riss-Reparatur",
    "Master-Finish/Lautheit",
    "Bedienung/Echtzeit",
)

_STATUS = {"PARITAET", "GESICHERT", "LUECKE", "OFFEN", "NICHT_MESSBAR"}


def _fehler(m: str, fehler: list[str]) -> None:
    fehler.append(m)


def main() -> int:
    fehler: list[str] = []
    if not LEDGER.is_file():
        _fehler("Wettbewerbs-Parität: .github/WETTBEWERBS_PARITAET.md fehlt", fehler)
        print("🔌 WETTBEWERBS-PARITAET — VERSTOSS:")
        for f in fehler:
            print(f"  ❌ {f}")
        return 1
    q = LEDGER.read_text(encoding="utf-8")
    for d in DISCIPLINEN:
        if f"| {d} " not in q and f"| {d} (" not in q:
            _fehler(f"Disziplin {d!r} fehlt im Paritäts-Register", fehler)
    for zeile in q.splitlines():
        if zeile.startswith("| ") and zeile.count("|") >= 4:
            for tok in _STATUS:
                if re.search(rf"\| {tok} \|", zeile):
                    break
            else:
                # Kopf-/Trennzeilen überspringen
                if "Amtsinhaber-Referenz" in zeile or set(zeile.strip()) <= {"|", "-", " "}:
                    continue
                _fehler(f"Zeile ohne gültigen Status-Token: {zeile[:70]}", fehler)
    if "GESICHERT" in q and "Runbook" not in q:
        _fehler("GESICHERT-Eintrag ohne Runbook-Verweis (Hör-Pfad fehlt)", fehler)
    if "LUECKE" in q and "Nächster Schritt" not in q:
        _fehler("LUECKE-Einträge ohne Schritt-Pflicht (Spalte „Nächster Schritt“ fehlt)", fehler)
    for pfad, name in ((HANDOVER, "SOTA_HANDOVER §0"), (AGENTS, "AGENTS.md"), (SPEC01, "Spec 01")):
        if not pfad.is_file():
            _fehler(f"{name}: Datei fehlt (Missions-Verankerung nicht prüfbar)", fehler)
            continue
        if "Wettbewerbs-Parität" not in pfad.read_text(encoding="utf-8"):
            _fehler(f"{name}: Missions-Marker „Wettbewerbs-Parität“ fehlt — Ziel nicht verankert", fehler)
    if fehler:
        print("🔌 WETTBEWERBS-PARITAET — VERSTOSS:")
        for f in fehler:
            print(f"  ❌ {f}")
        return 1
    print("🔌 WETTBEWERBS-PARITAET — ✅ alle Disziplinen geführt, Mission verankert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
