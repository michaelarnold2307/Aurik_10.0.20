#!/usr/bin/env python3
"""Pfad-Wirksamkeits-Gate für B3-Stufenmessungen (Muster C, W-3-Lehre).

Befund (Report `docs/reports/current/2026-11-06_w3_pfadklaerung.md`,
Commit `c0ff1a26`): Die W-3-Nachmessung 2026-11-05 maß Δ ≈ 0 und die Stufe
wurde formal ABGELEHNT — obwohl der gemessene Pfad SGMSE+ Tier-0 NIE
erreichte (Dauer-Achsen-Defekt in `phase_49_advanced_dereverb.py`). Eine
Messung ohne Wirksamkeits-Nachweis ist kein Qualitätsurteil über das Modell,
sondern ein Mess-Artefakt.

Dieses Modul liefert den Nachweis generisch und ohne phasen-individuelle
Schwellwerte (§V7 (copilot-instructions.md)):

(a) Flag-Readback: der Harness muss den Schalter nachweislich gesetzt haben;
(b) Wirkungs-Fußabdruck: sind AN- und AUS-Ausgabe in ALLEN Fenstern
    bit-identisch, hatte der Flag-Flip nirgends Wirkung ⇒ Messung ungültig
    (fail-closed);
(c) optionaler Modellzeuge je Stufe (deklarative Tabelle) — nur VERIFIZIERTE
    Metadaten-Schlüssel eintragen, kein Raten (§V6 (copilot-instructions.md));
    ohne Zeugen-Eintrag trägt das Fußabdruck-Gate (b).
"""

from __future__ import annotations

import numpy as np

# (c) Deklarative Zeugen-Tabelle — nur verifizierte Schlüssel:
# W-1: "hr_v1"-Witness in den Phasen-Metadaten (attempted/applied),
#      belegt in phase_03_denoise.py / phase_07_harmonic_restoration.py /
#      phase_23_spectral_repair.py / phase_50_spectral_repair.py.
# W-2..W-7: noch kein verifizierter Einzel-Schlüssel im Phasen-Metadaten-Raum
#      — dort trägt ausschließlich das Fußabdruck-Gate (b). Neue Einträge nur
#      mit belegtem Metadaten-Schlüssel ergänzen.
PFAD_ZEUGEN: dict[str, dict] = {
    "W-1": {"schluessel": "hr_v1", "aktiv_praedikat": "attempted_or_applied"},
}


def fussabdruck(an_audio, aus_audio) -> float:
    """RMS der AN-minus-AUS-Differenz; 0.0 == bit-identisch (layout-sicher)."""
    a = np.asarray(an_audio, dtype=np.float64).ravel()
    b = np.asarray(aus_audio, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("inf")
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def zeugnis(metadata: dict, stufe: str) -> dict:
    """Extrahiert den Modellzeugen einer Seite aus den Restore-Metadaten.

    Rückgabe: {"vorhanden": bool|None, "wert": ..., "aktiv": bool|None}.
    `vorhanden=None` bedeutet: für diese Stufe ist kein Zeuge konfiguriert
    (das Fußabdruck-Gate trägt) — KEIN Fehlalarm.
    """
    konf = PFAD_ZEUGEN.get(stufe)
    if konf is None:
        return {"vorhanden": None, "wert": None, "aktiv": None}
    wert = metadata.get(konf["schluessel"])
    if wert is None:
        return {"vorhanden": False, "wert": None, "aktiv": False}
    aktiv = None
    if konf["aktiv_praedikat"] == "attempted_or_applied" and isinstance(wert, dict):
        aktiv = bool(wert.get("attempted") or wert.get("applied"))
    return {"vorhanden": True, "wert": wert, "aktiv": aktiv}


def pfad_wirksam_ok(zeilen: list[dict]) -> tuple[bool, dict]:
    """Fail-closed-Gate über alle Fenster einer Stufenmessung.

    `zeilen`: Einträge mit `flag_an_gelesen`, `flag_aus_gelesen`,
    `fussabdruck_rms`, `zeuge_an`, `zeuge_aus` (je Fenster).
    """
    probleme: list[str] = []
    fuss_max = max((float(z.get("fussabdruck_rms", 0.0)) for z in zeilen), default=0.0)
    fuss_gesamt = sum(float(z.get("fussabdruck_rms", 0.0)) for z in zeilen)
    if fuss_gesamt == 0.0:
        probleme.append("fussabdruck_null: AN==AUS bit-identisch in allen Fenstern (W-3-Lehre)")
    for z in zeilen:
        if z.get("flag_an_gelesen") is not True:
            probleme.append(f"{z.get('trial_id', '?')}: Flag auf AN-Seite nicht als aktiv gelesen")
            break
        if z.get("flag_aus_gelesen") is not False:
            probleme.append(f"{z.get('trial_id', '?')}: Flag auf AUS-Seite nicht als inaktiv gelesen")
            break
    for z in zeilen:
        za = z.get("zeuge_an") or {}
        if za.get("vorhanden") is False:
            probleme.append(f"{z.get('trial_id', '?')}: Modellzeuge auf AN-Seite fehlt")
            break
        if "aktiv" in za and za.get("aktiv") is not None and za["aktiv"] is not True:
            probleme.append(f"{z.get('trial_id', '?')}: Modellzeuge auf AN-Seite inaktiv")
            break
    ok = not probleme
    return ok, {
        "pfad_wirksam": ok,
        "fussabdruck_max_rms": fuss_max,
        "probleme": probleme,
    }
