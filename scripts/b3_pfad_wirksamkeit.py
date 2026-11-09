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
    ohne Zeugen-Eintrag trägt das Fußabdruck-Gate (b);
(d) Modell-Zeuge für Modell-Stufen (2026-11-09, D-K3-85): bei `M12-WF4` und
    `M12-TPV2` ist das Fußabdruck-Gate (b) blind — die Pipeline ist nicht
    bit-reproduzierbar, also liefert sie auch bei totem Modellpfad ein
    Fußabdruck-Rauschen > 0. Der Zeuge lädt das Modell und meldet, ob das
    TRAINIERTE Artefakt die Quelle ist (`checkpoint:…`), nicht Seed-Gewichte
    und nicht „nicht geladen".
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ── Modell-Zeugen (Stufen, deren Wirkung ein trainiertes Artefakt ist) ───────


def wf4_modell_zeugnis() -> dict:
    """Lädt den WF-V4-Warp-Head und meldet, ob ein trainiertes Artefakt trägt.

    `aktiv=False` heißt: der Flag-Pfad ist tot (kein ONNX, kein Checkpoint) —
    dann kann eine AN/AUS-Messung das Modell nicht beurteilen.
    """
    from backend.core.dsp.wf4_warp_head import WF4WarpHead

    head = WF4WarpHead()
    verfuegbar = bool(head.is_available())
    return {
        "vorhanden": verfuegbar,
        "wert": {"quelle": head.source},
        "aktiv": verfuegbar,
    }


def tp_v2_modell_zeugnis() -> dict:
    """Lädt den TP-V2-Phasenschätzer und prüft die Herkunft der Gewichte.

    `source == "seed"`/`"seed_fallback"` ⇒ untrainierte Gewichte: die Stufe
    ist dann NICHT das gemessene Modell (2026-11-05/08: `_PROJECT_ROOT` zeigte
    auf `backend/`, der Default-Checkpoint existierte nie).
    """
    from backend.core.dsp.tp_v2_phase import DEFAULT_CHECKPOINT, TPv2PhaseEstimator

    est = TPv2PhaseEstimator(checkpoint=DEFAULT_CHECKPOINT)
    quelle = str(est.source)
    trainiert = quelle.startswith("checkpoint:")
    return {
        "vorhanden": bool(DEFAULT_CHECKPOINT.is_file()),
        "wert": {"quelle": quelle, "checkpoint": str(DEFAULT_CHECKPOINT)},
        "aktiv": trainiert,
    }


# (c) Deklarative Zeugen-Tabelle — nur verifizierte Schlüssel:
# W-1: "hr_v1"-Witness in den Phasen-Metadaten (attempted/applied),
#      belegt in phase_03_denoise.py / phase_07_harmonic_restoration.py /
#      phase_23_spectral_repair.py / phase_50_spectral_repair.py.
# W-2..W-7: noch kein verifizierter Einzel-Schlüssel im Phasen-Metadaten-Raum
#      — dort trägt ausschließlich das Fußabdruck-Gate (b). Neue Einträge nur
#      mit belegtem Metadaten-Schlüssel ergänzen.
# M12-WF4/M12-TPV2 (d): Modell-Zeuge statt Metadaten-Schlüssel, weil die
#      Wirkung ein Artefakt ist (D-K3-85).
PFAD_ZEUGEN: dict[str, dict] = {
    "W-1": {"schluessel": "hr_v1", "aktiv_praedikat": "attempted_or_applied"},
    "M12-WF4": {"modell": wf4_modell_zeugnis},
    "M12-TPV2": {"modell": tp_v2_modell_zeugnis},
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
    modell_fn = konf.get("modell")
    if modell_fn is not None:
        try:
            status = dict(modell_fn())
        except Exception as exc:  # pragma: no cover - Systemgrenze (Modell-Load)
            logger.warning("Pfad-Zeuge '%s' nicht abfragbar: %s", stufe, exc)
            return {"vorhanden": False, "wert": {"fehler": str(exc)[:200]}, "aktiv": False}
        status.setdefault("vorhanden", None)
        status.setdefault("aktiv", None)
        return status
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
