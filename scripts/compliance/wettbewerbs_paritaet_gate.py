"""scripts/compliance/wettbewerbs_paritaet_gate.py — Pre-Commit-Gate (fail-closed).

Mission (2026-11-17): Aurik muss für das menschliche Gehör in jeder Disziplin
mindestens die Stufe von iZotope RX, Cedar und Acon erreichen. Dieses Gate
stellt sicher, dass die Paritäts-Verpflichtung **nie aus den Augen verloren**
wird — maschinell, bei jedem Commit:

  A) Das Paritäts-Register (.github/WETTBEWERBS_PARITAET.md) existiert und
     trägt ALLE Disziplin-Zeilen (Marker-Pin).
  B) Jede Disziplin-Zeile hat einen gültigen Status-Token
     (PARITAET/GESICHERT/LUECKE/OFFEN/NICHT_MESSBAR).
  C) GESICHERT-Zeilen verweisen auf ein zeileneigenes Hör-Runbook und einen
     maschinengeprüften Kriterium-(a)- oder Kriterium-(b)-Beleg; LUECKE-Zeilen
     tragen einen konkreten nächsten Schritt.
  D) Die Mission ist in SOTA_HANDOVER §0, AGENTS.md und Spec 01 verankert
     (Marker-Pin: „Wettbewerbs-Parität“).
  E) PARITAET-Zeilen verlinken einen validierten DRT-16-Beleg mit RX-12-
     Identität, Holdout, Nichtunterlegenheit und bestandenem Hörpanel.

Kein Mutator — nur Prüfung. Exit 1 bei jedem Verstoß.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent

LEDGER = _ROOT / ".github" / "WETTBEWERBS_PARITAET.md"
HANDOVER = _ROOT / "docs" / "SOTA_HANDOVER.md"
AGENTS = _ROOT / "AGENTS.md"
SPEC01 = _ROOT / ".github" / "specs" / "01_musical_goals.md"
SPEC26 = _ROOT / ".github" / "specs" / "26_defekt_reparatur_trainingsvertrag.md"

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
_INTERNAL_REPORT_SCHEMA = "aurik.schadensbewertung/1"


def _fehler(m: str, fehler: list[str]) -> None:
    fehler.append(m)


def _zahlenpaar(wert: Any) -> tuple[float, float] | None:
    if not isinstance(wert, list) or len(wert) != 2:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in wert):
        return None
    return float(wert[0]), float(wert[1])


def _sha256_gultig(wert: Any) -> bool:
    return isinstance(wert, str) and re.fullmatch(r"[0-9a-fA-F]{64}", wert) is not None


def _finite_float(wert: Any) -> float | None:
    if isinstance(wert, (int, float)) and not isinstance(wert, bool) and math.isfinite(wert):
        return float(wert)
    return None


def _finite_zahl(wert: Any) -> bool:
    return _finite_float(wert) is not None


def _ci95_aus_deltas(values: list[float]) -> tuple[float, float] | None:
    """Berechnet das zweiseitige Student-t-CI des gepaarten Mittelwerts."""
    n = len(values)
    if n < 2:
        return None
    from scipy.stats import t as student_t

    mean = math.fsum(values) / n
    squared_deviation_sum = math.fsum((value - mean) ** 2 for value in values)
    standard_error = math.sqrt(squared_deviation_sum / (n * (n - 1)))
    critical_value = float(student_t.ppf(0.975, n - 1))
    margin = critical_value * standard_error
    return mean - margin, mean + margin


def _ci_ist_reproduzierbar(
    reported: tuple[float, float] | None,
    values: list[float] | None,
) -> bool:
    if reported is None or values is None:
        return False
    calculated = _ci95_aus_deltas(values)
    return calculated is not None and all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6) for actual, expected in zip(reported, calculated)
    )


def _delta_values(wert: Any, n_titles: int) -> list[float] | None:
    if not isinstance(wert, list) or len(wert) != n_titles:
        return None
    if not all(_finite_zahl(value) for value in wert):
        return None
    return [float(value) for value in wert]


def _paar_deltas_gebunden(
    pairs: Any,
    mos_values: list[float] | None,
    defect_values: list[float] | None,
) -> bool:
    if not isinstance(pairs, list) or mos_values is None or defect_values is None or len(pairs) != len(mos_values):
        return False
    if not all(
        isinstance(pair, dict)
        and _finite_zahl(pair.get("musik_mos_delta"))
        and _finite_zahl(pair.get("defektlast_delta"))
        for pair in pairs
    ):
        return False
    return all(
        math.isclose(float(pair["musik_mos_delta"]), mos_values[index], rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(pair["defektlast_delta"]), defect_values[index], rel_tol=0.0, abs_tol=1e-12)
        for index, pair in enumerate(pairs)
    )


def _pruefe_paritaets_beleg(pfad: str, disziplin: str, fehler: list[str]) -> tuple[bool, bool, str]:
    start_fehler = len(fehler)
    rel = Path(pfad)
    if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".json":
        _fehler(f"{disziplin}: unsicherer oder ungültiger RX-12-Belegpfad {pfad!r}", fehler)
        return False, False, ""
    if not pfad.startswith("docs/reports/current/"):
        _fehler(f"{disziplin}: RX-12-Beleg liegt nicht unter docs/reports/current/: {pfad}", fehler)
        return False, False, ""
    datei = _ROOT / rel
    try:
        beleg = json.loads(datei.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fehler(f"{disziplin}: RX-12-Beleg nicht lesbar ({pfad}): {exc}", fehler)
        return False, False, ""
    if not isinstance(beleg, dict):
        _fehler(f"{disziplin}: RX-12-Beleg muss ein JSON-Objekt sein: {pfad}", fehler)
        return False, False, ""
    if beleg.get("schema") != "aurik.rx12_parity/1":
        _fehler(f"{disziplin}: falsches RX-12-Belegschema in {pfad}", fehler)
    if beleg.get("discipline") != disziplin:
        _fehler(f"{disziplin}: Disziplin im RX-12-Beleg stimmt nicht überein: {pfad}", fehler)
    if beleg.get("decision") != "PARITAET":
        _fehler(f"{disziplin}: RX-12-Beleg ohne PARITAET-Entscheidung: {pfad}", fehler)

    reference = beleg.get("reference")
    if (
        not isinstance(reference, dict)
        or not all(
            isinstance(reference.get(key), str) and reference[key].strip()
            for key in ("product", "version", "build", "module")
        )
        or not _sha256_gultig(reference.get("settings_sha256") if isinstance(reference, dict) else None)
    ):
        _fehler(f"{disziplin}: RX-12-Beleg ohne vollständige Referenzidentität: {pfad}", fehler)
        reference = {}
    product = str(reference.get("product", ""))
    version = str(reference.get("version", ""))
    is_rx12 = (
        "izotope" in product.casefold()
        and re.search(r"\brx\b", product, re.IGNORECASE) is not None
        and re.search(r"\b12(?:\D|$)", version) is not None
    )
    if "rx" in product.casefold() and not is_rx12:
        _fehler(f"{disziplin}: RX-Referenz ist nicht als RX 12 ausgewiesen: {pfad}", fehler)
    aurik = beleg.get("aurik")
    if (
        not isinstance(aurik, dict)
        or not all(
            isinstance(aurik.get(key), str) and aurik[key].strip() for key in ("version", "commit", "production_path")
        )
        or not isinstance(aurik.get("active_flags"), list)
        or not _sha256_gultig(aurik.get("settings_sha256") if isinstance(aurik, dict) else None)
    ):
        _fehler(f"{disziplin}: RX-12-Beleg ohne Aurik-Produktionspfad/Version: {pfad}", fehler)
    dataset = beleg.get("dataset")
    n_titles = dataset.get("n_titles", 0) if isinstance(dataset, dict) else 0
    pairs = dataset.get("pairs") if isinstance(dataset, dict) else None
    pairs_valid = (
        isinstance(pairs, list)
        and len(pairs) == n_titles
        and all(
            isinstance(pair, dict)
            and isinstance(pair.get("source_id"), str)
            and pair["source_id"].strip()
            and _sha256_gultig(pair.get("input_pcm_sha256_aurik"))
            and pair.get("input_pcm_sha256_aurik") == pair.get("input_pcm_sha256_reference")
            and _finite_zahl(pair.get("musik_mos_delta"))
            and _finite_zahl(pair.get("defektlast_delta"))
            for pair in pairs
        )
        and len({pair["source_id"] for pair in pairs}) == n_titles
    )
    if (
        not isinstance(dataset, dict)
        or not _sha256_gultig(dataset.get("paired_input_manifest_sha256"))
        or dataset.get("holdout") is not True
        or not isinstance(n_titles, int)
        or isinstance(n_titles, bool)
        or n_titles < 10
        or not pairs_valid
    ):
        _fehler(f"{disziplin}: RX-12-Beleg ohne ≥10 bit-identische Holdout-Paare: {pfad}", fehler)
    objective = beleg.get("objective")
    if not isinstance(objective, dict):
        _fehler(f"{disziplin}: RX-12-Beleg ohne objektive Nichtunterlegenheit: {pfad}", fehler)
        return False, is_rx12, product.casefold()
    jnd = objective.get("jnd")
    mos_ci = _zahlenpaar(objective.get("musik_mos_delta_ci95"))
    defect_ci = _zahlenpaar(objective.get("defektlast_delta_ci95"))
    mos_values = _delta_values(objective.get("musik_mos_delta_by_title"), n_titles)
    defect_values = _delta_values(objective.get("defektlast_delta_by_title"), n_titles)
    deltas_valid = mos_values is not None and defect_values is not None
    paired_deltas_match = pairs_valid and _paar_deltas_gebunden(pairs, mos_values, defect_values)
    if not paired_deltas_match:
        _fehler(
            f"{disziplin}: Einzeltitel-Deltas stimmen nicht mit den gehashten Quellenpaaren überein: {pfad}", fehler
        )
    ci_reproducible = _ci_ist_reproduzierbar(mos_ci, mos_values) and _ci_ist_reproduzierbar(defect_ci, defect_values)
    if not ci_reproducible:
        _fehler(
            f"{disziplin}: RX-12-Konfidenzintervalle stimmen nicht mit der Student-t-Neuberechnung überein: {pfad}",
            fehler,
        )
    if (
        not isinstance(jnd, (int, float))
        or isinstance(jnd, bool)
        or not math.isfinite(jnd)
        or jnd <= 0
        or objective.get("jnd_source") != "wohlklang_zeuge.JND_TOLERANZ"
        or mos_ci is None
        or defect_ci is None
        or not deltas_valid
        or not paired_deltas_match
        or not ci_reproducible
        or objective.get("noninferiority_passed") is not True
        or mos_ci[0] < -jnd
        or defect_ci[1] > jnd
    ):
        _fehler(f"{disziplin}: RX-12-Nichtunterlegenheitsgrenzen verfehlt/fehlen: {pfad}", fehler)
    panel = beleg.get("panel")
    if (
        not isinstance(panel, dict)
        or panel.get("design") != "MUSHRA"
        or not isinstance(panel.get("n_listeners"), int)
        or isinstance(panel.get("n_listeners"), bool)
        or panel["n_listeners"] < 30
        or panel.get("blind") is not True
        or panel.get("randomized") is not True
        or panel.get("level_matched") is not True
        or panel.get("verdict") != "nicht_unterlegen"
        or panel.get("signoff") != "passed"
    ):
        _fehler(f"{disziplin}: RX-12-Beleg ohne bestandenes verblindetes §4-Panel: {pfad}", fehler)
    ref_key = f"{product.casefold()}|{str(reference.get('module', '')).casefold()}"
    return len(fehler) == start_fehler, is_rx12, ref_key


def _referenz_anzahl(referenzzelle: str) -> int:
    return referenzzelle.count("/") + 1


def _interne_json_belege(disziplin: str, evidenz: str, fehler: list[str]) -> list[tuple[str, dict[str, Any]]]:
    belege: list[tuple[str, dict[str, Any]]] = []
    for pfad in re.findall(r"`((?:docs/reports/current|data)/[^`]+\.json)`", evidenz):
        rel = Path(pfad)
        if rel.is_absolute() or ".." in rel.parts:
            _fehler(f"{disziplin}: unsicherer interner Belegpfad {pfad!r}", fehler)
            continue
        try:
            bericht = json.loads((_ROOT / rel).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fehler(f"{disziplin}: interner JSON-Beleg nicht lesbar ({pfad}): {exc}", fehler)
            continue
        if isinstance(bericht, dict):
            belege.append((pfad, bericht))
        else:
            _fehler(f"{disziplin}: interner JSON-Beleg muss ein Objekt sein: {pfad}", fehler)
    return belege


def _sinnbeweis_zeilen(bericht: dict[str, Any], rezept: str) -> tuple[list[dict[str, Any]] | None, str]:
    if bericht.get("schema") != _INTERNAL_REPORT_SCHEMA:
        return None, "falsches Schadensbewertungs-Schema"
    if bericht.get("pegel_matching") is not True:
        return None, "Pegelmatching fehlt"
    zeilen = bericht.get("zeilen")
    if not isinstance(zeilen, list) or not zeilen or any(not isinstance(zeile, dict) for zeile in zeilen):
        return None, "Einzeltitel-Zeilen fehlen"
    rezept_zeilen = [zeile for zeile in zeilen if zeile.get("rezept") == rezept]
    if not rezept_zeilen:
        return None, f"Rezept {rezept!r} fehlt"
    titel = [zeile.get("song") for zeile in rezept_zeilen]
    if not all(isinstance(name, str) and name.strip() for name in titel) or len(set(titel)) != len(titel):
        return None, "Titel-IDs fehlen oder sind nicht eindeutig"
    if not all(
        _finite_zahl(zeile.get("delta_defektlast"))
        and _finite_zahl(zeile.get("delta_musik_mos"))
        and isinstance(zeile.get("besser"), bool)
        and isinstance(zeile.get("ueberrestauration"), bool)
        for zeile in rezept_zeilen
    ):
        return None, "Einzeltitel-Metriken fehlen oder sind ungültig"
    return rezept_zeilen, ""


def _sinnbeweis_aggregat_konsistent(bericht: dict[str, Any], rezept: str, zeilen: list[dict[str, Any]]) -> bool:
    alle_zeilen = bericht.get("zeilen")
    if not isinstance(alle_zeilen, list):
        return False
    counts = ("n_paare", "n_besser", "n_ueberrestauration")
    if (
        any(not isinstance(bericht.get(key), int) or isinstance(bericht.get(key), bool) for key in counts)
        or bericht.get("n_paare") != len(alle_zeilen)
        or bericht.get("n_besser")
        != sum(isinstance(zeile, dict) and zeile.get("besser") is True for zeile in alle_zeilen)
        or bericht.get("n_ueberrestauration")
        != sum(isinstance(zeile, dict) and zeile.get("ueberrestauration") is True for zeile in alle_zeilen)
    ):
        return False
    aggregat = bericht.get("aggregat")
    summary = aggregat.get(rezept) if isinstance(aggregat, dict) else None
    if (
        not isinstance(summary, dict)
        or not isinstance(summary.get("n"), int)
        or isinstance(summary.get("n"), bool)
        or not isinstance(summary.get("besser"), int)
        or isinstance(summary.get("besser"), bool)
    ):
        return False
    n = len(zeilen)
    n_besser = sum(zeile["besser"] for zeile in zeilen)
    mean_defektlast = math.fsum(float(zeile["delta_defektlast"]) for zeile in zeilen) / n
    mean_mos = math.fsum(float(zeile["delta_musik_mos"]) for zeile in zeilen) / n
    return (
        summary.get("n") == n
        and summary.get("besser") == n_besser
        and _finite_zahl(summary.get("delta_defektlast"))
        and _finite_zahl(summary.get("delta_musik_mos"))
        and math.isclose(float(summary["delta_defektlast"]), mean_defektlast, rel_tol=0.0, abs_tol=1e-5)
        and math.isclose(float(summary["delta_musik_mos"]), mean_mos, rel_tol=0.0, abs_tol=1e-4)
    )


def _kriterium_a_bestanden(bericht: dict[str, Any], rezept: str) -> tuple[bool, str]:
    zeilen, grund = _sinnbeweis_zeilen(bericht, rezept)
    if zeilen is None:
        return False, grund
    if not _sinnbeweis_aggregat_konsistent(bericht, rezept, zeilen):
        return False, "Aggregat stimmt nicht mit den Einzeltiteln überein"
    n = len(zeilen)
    if n < 10:
        return False, f"nur {n} eindeutige Titel (benötigt n≥10)"
    mean_defektlast = math.fsum(float(zeile["delta_defektlast"]) for zeile in zeilen) / n
    mean_mos = math.fsum(float(zeile["delta_musik_mos"]) for zeile in zeilen) / n
    rate_besser = sum(zeile["besser"] for zeile in zeilen) / n
    if (
        rate_besser < 0.5
        or mean_mos < 0.05
        or mean_defektlast > 0.0
        or any(zeile["ueberrestauration"] for zeile in zeilen)
    ):
        return False, "Kriterium (a): Trefferquote, ΔMOS, Δdefektlast oder Überrestauration verfehlt"
    return True, ""


def _kriterium_b_bestanden(
    regionenbericht: dict[str, Any],
    titelbericht: dict[str, Any],
    klasse: str,
    rezept: str,
) -> tuple[bool, str]:
    if regionenbericht.get("schema") != "aurik.witness_kopf_gate/2" or regionenbericht.get("klasse") != klasse:
        return False, "DRT-5-Bericht fehlt oder Klasse stimmt nicht"
    holdout_titel = regionenbericht.get("holdout_titel")
    q_values = regionenbericht.get("q_values")
    never_worsen_flags = regionenbericht.get("never_worsen_flags")
    q_median = regionenbericht.get("q_median")
    q_anteil = regionenbericht.get("q_anteil_ge_0_5")
    never_worsen = regionenbericht.get("never_worsen_anteil")
    q_median_value = _finite_float(q_median)
    q_anteil_value = _finite_float(q_anteil)
    never_worsen_value = _finite_float(never_worsen)
    n_regionen = regionenbericht.get("n_regionen")
    n_uebersprungen = regionenbericht.get("n_uebersprungen")
    n_q_bewertbar = regionenbericht.get("n_q_bewertbar")
    n_never_worsen_bewertbar = regionenbericht.get("n_never_worsen_bewertbar")
    if (
        not isinstance(q_values, list)
        or not all(_finite_zahl(value) for value in q_values)
        or not isinstance(never_worsen_flags, list)
        or not all(isinstance(value, bool) for value in never_worsen_flags)
        or not isinstance(n_q_bewertbar, int)
        or isinstance(n_q_bewertbar, bool)
        or n_q_bewertbar != len(q_values)
        or not isinstance(n_never_worsen_bewertbar, int)
        or isinstance(n_never_worsen_bewertbar, bool)
        or n_never_worsen_bewertbar != len(never_worsen_flags)
        or len(q_values) < 6
        or len(never_worsen_flags) < 6
    ):
        return False, "DRT-5-Bericht enthält keine prüfbaren Rohwerte für mindestens sechs Paare"
    q_values_sorted = sorted(float(value) for value in q_values)
    mid = len(q_values_sorted) // 2
    q_median_recomputed = (
        q_values_sorted[mid] if len(q_values_sorted) % 2 else (q_values_sorted[mid - 1] + q_values_sorted[mid]) / 2.0
    )
    q_anteil_recomputed = sum(value >= 0.5 for value in q_values_sorted) / len(q_values_sorted)
    never_worsen_recomputed = sum(never_worsen_flags) / len(never_worsen_flags)
    if (
        regionenbericht.get("gate_bestanden") is not True
        or not isinstance(holdout_titel, str)
        or not holdout_titel.strip()
        or not isinstance(n_regionen, int)
        or isinstance(n_regionen, bool)
        or not isinstance(n_uebersprungen, int)
        or isinstance(n_uebersprungen, bool)
        or n_uebersprungen < 0
        or n_regionen < n_uebersprungen
        or n_regionen - n_uebersprungen < 6
        or q_median_value is None
        or not math.isclose(q_median_value, round(q_median_recomputed, 4), rel_tol=0.0, abs_tol=1e-6)
        or q_median_value < 0.95
        or q_anteil_value is None
        or not math.isclose(q_anteil_value, round(q_anteil_recomputed, 4), rel_tol=0.0, abs_tol=1e-6)
        or q_anteil_value < 0.96
        or never_worsen_value is None
        or not math.isclose(never_worsen_value, round(never_worsen_recomputed, 4), rel_tol=0.0, abs_tol=1e-6)
        or never_worsen_value < 0.96
    ):
        return False, "DRT-5-Schwellen (Q-Median, Q-Anteil oder Never-worsen) verfehlt"
    zeilen, grund = _sinnbeweis_zeilen(titelbericht, rezept)
    if zeilen is None:
        return False, f"DRT-8: {grund}"
    if not _sinnbeweis_aggregat_konsistent(titelbericht, rezept, zeilen):
        return False, "DRT-8: Aggregat stimmt nicht mit den Einzeltiteln überein"
    if len(zeilen) < 10:
        return False, f"DRT-8: nur {len(zeilen)} eindeutige Titel (benötigt n≥10)"
    from backend.core.dsp.wohlklang_zeuge import JND_TOLERANZ
    from scripts.bewerte_schadens_proben import JND_MOS

    if any(
        float(zeile["delta_musik_mos"]) < -JND_MOS or float(zeile["delta_defektlast"]) > JND_TOLERANZ
        for zeile in zeilen
    ):
        return False, "DRT-8: mindestens ein Titel verletzt die Zwei-Metrik-Never-worsen-Grenze"
    return True, ""


def _pruefe_gesichert_evidenz(disziplin: str, evidenz: str, fehler: list[str]) -> None:
    belege = _interne_json_belege(disziplin, evidenz, fehler)
    rezept_match = re.search(r"Sinn-Beweis-Rezept:\s*`([^`]+)`", evidenz)
    if rezept_match:
        gruende: list[str] = []
        for _, bericht in belege:
            if bericht.get("schema") != _INTERNAL_REPORT_SCHEMA:
                continue
            bestanden, grund = _kriterium_a_bestanden(bericht, rezept_match.group(1))
            if bestanden:
                return
            gruende.append(grund)
        detail = "; ".join(dict.fromkeys(gruende)) or "kein verlinkter aurik.schadensbewertung/1-Beleg"
        _fehler(f"{disziplin}: interner GESICHERT-Nachweis Kriterium (a) nicht bestanden: {detail}", fehler)
        return

    klasse_match = re.search(r"DRT-5-Klasse:\s*`([^`]+)`", evidenz)
    drt8_match = re.search(r"DRT-8-Rezept:\s*`([^`]+)`", evidenz)
    if klasse_match and drt8_match:
        gruende = []
        regionenberichte = [bericht for _, bericht in belege if bericht.get("schema") == "aurik.witness_kopf_gate/2"]
        titelberichte = [bericht for _, bericht in belege if bericht.get("schema") == _INTERNAL_REPORT_SCHEMA]
        for regionenbericht in regionenberichte:
            for titelbericht in titelberichte:
                bestanden, grund = _kriterium_b_bestanden(
                    regionenbericht, titelbericht, klasse_match.group(1), drt8_match.group(1)
                )
                if bestanden:
                    return
                gruende.append(grund)
        detail = "; ".join(dict.fromkeys(gruende)) or "DRT-5-/DRT-8-JSON-Belege fehlen"
        _fehler(f"{disziplin}: interner GESICHERT-Nachweis Kriterium (b) nicht bestanden: {detail}", fehler)
        return
    _fehler(
        f"{disziplin}: GESICHERT ohne maschinenprüfbaren Kriterium-(a)- oder Kriterium-(b)-Beleg",
        fehler,
    )


def _pruefe_disziplin_tabelle(register: str, fehler: list[str]) -> None:
    marker = "## Disziplin-Tabelle"
    if marker not in register:
        _fehler("Disziplin-Tabelle fehlt im Wettbewerbsregister", fehler)
        return
    inhalt = register.split(marker, 1)[1]
    abschnitt = re.split(r"^## ", inhalt, maxsplit=1, flags=re.MULTILINE)[0]
    zeilen: dict[str, list[str]] = {}
    for zeile in abschnitt.splitlines():
        if not zeile.startswith("|"):
            continue
        zellen = [z.strip() for z in zeile.strip("|").split("|")]
        if zellen and zellen[0] == "Disziplin":
            continue
        if zellen and all(set(z) <= {"-", " "} for z in zellen):
            continue
        if len(zellen) != 5:
            _fehler(f"Disziplin-Tabelle: Zeile mit {len(zellen)} statt 5 Spalten", fehler)
            continue
        name, _, status, evidenz, naechster_schritt = zellen
        matching = [disziplin for disziplin in DISCIPLINEN if name == disziplin or name.startswith(f"{disziplin} (")]
        if len(matching) != 1:
            _fehler(f"Unbekannte Disziplin-Zeile im Register: {name!r}", fehler)
            continue
        disziplin = matching[0]
        if disziplin in zeilen:
            _fehler(f"Disziplin {disziplin!r} mehrfach im Paritätsregister", fehler)
            continue
        zeilen[disziplin] = zellen
        if status not in _STATUS:
            _fehler(f"{disziplin}: ungültiger Status-Token {status!r}", fehler)
        if status == "GESICHERT":
            zeilentext = " ".join(zellen)
            runbooks = re.findall(r"`(docs/guides/[^`]+\.md)`", zeilentext)
            if "Runbook" not in zeilentext or not runbooks:
                _fehler(f"{disziplin}: GESICHERT ohne zeileneigenes Hör-Runbook", fehler)
            for runbook in runbooks:
                rel = Path(runbook)
                if rel.is_absolute() or ".." in rel.parts or not (_ROOT / rel).is_file():
                    _fehler(f"{disziplin}: Hör-Runbook fehlt oder ungültig: {runbook}", fehler)
            _pruefe_gesichert_evidenz(disziplin, evidenz, fehler)
        if status in {"LUECKE", "OFFEN"} and not naechster_schritt:
            _fehler(f"{disziplin}: Status {status} ohne konkreten nächsten Schritt", fehler)

    for disziplin in DISCIPLINEN:
        if disziplin not in zeilen:
            _fehler(f"Disziplin {disziplin!r} fehlt in der Disziplin-Tabelle", fehler)


def _pruefe_rx12_vertrag(register: str, fehler: list[str]) -> None:
    if "## RX-12-Paritätsvertrag" not in register:
        _fehler("RX-12-Paritätsvertrag fehlt im Wettbewerbsregister", fehler)
    if not SPEC26.is_file():
        _fehler("Spec 26 fehlt (DRT-16-RX-12-Vertrag nicht prüfbar)", fehler)
    else:
        spec26 = SPEC26.read_text(encoding="utf-8")
        if "## DRT-16 — Externer RX-12-Paritätsvergleich" not in spec26:
            _fehler("Spec 26 ohne DRT-16 (externer RX-12-Vergleich)", fehler)

    for zeile in register.splitlines():
        if not zeile.startswith("| "):
            continue
        zellen = [z.strip() for z in zeile.strip("|").split("|")]
        if len(zellen) < 5 or zellen[2] != "PARITAET":
            continue
        belege = re.findall(r"`(docs/reports/current/[^`]+\.json)`", zellen[3])
        if not belege:
            _fehler(f"{zellen[0]}: PARITAET ohne verlinkten maschinenlesbaren RX-Beleg", fehler)
        rx12_bestanden = False
        referenzen_belegt: set[str] = set()
        for beleg in belege:
            gueltig, ist_rx12, referenz = _pruefe_paritaets_beleg(beleg, zellen[0], fehler)
            if gueltig:
                referenzen_belegt.add(referenz)
            rx12_bestanden = rx12_bestanden or (gueltig and ist_rx12)
        if len(referenzen_belegt) < _referenz_anzahl(zellen[1]):
            _fehler(f"{zellen[0]}: nicht jede genannte Referenz hat einen eigenen gültigen Beleg", fehler)
        referenztext = zellen[1].casefold()
        if "cedar" in referenztext and not any("cedar" in ref for ref in referenzen_belegt):
            _fehler(f"{zellen[0]}: PARITAET ohne Beleg für die genannte Cedar-Referenz", fehler)
        if "acon" in referenztext and not any("acon" in ref for ref in referenzen_belegt):
            _fehler(f"{zellen[0]}: PARITAET ohne Beleg für die genannte Acon-Referenz", fehler)
        if re.search(r"\bRX\b", referenztext) and not rx12_bestanden:
            _fehler(f"{zellen[0]}: PARITAET ohne bestandenen RX-12-Referenzbeleg", fehler)


def main() -> int:
    fehler: list[str] = []
    if not LEDGER.is_file():
        _fehler("Wettbewerbs-Parität: .github/WETTBEWERBS_PARITAET.md fehlt", fehler)
        print("🔌 WETTBEWERBS-PARITAET — VERSTOSS:")
        for f in fehler:
            print(f"  ❌ {f}")
        return 1
    q = LEDGER.read_text(encoding="utf-8")
    _pruefe_disziplin_tabelle(q, fehler)
    _pruefe_rx12_vertrag(q, fehler)
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
