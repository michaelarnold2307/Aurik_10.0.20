"""defect_audibility_gate - Hörbarkeits-Gate für Restdefekte (Hörordnung Ebene 2).

hoerordnung.instructions.md §4: "Reparatur gilt als abgeschlossen, wenn ein
Defekt unter der Maskierungsschwelle liegt - nicht wenn sein Messwert Null
ist."  Dieser Guard übersetzt das in ein Entscheidungs-Gate am Lauf-Ende:

  * Eingabe: Per-Defekt-Reduktion aus dem §v10.702/§v10.703 Post-Scan
    (self._defect_reduction_per_type): pre/post-Severity, reduction,
    masked_events (ERB-maskierte Events laut DefectScanner).
  * Schwelle: material-/ketten-adaptive JND-Schwelle (Severity-Proxy,
    §v10.704 S3) - kanonische Quelle; der Inline-Block in
    unified_restorer_v3.py importiert MATERIAL_JND_OFFSET hierher.
  * Status je Defekttyp: resolved | never_audible | masked | audible |
    physical_cap.  "Maskiert" (ERB) und "physikalische Obergrenze"
    (z. B. bandwidth_loss am Ketten-Ende einer mp3-Quelle) gelten als
    hörbarkeits-erfüllt; nur *unmaskierte* Restdefekte über Schwelle
    lassen das Gate kippen (gate_passed=False) und werden als
    "nachbehandlungswürdig" (improvable_types) ausgewiesen.

Konsistenz-Slice 2 (2026-09-27, „eine Hör-Instanz, eine Wahrheit",
PHASE_SOTA_GAP_ANALYSE.md §Sub-audible SOTA-Konsistenz): Drei Instanzen
beantworteten „Defekt hörbar?" auf drei Skalen. Seit diesem Slice gilt ein
kanonischer Entscheidungsfluss:

  1. KANONISCHE MASKIERUNG (backend/core/dsp/audibility_gate.py -
     dieselbe Instanz wie ~25 Phasen-Aufrufer): Liegt Audio + Defekt-Locations
     vor (``audio``/``sample_rate``/``defect_locations``), entscheidet das
     Maskierungsmodell (ISO 11172-3 Bark bzw. Zwicker ISO 532-1) + Pegel-JND-
     Floor (hearing_jnd level_broadband, Mills 1960) über „maskiert"/"audible".
  2. SEVERITY-SKALA als dokumentierte operative Näherung NUR im Fallback
     (kein Audio, keine Locations, FM-Zeitachsen-Defekte ohne Energie-Domäne
     oder Fehler im kanonischen Pfad -> fail-open §V6 (copilot-instructions.md)).
  3. PERCEPTUAL-SALIENCE (n_masked_events, PerceptualSalienceEstimator) bleibt
     Evidenz im Report, trägt aber keine Entscheidung mehr, wenn die kanonische
     Instanz verfügbar ist.

Der deklarative Pfad bleibt numpy-frei; der kanonische Pfad importiert numpy
lazy (kein zweiter Voll-Scan - nur Defekt-Locations im finalen Audio).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --- Kanonische Hörbarkeits-Schwellen (§v10.704 S3) --------------------------
# Basis 0.08 (Severity-Skala 0..1); Material-Offset (Vinyl: breiterer
# Frequenzgang → niedrigere Schwelle = mehr hörbar; Cassette: höherer
# Noise-Floor → mehr Maskierung); tiefere Transfer-Kette → +0.01/Ebene.
AUDIBLE_BASE = 0.08
MATERIAL_JND_OFFSET: dict[str, float] = {
    "cassette": 0.04,
    "cassette_tape": 0.04,
    "vinyl": -0.02,
    "lp": -0.02,
    "shellac": 0.02,
    "reel_tape": 0.00,
    "cd_digital": -0.03,
    "mp3_low": 0.05,
    "mp3_high": 0.03,
    "aac": 0.03,
    "streaming": 0.02,
}
AUDIBLE_CLIP_MIN = 0.03
AUDIBLE_CLIP_MAX = 0.15
DEPTH_OFFSET_PER_LEVEL = 0.01

# Defekttypen, deren Rest an der physikalischen Obergrenze der Quelle liegt
# (z. B. durch Codec-Kette verlorenes Band): kein hörbarkeits-relevanter
# Nachbehandlungsspielraum - als "erfüllt mit Dokumentation" gewertet.
PHYSICAL_CAP_DEFECT_TYPES: frozenset[str] = frozenset({"bandwidth_loss"})

# Maskierte Events gelten nur bis zu dieser post-Severity als „maskiert“;
# darüber ist der Restdefekt sicher exponiert (konservativ).
_MASKED_EVENTS_MAX_POST = 0.35

# Defekttyp → zuständige Nachbehandlungs-Phase (m1b, gezielte Stufe-2-Queue).
# Nur Typen mit klarer, sicherer Phasen-Zuordnung; ohne Eintrag kein Deferral
# (kein blindes „mehr von allem“). Phasen 21/35/42 sind Restoration-verboten.
DEFECT_RETRY_PHASE_MAP: dict[str, str] = {
    "hum": "phase_02_hum_removal",
    "hum_buzz": "phase_02_hum_removal",
    "clicks": "phase_01_click_removal",
    "click": "phase_01_click_removal",
    "click_pop": "phase_27_click_pop_removal",
    "crackle": "phase_09_crackle_removal",
    "wow": "phase_12_wow_flutter_fix",
    "flutter": "phase_12_wow_flutter_fix",
    "jitter_artifacts": "phase_14_phase_correction",  # V27: JITTER nie über phase_12
    "motor_interference": "phase_12_wow_flutter_fix",
    "speed_variation": "phase_12_wow_flutter_fix",
    "hiss": "phase_29_tape_hiss_reduction",
    "high_frequency_hiss": "phase_29_tape_hiss_reduction",
    "tape_hiss": "phase_29_tape_hiss_reduction",
    "reverb_excess": "phase_49_advanced_dereverb",
    "echo": "phase_61_groove_echo_cancellation",
    "compression_artifacts": "phase_10_compression",
}


def retry_phases_for_types(improvable_types: list[str]) -> list[str]:
    """Mappt hörbar gebliebene Defekttypen auf ihre Nachbehandlungs-Phase(n)."""
    seen: set[str] = set()
    out: list[str] = []
    for dt in improvable_types or []:
        ph = DEFECT_RETRY_PHASE_MAP.get(str(dt).lower())
        if ph and ph not in seen:
            seen.add(ph)
            out.append(ph)
    return out


def audible_threshold(material_key: str, chain_depth: int = 1) -> float:
    """Material-/ketten-adaptive JND-Hörbarkeitsschwelle (Severity-Skala)."""
    mat = str(material_key or "").lower()
    depth = max(1, int(chain_depth or 1))
    thr = AUDIBLE_BASE + MATERIAL_JND_OFFSET.get(mat, 0.0) + (depth - 1) * DEPTH_OFFSET_PER_LEVEL
    return float(max(AUDIBLE_CLIP_MIN, min(AUDIBLE_CLIP_MAX, thr)))


@dataclass
class DefectAudibilityReport:
    """Ergebnis des Hörbarkeits-Gates am Lauf-Ende."""

    threshold: float
    material_key: str
    chain_depth: int
    per_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_total: int = 0
    n_audible_pre: int = 0
    n_audible_post_raw: int = 0  # post >= Schwelle (unabhängig von Maskierung)
    n_masked: int = 0  # post >= Schwelle, aber ERB-maskierte Events vorhanden
    n_audible_unmasked: int = 0  # hörbar geblieben → Gate-relevant
    n_resolved: int = 0  # pre >= Schwelle → post < Schwelle
    n_never_audible: int = 0
    n_physical_cap: int = 0
    improvable_types: list[str] = field(default_factory=list)
    gate_passed: bool = True
    # §G8 (copilot-instructions.md) Evidenz-Zustand (2026-10-07): "passed" ohne
    # Post-Scan ist KEIN Nachweis. Fail-open gilt nur fuer den BLOCK (keine
    # Blockade ohne Daten), nicht fuer die Qualitaets-Zusage.
    n_verified: int = 0
    n_unevaluable: int = 0  # Zeilen ohne Zahl (nicht gemessen, keine Evidenz)
    evidence_state: str = "evaluated"  # evaluated | no_residual_defects | scan_missing
    gate_verified: bool = True  # gate_passed UND Evidenz vorhanden

    def to_metadata(self) -> dict[str, Any]:
        return {
            "gate_passed": bool(self.gate_passed),
            "gate_verified": bool(self.gate_verified),
            "evidence_state": str(self.evidence_state),
            "n_verified": int(self.n_verified),
            "n_unevaluable": int(self.n_unevaluable),
            "threshold": round(float(self.threshold), 4),
            "material": str(self.material_key),
            "chain_depth": int(self.chain_depth),
            "n_total": int(self.n_total),
            "n_audible_pre": int(self.n_audible_pre),
            "n_audible_post_raw": int(self.n_audible_post_raw),
            "n_masked": int(self.n_masked),
            "n_audible_unmasked": int(self.n_audible_unmasked),
            "n_resolved": int(self.n_resolved),
            "n_physical_cap": int(self.n_physical_cap),
            "improvable_types": list(self.improvable_types),
            "per_type": {k: {kk: vv for kk, vv in v.items() if kk != "phase_hint"} for k, v in self.per_type.items()},
        }


# --- Kanonische Maskierungs-Instanz (Konsistenz-Slice 2) ---------------------
# Band-Konventionen je Defekt-Domäne, abgeleitet aus den Produktions-Bändern der
# Phasen-Aufrufer von defect_audibility (z. B. phase_01: 1200-16000 Hz,
# phase_19: 4000-12000 Hz). Typen OHNE Eintrag nutzen das Default-Band;
# FM-/Zeitachsen-Defekte (wow/flutter/...) haben keine Energie-Delta-Domäne
# und bleiben auf der Severity-Skala (ehrlich dokumentiert, kein Pseudo-Band).
_CANONICAL_BANDS: dict[str, tuple[float, float]] = {
    "hum": (45.0, 1000.0),
    "hum_buzz": (45.0, 1000.0),
    "motor_interference": (45.0, 1000.0),
    "clicks": (1200.0, 16000.0),
    "click": (1200.0, 16000.0),
    "click_pop": (1200.0, 16000.0),
    "crackle": (1200.0, 16000.0),
    "hiss": (3000.0, 16000.0),
    "high_frequency_hiss": (3000.0, 16000.0),
    "tape_hiss": (3000.0, 16000.0),
    "sibilance": (4000.0, 12000.0),
    "echo": (100.0, 8000.0),
    "groove_echo": (100.0, 8000.0),
    "reverb_excess": (100.0, 8000.0),
    "compression_artifacts": (1000.0, 12000.0),
}
_CANONICAL_DEFAULT_BAND: tuple[float, float] = (200.0, 16000.0)
# Zeitachsen-/FM-Defekte: keine sinnvolle Energie-Delta-Domäne im kanonischen
# Gate -> Severity-Skala (Fallback-Pfad) bleibt die Entscheidungs-Instanz.
_CANONICAL_NO_BAND_TYPES: frozenset[str] = frozenset(
    {
        "wow",
        "flutter",
        "speed_variation",
        "speed_calibration_error",
        "jitter_artifacts",
        "pitch_drift",
        "scrape_flutter",
        "multiband_wow_flutter",
        "flutter_spectral_sidebands",
        "transport_bump",
    }
)
# Obergrenze der je Typ kanonisch geprüften Locations (Determinismus §G5 (GEBOTE.md)
# (copilot-instructions.md), Laufzeit-Budget: Maskierungs-Modell je Location).
_CANONICAL_MAX_LOCATIONS_PER_TYPE = 32


def _canonical_band_for_type(dt_name: str) -> tuple[float, float] | None:
    key = str(dt_name or "").strip().lower()
    if key in _CANONICAL_NO_BAND_TYPES:
        return None
    return _CANONICAL_BANDS.get(key, _CANONICAL_DEFAULT_BAND)


def canonical_audibility_verdicts(
    audio: Any,
    sample_rate: Any,
    defect_locations: dict[str, list[tuple[float, float]]] | None,
    types: list[str] | None = None,
    *,
    model: str = "mpeg1",
) -> dict[str, dict[str, Any]]:
    """Kanonische Hörbarkeits-Verdikte je Defekttyp (eine Hör-Instanz, Slice 2).

    Verwendet DIESELBE Instanz wie die reparierenden Phasen
    (``backend.core.dsp.audibility_gate.defect_audibility_from_signal`` -
    Maskierungsmodell + Pegel-JND-Floor). Pro Typ wird die Sanitisierung des
    Signals EINMAL durchgeführt (§PERF-R6) und höchstens
    ``_CANONICAL_MAX_LOCATIONS_PER_TYPE`` Locations geprüft.

    Returns:
        {type: {"audible": bool, "checked": int, "locations": int}} -
        ``audible=True`` sobald EINE Location über der Maskierungsschwelle
        liegt. Typen ohne Locations, ohne kanonisches Band oder bei Fehlern
        fehlen im Ergebnis (Aufrufer fällt auf die Severity-Skala zurück,
        fail-open §V6 (copilot-instructions.md)). Deterministisch (§G5).
    """
    if defect_locations is None:
        return {}
    try:
        sr = int(sample_rate or 0)
    except (TypeError, ValueError):
        logger.warning(
            "defect_audibility_gate: sample_rate=%r unbrauchbar -> keine Maskierungsschwellen (§V6, copilot-instructions.md)",
            sample_rate,
        )
        return {}
    if sr <= 0:
        return {}
    try:
        import numpy as np

        from backend.core.dsp.audibility_gate import (
            SanitizedSignal,
            defect_audibility_from_signal,
        )
    except Exception as _imp_exc:  # §V6 (copilot-instructions.md): nie blockieren
        logger.warning("Kanonische Hör-Instanz nicht ladbar (%s) - Severity-Ersatzpfad", _imp_exc)
        return {}
    try:
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim > 1:  # Stereo-Layout-Invariante (AGENTS.md §3)
            if a.shape[0] == 2 and a.shape[1] > 2:
                mono = a.mean(axis=0)
            else:
                mono = a.mean(axis=1)
        else:
            mono = a
    except Exception as _np_exc:  # pragma: no cover - Defensivpfad
        logger.warning("Kanonische Hör-Instanz: Layout-Fehler (%s) - Severity-Ersatzpfad", _np_exc)
        return {}
    out: dict[str, dict[str, Any]] = {}
    wanted = {str(t).strip().lower() for t in (types or [])} | {str(k).strip().lower() for k in defect_locations}
    for dt_name, locs in defect_locations.items():
        key = str(dt_name).strip().lower()
        if wanted and key not in wanted:
            continue
        band = _canonical_band_for_type(key)
        if band is None or not locs:
            continue
        lo_hz, hi_hz = band
        checked = 0
        audible = False
        try:
            sig = SanitizedSignal(mono)
            for loc in list(locs)[:_CANONICAL_MAX_LOCATIONS_PER_TYPE]:
                start_s, end_s = float(loc[0]), float(loc[1])
                d0 = int(round(start_s * sr))
                d1 = int(round(end_s * sr))
                if d1 <= d0:
                    continue
                verdict = defect_audibility_from_signal(sig, sr, d0, d1, lo_hz=lo_hz, hi_hz=hi_hz, model=model)
                checked += 1
                if bool(verdict.get("audible")):
                    audible = True
                    break
            if checked:
                out[key] = {"audible": audible, "checked": checked, "locations": len(locs)}
        except Exception as _ver_exc:  # §V6 (copilot-instructions.md): fail-open
            logger.warning("Kanonische Hör-Instanz für %s fehlgeschlagen (%s) - Severity-Ersatzpfad", key, _ver_exc)
            continue
    return out


def _sev(value: Any) -> float:
    try:
        v = float(value or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    if v != v:  # NaN-Schutz
        return 0.0
    return max(0.0, min(1.0, v))


def _sev_or_none(value: Any) -> float | None:
    """Severity oder ``None``, wenn der Wert KEINE Zahl ist (§G8 copilot-instructions.md).

    ``_sev`` mappt fehlende/kaputte Werte bewusst auf 0,0 (Severity-Skala-
    Fallback). Für den Evidenz-Zustand ist diese Gleichsetzung falsch: eine
    Zeile ohne Zahl ist NICHT gemessen und darf weder als „never_audible"
    durchgehen noch als Beleg zählen.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def evaluate_defect_audibility(
    defect_reduction_per_type: dict[str, dict[str, Any]] | None,
    *,
    material_key: str = "vinyl",
    chain_depth: int = 1,
    physical_cap_types: set[str] | None = None,
    audio: Any = None,
    sample_rate: Any = None,
    defect_locations: dict[str, list[tuple[float, float]]] | None = None,
    canonical_model: str = "mpeg1",
    post_scan_ran: bool | None = None,
) -> DefectAudibilityReport:
    """Bewertet die Restdefekte gegen die Hörbarkeitsschwelle (reine Funktion).

    Konsistenz-Slice 2 (2026-09-27): Liegt ``audio`` + ``sample_rate`` +
    ``defect_locations`` (Final-Audio und Post-Scan-Locations in Sekunden) vor,
    entscheidet die KANONISCHE Maskierungs-Instanz
    (``dsp/audibility_gate.defect_audibility_from_signal`` - dieselbe wie in
    ~25 Phasen-Aufrufern) über „maskiert"/"audible". Die Severity-Skala bleibt
    der dokumentierte Fallback (kein Audio/Locations, FM-Zeitachsen-Defekte,
    Fehler - fail-open §V6 (copilot-instructions.md)) und für alle Typen der
    Pre-Filter (post >= Schwelle). ``n_masked_events`` (Perceptual-Salience)
    bleibt Evidenz im Report (``evidence``-Feld je Typ dokumentiert die
    entscheidende Instanz). Deterministisch (§G5 (copilot-instructions.md)).

    Args:
        defect_reduction_per_type: §B2-Post-Scan-Daten {type: {pre, post,
            reduction, masked_events, ...}}. Fehlt der Eintrag (kein Post-Scan),
            gilt das Gate als nicht bewertbar → passed=True (kein Block).
        material_key: Material (z. B. "vinyl", "mp3_low").
        chain_depth: Tiefe der Transfer-Kette (1 = keine Zwischenstufen).
        physical_cap_types: zusätzliche Typen ohne Nachbesserungsspielraum.
        audio: finales (restauriertes) Audio für die kanonische Instanz (optional).
        sample_rate: Abtastrate zu ``audio`` (optional).
        defect_locations: {type: [(start_s, end_s), ...]} aus dem Post-Scan (optional).
        canonical_model: Maskierungsmodell der kanonischen Instanz ("mpeg1" Default).
        post_scan_ran: Hat der §B2-Post-Scan fuer DIESEN Song gelaufen? ``False``
            ⇒ ``evidence_state="scan_missing"`` und ``gate_verified=False``
            (§G8 (copilot-instructions.md): ohne Messung keine Zusage). ``None``
            leitet ab: Eintraege vorhanden ⇒ "evaluated", sonst "scan_missing".
    """
    thr = audible_threshold(material_key, chain_depth)
    caps = set(PHYSICAL_CAP_DEFECT_TYPES) | set(physical_cap_types or set())
    report = DefectAudibilityReport(
        threshold=thr,
        material_key=str(material_key),
        chain_depth=max(1, int(chain_depth or 1)),
    )
    data = defect_reduction_per_type or {}
    report.n_total = len(data)
    # Kanonische Maskierungs-Verdikte NUR für Severity-Kandidaten berechnen
    # (post >= Schwelle) - Laufzeit-Budget, Pre-Filter bleibt die Severity-Skala.
    canonical: dict[str, dict[str, Any]] = {}
    if audio is not None and sample_rate is not None and defect_locations:
        try:
            cand_types = [str(k) for k, v in data.items() if isinstance(v, dict) and _sev(v.get("post")) >= thr]
            if cand_types:
                canonical = canonical_audibility_verdicts(
                    audio, sample_rate, defect_locations, types=cand_types, model=canonical_model
                )
        except Exception as _canon_exc:  # §V6 (copilot-instructions.md): fail-open
            logger.warning("Kanonische Hör-Instanz nicht verfügbar (%s) - Severity-Ersatzpfad", _canon_exc)
            canonical = {}
    for dt_name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        pre = _sev(entry.get("pre"))
        post = _sev(entry.get("post"))
        if _sev_or_none(entry.get("pre")) is None or _sev_or_none(entry.get("post")) is None:
            # §G8 (copilot-instructions.md): Zeile ohne Zahl ist NICHT gemessen.
            report.per_type[dt_name] = {
                "pre": pre,
                "post": post,
                "reduction": 0.0,
                "masked_events": 0,
                "audible_pre": False,
                "audible_post": False,
                "status": "unevaluable",
                "evidence": "malformed_entry",
            }
            report.n_unevaluable += 1
            continue
        masked = 0
        try:
            masked = int(entry.get("masked_events", 0) or 0)
        except (TypeError, ValueError):
            masked = 0
        aud_pre = pre >= thr
        aud_post = post >= thr
        canon = canonical.get(str(dt_name).strip().lower())
        status: str
        evidence: str
        if canon is not None:
            # Eine Hör-Instanz: kanonische Maskierung entscheidet.
            evidence = "canonical_masking"
            if not bool(canon.get("audible")):
                status = "masked"
            elif dt_name in caps:
                status = "physical_cap"
            else:
                status = "audible"
        else:
            evidence = "severity_scale"
            if not aud_pre and not aud_post:
                status = "never_audible"
            elif aud_pre and not aud_post:
                status = "resolved"
            elif aud_post and masked > 0 and post <= _MASKED_EVENTS_MAX_POST:
                status = "masked"
            elif aud_post and dt_name in caps:
                status = "physical_cap"
            elif aud_post:
                status = "audible"
            else:  # pre < thr <= post ist durch obige Zweige abgedeckt
                status = "audible"
        _pt_entry: dict[str, Any] = {
            "pre": round(pre, 4),
            "post": round(post, 4),
            "reduction": round(max(0.0, pre - post), 4),
            "masked_events": masked,
            "audible_pre": bool(aud_pre),
            "audible_post": bool(aud_post),
            "status": status,
            "evidence": evidence,
        }
        if canon is not None:
            _pt_entry["canon_audible"] = bool(canon.get("audible"))
            _pt_entry["canon_checked"] = int(canon.get("checked", 0) or 0)
            _pt_entry["canon_locations"] = int(canon.get("locations", 0) or 0)
        report.per_type[dt_name] = _pt_entry
        if aud_pre:
            report.n_audible_pre += 1
        if aud_post:
            report.n_audible_post_raw += 1
        if status == "resolved":
            report.n_resolved += 1
        elif status == "masked":
            report.n_masked += 1
        elif status == "physical_cap":
            report.n_physical_cap += 1
        elif status == "audible":
            report.n_audible_unmasked += 1
            if pre - post > 0.005:  # Reduktion fand statt → Spielraum für mehr
                report.improvable_types.append(dt_name)
        elif status == "never_audible":
            report.n_never_audible += 1
    report.gate_passed = report.n_audible_unmasked == 0
    # §G8 (copilot-instructions.md) Evidenz-Zustand: "bestanden" (kein hörbarer
    # Restdefekt) und "belegt" (es wurde überhaupt gemessen) sind ZWEI Aussagen.
    report.n_verified = sum(1 for v in report.per_type.values() if str(v.get("status")) != "unevaluable")
    if post_scan_ran is False:
        report.evidence_state = "scan_missing"
    elif report.n_verified > 0:
        report.evidence_state = "evaluated"
    elif post_scan_ran is True:
        report.evidence_state = "no_residual_defects"
    else:
        report.evidence_state = "scan_missing"
    report.gate_verified = bool(report.gate_passed and report.evidence_state != "scan_missing")
    if report.evidence_state == "scan_missing":
        logger.warning(
            "§Hörbarkeits-Gate UNGEPRÜFT (§G8 (copilot-instructions.md)): kein §B2-Post-Scan-Ergebnis "
            "(%d Typ-Einträge) — gate_passed=%s ist damit KEIN Nachweis "
            "„Residuum unter der Maskierungsschwelle“ (fail-open nur für den Block, "
            "fail-closed für die Qualitäts-Zusage, §V6 (copilot-instructions.md))",
            report.n_verified,
            report.gate_passed,
        )
    return report


def log_audibility_report(report: DefectAudibilityReport) -> None:
    """Einheitliche Log-Ausgabe (INFO bei bestanden, WARNING bei Resthörbarem)."""
    if report.gate_passed:
        logger.info(
            "§Hörbarkeits-Gate BESTANDEN (thr=%.3f, material=%s): total=%d "
            "audible_pre=%d resolved=%d masked=%d physical_cap=%d",
            report.threshold,
            report.material_key,
            report.n_total,
            report.n_audible_pre,
            report.n_resolved,
            report.n_masked,
            report.n_physical_cap,
        )
    else:
        logger.warning(
            "§Hörbarkeits-Gate NICHT bestanden (thr=%.3f, material=%s): "
            "%d unmaskierte Restdefekt-Typen über Schwelle - nachbehandlungswürdig: %s",
            report.threshold,
            report.material_key,
            report.n_audible_unmasked,
            ", ".join(report.improvable_types) or "(keine Reduktion erzielt)",
        )
