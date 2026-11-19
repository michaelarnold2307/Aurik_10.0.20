"""backend/core/dsp/wohlklang_zeuge.py — referenzfreier Wohlklang-Zeuge je Lauf.

Hintergrund (``docs/SOTA_HANDOVER.md`` §2b Hebel F): Aurik besitzt genau zwei
**richtungszertifizierte, referenzfreie** Qualitätsmetriken —

* ``musik_mos`` (reines numpy, kein Modell-Artefakt, kein GPU-Bedarf) und
* ``defektlast`` (Defekt-Index aus Clipping, Spektral-Flatness, HF-Verlust),

beide mit Spearman 0,95 gegen die erwartete Degradations-Ordnung zertifiziert
(``docs/reports/current/qualitaetsmetrik_richtungszertifikat.json``, Seed 42).
Bis 2026-11-08 liefen sie ausschließlich in Mess-Skripten (B3-Stufenmessung,
EAR-VAE-Vergleich, Golden-Ear-Prototyp) — der Produktionspfad hat sie nie
gesehen. Damit fehlte der „Dauer-Wohlklang-Zeuge": eine Aussage **je Lauf**,
ob die Restaurierung das Signal unter der Hörschwelle verbessert oder
verschlechtert hat.

Rolle (Hörordnung §7/§8a — „Metriken sind Zeugen, die Hör-Instanz
entscheidet"): **REPORT-ONLY**. Der Zeuge verändert kein Sample, löst keinen
Rollback aus und blockiert keinen Export. Er schreibt
``metadata["wohlklang_zeuge"]`` und markiert ``gerichteter_fall`` = True,
sobald eine Metrik über die JND hinaus in die falsche Richtung läuft — solche
Läufe sind der Hör-Instanz vorzulegen (``§V7`` (copilot-instructions.md):
kein stiller Flip, keine automatische Gegenmaßnahme).

**Eine** Ausnahme vom reinen Zeugen-Status ist der **DRT-10-Produktions-Verdikt**
(Spec 26 DRT-10, M1-Verschärfung 2026-11-17): Ist der Ganz-Titel-Befund
**Überrestauration** (defektlast-Gewinn ≥ JND bei musik_mos-Verlust ≥ JND),
dann markiert ``drt10_verdict_in_metadata`` **jeden** Lauf als ``degraded`` mit
``fail_reason=RESTORATION_DRT10_UEBERRESTAURATION`` (§0c: bestmögliches
sicheres Ergebnis wird exportiert — kein Hardstop, kein Sample-Eingriff durch
den Zeugen selbst). Der Aufrufer (UV3) setzt zusätzlich den
**Identitäts-Rollback** um (Eingang statt Ausgabe), sofern keine hohe
Hör-Instanz vorliegt (HPI ≥ 0,75 ⇒ §7-Konfliktregel: Messartefakt-Verdacht,
kein Rollback). Ohne Überrestauration bleibt alles beim REPORT-ONLY-Zeugen.

Determinismus (``§G5`` (copilot-instructions.md)): reine numpy-Operationen,
keine Zufallsgrößen, keine Zeit-/Umgebungsabhängigkeit — derselbe Eingang
liefert denselben Befund.

Die Metrik-Definitionen sind **bit-identisch** zur Mess-Referenz
(``scripts/quality_metric_direction_gate.py``) — die Zertifikatszahlen bleiben
damit gültig; ein Test pinnt die Gleichheit.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = "aurik.wohlklang_zeuge/1"
ZERTIFIKAT = "docs/reports/current/qualitaetsmetrik_richtungszertifikat.json"
# DRT-10-Produktions-Verdikt (Spec 26 DRT-10): Fehlercode des degradierten
# Exports, wenn ein aktiver Reparatur-Kandidat überrestauriert (§0c).
DRT10_FAIL_REASON = "RESTORATION_DRT10_UEBERRESTAURATION"
# Never-worsen-Toleranz der B3-/Audibility-Stufe (MOS-Punkte). EINE Quelle für
# Stufenmessung, Kompositions-Suche und Produktions-Zeugen — ein Test pinnt die
# Gleichheit, damit die drei Pfade nicht still auseinanderlaufen.
JND_TOLERANZ = 0.05
_MIN_SAMPLES = 2048  # Untergrenze der Metrik-Definition (2048er-FFT)


def _mono(audio: Any) -> np.ndarray:
    """Stereo-Layout-Invariante (AGENTS.md §3): (C, N) und (N, C) ⇒ mono.

    Bewusst **ohne** dtype-Zwang: der Defekt-Index muss für denselben Eingang
    bit-identisch zur Mess-Referenz bleiben (dtype-Wechsel verschiebt die
    4. Nachkommastelle).
    """
    x: np.ndarray = np.asarray(audio)
    if x.ndim == 2:
        if x.shape[0] <= 8 and x.shape[-1] > 8:
            x = x.T
        x = np.asarray(x.mean(axis=1), dtype=x.dtype)
    ergebnis: np.ndarray = np.ascontiguousarray(x.reshape(-1))
    return ergebnis


def defektlast(audio: Any, sample_rate: int) -> float:
    """Richtungstrivialer Defekt-Index — höher = mehr Defekte.

    Kanonische Definition (vormals ``scripts/quality_metric_direction_gate.py``):
    jede Komponente misst den Defekt selbst, die Richtung ist damit per
    Konstruktion bekannt. Fenster: erste 16384 Samples (Zertifikats-Referenz).
    """
    seg = _mono(audio)[: 2048 * 8]
    clip = float(np.mean(np.abs(seg) > 0.98))
    spec = np.abs(np.fft.rfft(seg, n=2048)) ** 2 + 1e-18
    flatness = float(np.clip(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-12), 0.0, 1.0))
    freqs = np.fft.rfftfreq(2048, d=1.0 / max(int(sample_rate), 1))
    hf = float(np.sum(spec[freqs > 8000]) / (np.sum(spec) + 1e-12))
    hf_verlust = max(0.0, 1.0 - min(1.0, hf / 0.02))
    return round(clip * 2.0 + flatness + hf_verlust * 0.5, 4)


def saettigung_kanal(audio: Any) -> float:
    """Gain-invarianter Sättigungs-Kanal: 1 / Crest-Faktor (kanonisch).

    D-K3-95/97-Lücke: die zertifizierte defektlast-Referenz sieht
    tanh-/Soft-Clipping-Sättigung nicht (gemessen 2026-11-14: defektlast
    identisch vor/nach k398-Clipping-Rezeptur; flat_top_anteil ≈ 0, weil
    tanh keine harten Plateaus erzeugt). Der Crest-Faktor (peak/rms) fällt
    bei Sättigung monoton — 1/crest ist gain-invariant (Skalierung kürzt
    sich) und normiert: Sinus ≈ 0,707, schwere Sättigung → 1,0.
    **Nicht zertifiziert** — Report-Only-Zeuge (§G191 (GEBOTE.md)),
    Richtung = höher ist mehr Defekt.
    """
    x = _mono(audio)
    if x.size < 4:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))))) + 1e-12
    if rms < 1e-9:
        return 0.0
    crest = float(np.max(np.abs(x))) / rms
    return round(1.0 / max(crest, 1.0), 4)


def jitter_kanal(audio: Any, sample_rate: int) -> float:
    """Gain-invarianter Jitter-Kanal: Seitenband-Energie-Verhältnis um die
    stärksten Spektrallinien (kanonisch, 2026-11-14).

    Q0-Witness-Forschung: die Q≈0-Klasse ``jitter_artifacts`` ist für
    defektlast blind; gemessen auf 116 verwertbaren k398-Regionen
    (docs/reports/current/2026-11-14_q0_witness_forschung.json):
    Median(schaden − clean) = +0,180, 98,3 % über JND 0,02 — der Kanal
    sieht den Schaden mit richtiger Richtung. Jitter (Zeitbasis-Unruhe)
    verbreitert Spektrallinien ⇒ mehr Energie in der ±0,5–20-Hz-Nachbarschaft
    der stärksten Linien relativ zur Linien-Energie. Das Verhältnis ist
    gain-invariant (Skalierung kürzt sich), Stille ⇒ 0,0.
    **Nicht zertifiziert** — Report-Only-Zeuge (§G191 (GEBOTE.md)),
    Richtung = höher ist mehr Defekt.
    """
    x = _mono(audio)
    sr = int(sample_rate)
    if x.size < _MIN_SAMPLES or sr <= 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))))) + 1e-12
    if rms < 1e-9:
        return 0.0
    win = np.hanning(x.size)
    P = np.abs(np.fft.rfft(x * win)) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    n = len(P)
    m = np.argpartition(P, -8)[-8:]
    linie = 0.0
    seite = 0.0
    df = f[1] - f[0] if n > 1 else 0.0
    r = max(int(20.0 / df), 2) if df > 0 else 2
    for i in m:
        lo = max(0, int(i) - r)
        hi = min(n, int(i) + r)
        linie += float(P[i])
        seite += float(np.sum(P[lo:hi]) - P[i])
    return round(seite / (linie + 1e-20), 4)


def _musik_mos(audio: Any, sample_rate: int) -> float:
    """Richtungszertifiziertes Musik-MOS (höher = besser), numpy-only."""
    from plugins.musik_mos_plugin import get_musik_mos  # pylint: disable=import-outside-toplevel

    return float(get_musik_mos().estimate_mos(_mono(audio), int(sample_rate)).mos)


def _rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)


def messe_wohlklang_zeuge(
    eingang: Any,
    ausgang: Any,
    sample_rate: int,
    *,
    jnd: float = JND_TOLERANZ,
) -> dict[str, Any]:
    """Misst Eingang und Ausgang je Lauf und meldet gerichtete Fälle.

    Raises:
        ValueError: Signal zu kurz für die Metrik-Definition (< 2048 Samples).
    """
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError(f"ungültige Sample-Rate: {sample_rate!r}")
    x_in = _mono(eingang)
    x_out = _mono(ausgang)
    if x_in.size < _MIN_SAMPLES or x_out.size < _MIN_SAMPLES:
        raise ValueError(
            f"Signal zu kurz für den Wohlklang-Zeugen (mind. {_MIN_SAMPLES} Samples, "
            f"eingang={x_in.size}, ausgang={x_out.size})"
        )

    mos_in = _musik_mos(x_in, sr)
    mos_out = _musik_mos(x_out, sr)
    last_in = defektlast(x_in, sr)
    last_out = defektlast(x_out, sr)
    # D-K3-95/97-Blindheit: defektlast sieht tanh-Sättigung nicht — der
    # Sättigungskanal ist der Report-Only-Zweitkanal (§G191 (GEBOTE.md)),
    # er ändert KEIN Gate und KEINEN gerichteter_fall-Zweig.
    saett_in = saettigung_kanal(x_in)
    saett_out = saettigung_kanal(x_out)
    # Q0-Witness-Forschung (2026-11-14): jitter_artifacts ist für defektlast
    # blind — der Jitter-Kanal ist der zweite Report-Only-Zweitkanal
    # (§G191 (GEBOTE.md)), ebenfalls ohne Gate-Wirkung.
    jitter_in = jitter_kanal(x_in, sr)
    jitter_out = jitter_kanal(x_out, sr)

    delta_mos = round(mos_out - mos_in, 4)
    delta_last = round(last_out - last_in, 4)
    delta_saett = round(saett_out - saett_in, 4)
    delta_jitter = round(jitter_out - jitter_in, 4)
    # musik_mos: höher = besser ⇒ Regression, wenn der Ausgang DEUTLICH tiefer liegt.
    # defektlast: höher = schlechter ⇒ Regression, wenn der Ausgang DEUTLICH steigt.
    reg_mos = bool(delta_mos < -float(jnd))
    reg_last = bool(delta_last > float(jnd))
    # DRT-10 (Spec 26): Überrestauration = defektlast-Gewinn ≥ JND bei
    # gleichzeitigem musik_mos-Verlust ≥ JND (Jets-vinyl-Befund 2026-11-15:
    # defektlast −0,203 bei musik_mos −0,551 — die Regionen-Gates sehen diesen
    # Wohlklang-Verlust nicht). Der Lauf ist dann nicht aktivierungsfähig; der
    # Produktions-Verdikt (``drt10_verdict_in_metadata``) degradiert den Export,
    # sobald ein Reparatur-Kandidat (Witness-Kopf) aktiv ist.
    ueberrestauration = bool(delta_last <= -float(jnd) and delta_mos <= -float(jnd))

    grund: list[str] = []
    if reg_mos:
        grund.append(f"musik_mos {delta_mos:+.4f} < -{float(jnd):.2f}")
    if reg_last:
        grund.append(f"defektlast {delta_last:+.4f} > +{float(jnd):.2f}")
    if ueberrestauration:
        grund.append(
            f"UEBERRESTAURATION (DRT-10): defektlast {delta_last:+.4f} ≤ -{float(jnd):.2f}"
            f" bei musik_mos {delta_mos:+.4f} ≤ -{float(jnd):.2f}"
        )

    gerichtet = bool(reg_mos or reg_last or ueberrestauration)
    befund: dict[str, Any] = {
        "schema": SCHEMA,
        "verfuegbar": True,
        "report_only": True,
        "jnd": float(jnd),
        "metriken": ["musik_mos", "defektlast", "saettigung_kanal", "jitter_kanal"],
        "zertifikat": ZERTIFIKAT,
        "eingang": {
            "musik_mos": round(mos_in, 4),
            "defektlast": round(last_in, 4),
            "saettigung_kanal": saett_in,
            "jitter_kanal": jitter_in,
        },
        "ausgang": {
            "musik_mos": round(mos_out, 4),
            "defektlast": round(last_out, 4),
            "saettigung_kanal": saett_out,
            "jitter_kanal": jitter_out,
        },
        "delta": {
            "musik_mos": delta_mos,
            "defektlast": delta_last,
            "saettigung_kanal": delta_saett,
            "jitter_kanal": delta_jitter,
        },
        "regression": {"musik_mos": reg_mos, "defektlast": reg_last},
        "ueberrestauration": ueberrestauration,
        "gerichteter_fall": gerichtet,
        "hoer_instanz": "vorzulegen" if gerichtet else "keine_vorlage",
        "grund": grund,
        # Diagnose (keine Metrik): Pegel-/Längen-Delta erklärt einen Teil der
        # Messung (z. B. Clipping-Anteil der defektlast bei Lautheitssprung).
        "pegel_delta_db": round(_rms_db(x_out) - _rms_db(x_in), 2),
        "laenge_delta_samples": int(x_out.size - x_in.size),
    }
    if gerichtet:
        logger.warning(
            "👂 Wohlklang-Zeuge: gerichteter Fall — %s (der Hör-Instanz vorzulegen, kein Eingriff)",
            "; ".join(grund),
        )
    return befund


def zeuge_meta(
    eingang: Any,
    ausgang: Any,
    sample_rate: int,
    *,
    jnd: float = JND_TOLERANZ,
) -> dict[str, Any]:
    """Nie werfende Produktions-Fassung (§V6 (copilot-instructions.md)): Warnung + Grund statt Abbruch."""
    try:
        return messe_wohlklang_zeuge(eingang, ausgang, sample_rate, jnd=jnd)
    except Exception as exc:
        logger.warning(
            "👂 Wohlklang-Zeuge nicht messbar (%s: %s) — Lauf bleibt gültig, kein Eingriff (§V6 (copilot-instructions.md))",
            type(exc).__name__,
            exc,
        )
        return {
            "schema": SCHEMA,
            "verfuegbar": False,
            "report_only": True,
            "jnd": float(jnd),
            "gerichteter_fall": False,
            "hoer_instanz": "keine_aussage",
            "grund": f"{type(exc).__name__}: {exc}",
        }


def drt10_verdict_in_metadata(meta: dict[str, Any]) -> bool:
    """DRT-10-Produktions-Verdikt (Spec 26 DRT-10, 2026-11-17).

    **Produktions-Invariante für JEDEN Lauf** (M1-Verschärfung 2026-11-17):
    Überrestauration wird unabhängig von aktiven Reparatur-Kandidaten
    geflaggt — auch die Bestands-Pipeline darf Ganz-Titel nicht
    überrestaurieren (Jets-vinyl-Befund: defektlast −0,203 bei
    musik_mos −0,551).

    Rein auf Metadaten wirkend (§0c): kein Sample-Eingriff, kein Hardstop —
    bei Überrestauration wird der Lauf als ``degraded`` markiert, damit das
    Export-Gate das bestmögliche sichere Ergebnis mit Status „degraded"
    exportiert. Den **Identitäts-Rollback** entscheidet der Aufrufer mit der
    §7-Konfliktregel (Hörordnung): hohe Hör-Instanz ⇒ Messartefakt-Verdacht,
    kein Rollback (``_drt10_rollback_zulaessig`` in UV3).

    Idempotent: ein zweiter Aufruf erzeugt keinen doppelten fail_reasons-Eintrag.

    Returns:
        True, wenn der Lauf degradiert wurde (für Tests/Logik), sonst False.
    """
    befund = meta.get("wohlklang_zeuge")
    if not isinstance(befund, dict):
        return False
    if not befund.get("verfuegbar") or not befund.get("ueberrestauration"):
        return False

    _reasons = meta.setdefault("fail_reasons", [])
    if not any(isinstance(r, dict) and r.get("error_code") == DRT10_FAIL_REASON for r in _reasons):
        _reasons.append(
            {
                "component": "drt10_ganztitel_verdict",
                "severity": "degraded",
                "error_code": DRT10_FAIL_REASON,
                "exc_type": "ueberrestauration",
                "exc_msg": (
                    "Ganz-Titel-Zwei-Metrik-Never-worsen verletzt (Spec 26 DRT-10): "
                    "defektlast gewinnt ≥ JND bei musik_mos-Verlust ≥ JND — "
                    "Lauf ist nicht aktivierungsfähig (§0c-degradierter Export)"
                ),
            }
        )
    if str(meta.get("degradation_status", "") or "").strip().lower() in {"", "ok", "none"}:
        meta["degradation_status"] = "degraded"
    if not meta.get("fail_reason"):
        meta["fail_reason"] = DRT10_FAIL_REASON
    logger.warning(
        "👂 DRT-10-Verdikt: Überrestauration (Ganz-Titel-Zwei-Metrik) — "
        "Lauf als degraded markiert (%s, §0c, kein Sample-Eingriff durch den Zeugen)",
        DRT10_FAIL_REASON,
    )
    return True


def reparatur_nachweis_bericht(meta: dict[str, Any], defect_scores: Any) -> dict[str, Any]:
    """DRT-14 (Spec 26, 2026-11-17): Wirksamkeits-Gebot — Klassen-Bilanz je Lauf.

    Für jede erkannte Defektklasse mit Schaden-Score ≥ 0,5 wird ein
    Reparatur-Nachweis geführt (Regionen-Q aus ``defekt_delta`` ≥ 0,5) ODER die
    Klasse als ``unrepariert`` mit Ursache berichtet. Report-Only (Hörordnung
    §7) — kein Hard-Fail, kein Rollback allein wegen fehlender Wirksamkeit.
    """
    scores = defect_scores or meta.get("defect_scores") or {}
    if not isinstance(scores, dict) or not scores:
        return {
            "schema": "aurik.reparatur_nachweis/1",
            "status": "nicht_bewertbar",
            "grund": "keine_defect_scores",
            "nachweis": [],
            "unrepariert": [],
        }
    delta = meta.get("defekt_delta")
    klassen = (delta or {}).get("aggregate") if isinstance(delta, dict) else None
    if not isinstance(klassen, dict):
        klassen = {}
    nachweis: list[str] = []
    unrepariert: list[dict[str, Any]] = []
    for klasse, wert in sorted(scores.items()):
        try:
            score = float(wert)
        except (TypeError, ValueError):
            continue
        if score < 0.5:
            continue
        kq = klassen.get(klasse) if isinstance(klassen, dict) else None
        q = 0.0
        if isinstance(kq, dict):
            try:
                q = float(kq.get("reparaturquote_median") or kq.get("q") or 0.0)
            except (TypeError, ValueError):
                q = 0.0
        if q >= 0.5:
            nachweis.append(klasse)
        else:
            unrepariert.append({"klasse": klasse, "q": round(q, 3), "grund": "reparaturquote_median < 0.5"})
    return {
        "schema": "aurik.reparatur_nachweis/1",
        "status": "vollstaendig" if not unrepariert else "teilweise",
        "nachweis": sorted(nachweis),
        "unrepariert": unrepariert,
    }
