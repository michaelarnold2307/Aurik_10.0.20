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
Rollback aus, blockiert keinen Export und setzt kein Flag. Er schreibt
``metadata["wohlklang_zeuge"]`` und markiert ``gerichteter_fall`` = True,
sobald eine Metrik über die JND hinaus in die falsche Richtung läuft — solche
Läufe sind der Hör-Instanz vorzulegen (``§V7`` (copilot-instructions.md):
kein stiller Flip, keine automatische Gegenmaßnahme).

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

    delta_mos = round(mos_out - mos_in, 4)
    delta_last = round(last_out - last_in, 4)
    # musik_mos: höher = besser ⇒ Regression, wenn der Ausgang DEUTLICH tiefer liegt.
    # defektlast: höher = schlechter ⇒ Regression, wenn der Ausgang DEUTLICH steigt.
    reg_mos = bool(delta_mos < -float(jnd))
    reg_last = bool(delta_last > float(jnd))

    grund: list[str] = []
    if reg_mos:
        grund.append(f"musik_mos {delta_mos:+.4f} < -{float(jnd):.2f}")
    if reg_last:
        grund.append(f"defektlast {delta_last:+.4f} > +{float(jnd):.2f}")

    gerichtet = bool(reg_mos or reg_last)
    befund: dict[str, Any] = {
        "schema": SCHEMA,
        "verfuegbar": True,
        "report_only": True,
        "jnd": float(jnd),
        "metriken": ["musik_mos", "defektlast"],
        "zertifikat": ZERTIFIKAT,
        "eingang": {"musik_mos": round(mos_in, 4), "defektlast": round(last_in, 4)},
        "ausgang": {"musik_mos": round(mos_out, 4), "defektlast": round(last_out, 4)},
        "delta": {"musik_mos": delta_mos, "defektlast": delta_last},
        "regression": {"musik_mos": reg_mos, "defektlast": reg_last},
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
