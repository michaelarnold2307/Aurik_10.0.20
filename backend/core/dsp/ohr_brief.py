"""Ohr-Brief — zentrale Informations-Schnittstelle des Ohrs für die Denker.

Zweck (2026-11-09, „Mein Ohr"): Das Ohr versorgt die Denker-Schicht
(`denker/`) mit ALLEN verfügbaren Hör-Informationen aus EINER Quelle, damit
die Denker daraus Maßnahmen ableiten und zuverlässig umsetzen können.

Inhalt des Briefs:
  * Ohr-Prior (`models/ohr/ohr_prior.json`) — kalibrierte persönliche
    Präferenz-Gewichte (Bradley-Terry, `ohr_prior.py`); fehlt das Artefakt,
    ist der Brief dennoch gültig (verfuegbar=False, Fail-safe).
  * Hör-Instanz-Werte (Goosebumps, HPE) — von den Denkern berechnet und
    hier gebündelt.
  * musik_mos — Report-Only; bleibt 0/None bis zur Zertifikats-Freigabe.

Rolle: Zeuge, nicht Richter (§7 Hörordnung). Der Brief liefert Fakten und
optional Priorisierungs-Gewichte für die Sweet-Spot-Optimierung; Schwellen
(GREEN_ZONE) und die finale Hör-Entscheidung bleiben unverändert.

Determinismus (§G5 (copilot-instructions.md)): ohne Artefakt liefert
`ziel_gewichte()` None → die Denker verhalten sich bit-identisch zum
Bestand. Mit Artefakt sind die Gewichte eine deterministische Funktion der
kalibrierten Gewichte (kein `time.time()`, kein Sampling).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.core.dsp.ohr_prior import FEATURE_NAMEN, OhrPriorFit

logger = logging.getLogger(__name__)

OHR_ARTEFAKT = Path("models/ohr/ohr_prior.json")

# Mapping der 7 Ohr-Features (ohr_prior.FEATURE_NAMEN) auf die 9
# Sweet-Spot-Metriken (sweet_spot_optimizer.GREEN_ZONE). Jede Spalte summiert
# auf 1 — die Präferenz-Stärke des Ohrs wird anteilig verteilt.
_FEATURE_ZU_METRIK: dict[str, dict[str, float]] = {
    "musik_mos": {"hpe": 0.5, "inviting": 0.5},
    "defektlast": {"comb_filter": 0.5, "masking_health": 0.5},
    "roughness_asper": {"transparency": 0.5},
    "sharpness_acum": {"spectral_color": 1.0},
    "loudness_phon": {"musical_compression": 1.0},
    "fluctuation_vacil": {"microdynamics": 1.0},
    "sii": {"transparency": 0.5},
}

SWEET_SPOT_METRIKEN: tuple[str, ...] = (
    "hpe",
    "inviting",
    "transparency",
    "goosebumps",
    "comb_filter",
    "musical_compression",
    "masking_health",
    "spectral_color",
    "microdynamics",
)


@dataclass
class OhrBrief:
    """Alle Hör-Informationen, die die Denker für Maßnahmen brauchen."""

    ohr_prior: OhrPriorFit | None
    ohr_prior_verfuegbar: bool
    getragen: float = 0.0
    goosebumps: float | None = None
    hpe: float | None = None
    musik_mos: float | None = None
    musik_mos_zertifiziert: bool = False
    grund: str = ""

    def as_dict(self) -> dict:
        return {
            "ohr_prior_verfuegbar": self.ohr_prior_verfuegbar,
            "ohr_prior_n_labels": int(self.ohr_prior.n_labels) if self.ohr_prior else 0,
            "ohr_prior_getragen": round(self.getragen, 4),
            "ohr_stufe": self.ohr_stufe(),
            "goosebumps": self.goosebumps,
            "hpe": self.hpe,
            "musik_mos": self.musik_mos,
            "musik_mos_zertifiziert": self.musik_mos_zertifiziert,
            "grund": self.grund,
        }

    def ohr_stufe(self) -> int:
        """Reifungs-Stufe des Ohrs (§7, 2026-11-09, M3) — gestufte Aktivierung
        statt binärem Deploy:
          0 = kein Artefakt ⇒ alles neutral (bit-identisch zum Bestand)
          1 = getragen < 0.5  ⇒ Priorisierungs-Gewichte (ziel_gewichte)
          2 = getragen ≥ 0.5  ⇒ zusätzlich JND-Zonen-Kalibrierung (±5 %)
          3 = getragen ≥ 0.8  ⇒ verstärkte Kalibrierung (±8 %)
        Deterministisch; die Stufe steigt nie, wenn das Artefakt fehlt.
        """
        if not self.ohr_prior_verfuegbar or self.ohr_prior is None:
            return 0
        if self.getragen >= 0.8:
            return 3
        if self.getragen >= 0.5:
            return 2
        return 1

    def zonen_kalibrierung(self) -> dict[str, float] | None:
        """Stufen-2/3: relative Zonen-Deltas je Sweet-Spot-Metrik in [-cap, cap].

        Basis ist die PERSÖNLICHE Abweichung der kalibrierten Gewichte vom
        Default-Prior, auf die 9 Sweet-Spot-Metriken gemappt und summen-neutral
        (Umverteilung der Strenge, keine globale Lockerung). Metriken, die das
        Ohr stärker gewichtet, werden strenger (positives Delta).
        Stufe 1/0: None ⇒ Default-Zonen unverändert.
        """
        _stufe = self.ohr_stufe()
        if _stufe < 2 or self.ohr_prior is None:
            return None
        _cap = 0.05 if _stufe == 2 else 0.08
        _w = np.asarray(self.ohr_prior.weights, dtype=np.float64)
        _wp = np.asarray(self.ohr_prior.weights_prior, dtype=np.float64)
        if _w.size != len(FEATURE_NAMEN) or _wp.size != len(FEATURE_NAMEN):
            return None
        _norm = max(float(np.abs(_wp).sum()), 1e-9)
        _abw = (_w - _wp) / _norm  # relative Abweichung je Feature
        _roh: dict[str, float] = dict.fromkeys(SWEET_SPOT_METRIKEN, 0.0)
        for _idx, _name in enumerate(FEATURE_NAMEN):
            for _m, _anteil in _FEATURE_ZU_METRIK.get(_name, {}).items():
                _roh[_m] += float(_abw[_idx]) * _anteil
        _mean = sum(_roh.values()) / len(_roh)
        _neutral = {_m: float(_v - _mean) for _m, _v in _roh.items()}
        _skala = max(max(abs(v) for v in _neutral.values()), 1e-9)
        return {_m: round(_cap * (_v / _skala), 6) for _m, _v in _neutral.items()}

    def ziel_gewichte(self) -> dict[str, float] | None:
        """Priorisierungs-Gewichte für die Sweet-Spot-Optimierung.

        Map die kalibrierten Präferenz-Stärken (|w_j|) deterministisch auf die
        9 Sweet-Spot-Metriken; Summe = 1. Ohne Artefakt: None → Denker nutzen
        den unveränderten Default-Pfad (bit-identisch zum Bestand).
        """
        if not self.ohr_prior_verfuegbar or self.ohr_prior is None:
            return None
        w = np.abs(np.asarray(self.ohr_prior.weights, dtype=np.float64))
        if w.size != len(FEATURE_NAMEN):
            return None
        out = dict.fromkeys(SWEET_SPOT_METRIKEN, 0.0)
        for idx, name in enumerate(FEATURE_NAMEN):
            for metrik, anteil in _FEATURE_ZU_METRIK.get(name, {}).items():
                out[metrik] += float(w[idx]) * anteil
        _sum = sum(out.values())
        if _sum <= 0.0:
            return None
        gew = {m: round(v / _sum, 6) for m, v in out.items()}
        # Rundungsrest deterministisch auf die lexikografisch letzte Metrik —
        # damit die Summe EXAKT 1 ist (§G5 (copilot-instructions.md)).
        _rest = 1.0 - sum(gew.values())
        _letzte = sorted(gew)[-1]
        gew[_letzte] = round(gew[_letzte] + _rest, 6)
        return gew


def lade_ohr_prior(artefakt: Path | None = None) -> tuple[OhrPriorFit | None, str]:
    """Lädt das kalibrierte Ohr-Artefakt — Fail-safe, wirft nie.

    Returns:
        (fit, grund): fit ist None bei fehlendem/defektem Artefakt; grund
        nennt die Ursache ("kein_artefakt", "json_fehler", …) oder "".
    """
    pfad = artefakt if artefakt is not None else OHR_ARTEFAKT
    try:
        if not pfad.is_file():
            return None, "kein_artefakt"
        fit = OhrPriorFit.from_dict(json.loads(pfad.read_text(encoding="utf-8")))
        return fit, ""
    except Exception as exc:
        logger.debug("Ohr-Prior-Artefakt nicht ladbar (%s): %s", type(exc).__name__, exc)
        return None, type(exc).__name__


def erstelle_ohr_brief(
    goosebumps: float | None = None,
    hpe: float | None = None,
    musik_mos: float | None = None,
    musik_mos_zertifiziert: bool = False,
    artefakt: Path | None = None,
) -> OhrBrief:
    """Baut den Ohr-Brief für die Denker — wirft nie (§V6 (copilot-instructions.md)).

    artefakt: optionaler Artefakt-Pfad (Standard: models/ohr/ohr_prior.json).
    """
    fit, grund = lade_ohr_prior(artefakt)
    if fit is None:
        return OhrBrief(
            ohr_prior=None,
            ohr_prior_verfuegbar=False,
            grund=grund,
            goosebumps=goosebumps,
            hpe=hpe,
            musik_mos=musik_mos,
            musik_mos_zertifiziert=musik_mos_zertifiziert,
        )
    getragen = float(fit.n_labels) / (float(fit.n_labels) + float(fit.pseudo_count))
    return OhrBrief(
        ohr_prior=fit,
        ohr_prior_verfuegbar=True,
        getragen=getragen,
        goosebumps=goosebumps,
        hpe=hpe,
        musik_mos=musik_mos,
        musik_mos_zertifiziert=musik_mos_zertifiziert,
    )
