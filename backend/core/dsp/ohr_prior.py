"""Ohr-Prior — personalisiertes Hör-Orakel über Auriks vorhandene Metriken.

Ziel (2026-11-09, „Mein Ohr"): Aus der Summe der bereits gemessenen
psychoakustischen Größen ein Modell des persönlichen Hörurteils bauen, das
sich mit wachsender Urteilszahl **selbst trägt** — als kalibrierter
Bradley-Terry-Head über Feature-DIFFERENZEN von A/B-Paaren.

Feature-Basis (ausschließlich vorhandene Messungen, kein neues Messmodul):
  musik_mos (zertifiziert Spearman 0,95), defektlast (kanonische Definition
  aus wohlklang_zeuge), roughness_asper, sharpness_acum, loudness_phon,
  fluctuation_vacil, sii_index — je als **Qualitäts-Wert** auf [0,1]
  normalisiert (höher = besser).

Richtungs-Kontrakt (Fix 2026-11-09): `defektlast` ist per Definition
höher = mehr Defekte, `roughness`/`sharpness`/`fluctuation` sind
höher = auffälliger. Ein positiver Prior auf dem ROHWERT hätte dem Ohr
also „mehr Defekte sind besser“ beigebracht — der Vektor dreht diese
Größen deshalb in Qualität um. `loudness_phon` ist nicht monoton: bewertet
wird die Nähe zu einem musik-typischen Band (70 phon ± 50); der gelernte
Fit darf das überstimmen. Die Richtungen sind in `FEATURE_RICHTUNG`
dokumentiert und durch `test_feature_qualitaets_richtung` gepinnt.

Modell: P(A gewinnt) = sigmoid(w · Δf). Bayesianische Mischung gegen
Überanpassung bei wenigen Urteilen:
  w_eff = (n·w_learned + pseudo_count·w_prior) / (n + pseudo_count)
Bei n=0 gilt der Prior (die zwei richtungszertifizierten Metriken dominieren);
mit n→∞ trägt sich das gelernte Ohr selbst. Deterministisch (§G5 (copilot-instructions.md)):
Gradient Descent mit festem Seed, keine
Zeit-Abhängigkeit, kein stochastisches Sampling.

Rolle: Zeuge, nicht Richter — das Orakel liefert P(A gewinnt) für die
§7-Konfliktregel und personalisierte Gate-Kalibrierung; die harte Grenze
bleibt das Never-worsen-Gate und die finale Hör-Entscheidung des Menschen.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = "aurik.ohr_prior/2"
FEATURE_NAMEN = [
    "musik_mos",
    "defektlast",
    "roughness_asper",
    "sharpness_acum",
    "loudness_phon",
    "fluctuation_vacil",
    "sii",
]
# Richtung je Feature: "hoeher_besser" = Rohwert ist bereits Qualität,
# "niedriger_besser" = Rohwert wird in 1 - x gedreht, "band" = Nähe zum Ziel.
FEATURE_RICHTUNG: dict[str, str] = {
    "musik_mos": "hoeher_besser",
    "defektlast": "niedriger_besser",
    "roughness_asper": "niedriger_besser",
    "sharpness_acum": "niedriger_besser",
    "loudness_phon": "band",
    "fluctuation_vacil": "niedriger_besser",
    "sii": "hoeher_besser",
}
# Musik-typisches Loudness-Band für die Band-Güte (Prior, kein Gate):
# 70 phon Ziel, 50 phon halbe Bandbreite ⇒ 20 bzw. 120 phon = 0.0.
LOUDNESS_ZIEL_PHON = 70.0
LOUDNESS_BAND_PHON = 50.0
# Start-Prior: die zwei richtungszertifizierten Metriken dominieren; die
# übrigen sind feine persönliche Korrekturen (Summe = 1).
OHR_PRIOR_WEIGHTS = np.array([0.40, 0.30, 0.05, 0.04, 0.08, 0.03, 0.10], dtype=np.float64)
PSEUDO_COUNT_DEFAULT = 10
FIT_SEED = 42


def _clamp01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def band_qualitaet(wert: float, ziel: float, halbe_bandbreite: float) -> float:
    """Qualität einer band-bewerteten Größe: 1 im Ziel, 0 am Bandrand."""
    if halbe_bandbreite <= 0.0:
        return 1.0 if wert == ziel else 0.0
    return float(max(0.0, 1.0 - min(1.0, abs(float(wert) - float(ziel)) / float(halbe_bandbreite))))


def qualitaets_vektor(rohwerte: dict[str, float]) -> np.ndarray:
    """Roh-Messwerte → Qualitäts-Vektor (höher = besser), rein & testbar.

    `rohwerte` verwendet die Namen aus `FEATURE_NAMEN` (Rohwerte bereits auf
    [0,1] normiert). Fehlende Werte gelten als 0.5 (neutral), damit der
    Vektor nie wirft (§V6 (copilot-instructions.md)).
    """
    out = np.zeros(len(FEATURE_NAMEN), dtype=np.float64)
    for idx, name in enumerate(FEATURE_NAMEN):
        roh = _clamp01(float(rohwerte.get(name, 0.5)))
        richtung = FEATURE_RICHTUNG[name]
        if richtung == "niedriger_besser":
            out[idx] = 1.0 - roh
        elif richtung == "band":
            out[idx] = band_qualitaet(
                roh,
                LOUDNESS_ZIEL_PHON / 100.0,
                LOUDNESS_BAND_PHON / 100.0,
            )
        else:
            out[idx] = roh
    return cast(np.ndarray, out)


def feature_vector(audio: np.ndarray, sr: int) -> np.ndarray:
    """Qualitäts-Vektor (höher = besser) aus den vorhandenen Messgrößen.

    Rohwerte werden hier nur gemessen und auf [0,1] normiert; die
    Richtungs-Drehung passiert zentral in `qualitaets_vektor`.
    Fail-safe je Feature (§V6 (copilot-instructions.md)): Fehler ⇒ 0.5 mit
    Debug-Log — der Vektor wirft nie.
    """
    roh: dict[str, float] = dict.fromkeys(FEATURE_NAMEN, 0.5)
    try:
        from plugins.musik_mos_plugin import get_musik_mos

        roh["musik_mos"] = _clamp01(float(get_musik_mos().estimate_mos(audio, sr).mos) / 5.0)
    except Exception as exc:
        logger.debug("ohr_prior: musik_mos nicht verfügbar: %s", exc)
    try:
        from backend.core.dsp.wohlklang_zeuge import defektlast

        roh["defektlast"] = _clamp01(float(defektlast(audio, sr)))
    except Exception as exc:
        logger.debug("ohr_prior: defektlast nicht verfügbar: %s", exc)
    try:
        from backend.core.dsp.zwicker_metrics import compute_roughness_asper

        roh["roughness_asper"] = _clamp01(float(compute_roughness_asper(audio, sr)) / 10.0)
    except Exception as exc:
        logger.debug("ohr_prior: roughness nicht verfügbar: %s", exc)
    try:
        from backend.core.inviting_sound_gate import compute_sharpness_acum

        roh["sharpness_acum"] = _clamp01(float(compute_sharpness_acum(audio, sr)) / 4.0)
    except Exception as exc:
        logger.debug("ohr_prior: sharpness nicht verfügbar: %s", exc)
    try:
        from backend.core.dsp.zwicker_loudness import compute_loudness_phon

        roh["loudness_phon"] = _clamp01(float(compute_loudness_phon(np.asarray(audio, dtype=np.float32), sr)) / 100.0)
    except Exception as exc:
        logger.debug("ohr_prior: loudness nicht verfügbar: %s", exc)
    try:
        from backend.core.dsp.zwicker_metrics import compute_fluctuation_strength_vacil

        roh["fluctuation_vacil"] = _clamp01(float(compute_fluctuation_strength_vacil(audio, sr)) / 10.0)
    except Exception as exc:
        logger.debug("ohr_prior: fluctuation nicht verfügbar: %s", exc)
    try:
        from backend.core.dsp.speech_intelligibility import sii_index

        roh["sii"] = _clamp01(float(sii_index(audio, sr)))
    except Exception as exc:
        logger.debug("ohr_prior: sii nicht verfügbar: %s", exc)
    return qualitaets_vektor(roh)


def feature_diff(a: np.ndarray, b: np.ndarray, sr: int) -> np.ndarray:
    """Differenz-Vektor f(A) − f(B) für den Bradley-Terry-Fit."""
    return cast(np.ndarray, feature_vector(a, sr) - feature_vector(b, sr))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


@dataclass
class OhrPriorFit:
    """Ergebnis der Ohr-Kalibrierung (Bradley-Terry mit Pseudo-Count-Prior)."""

    weights: np.ndarray = field(default_factory=lambda: OHR_PRIOR_WEIGHTS.copy())
    weights_prior: np.ndarray = field(default_factory=lambda: OHR_PRIOR_WEIGHTS.copy())
    weights_learned: np.ndarray = field(default_factory=lambda: OHR_PRIOR_WEIGHTS.copy())
    n_labels: int = 0
    pseudo_count: int = PSEUDO_COUNT_DEFAULT
    log_loss: float = 0.0
    seed: int = FIT_SEED

    def predict(self, a: np.ndarray, b: np.ndarray, sr: int) -> dict[str, float | str]:
        """P(A gewinnt) für ein Paar — Zeuge für §7, kein Richter."""
        d = feature_diff(a, b, sr)
        p_a = float(_sigmoid(np.dot(self.weights, d))[()])
        if p_a > 0.55:
            bevorzugt = "A"
        elif p_a < 0.45:
            bevorzugt = "B"
        else:
            bevorzugt = "unentschieden"
        return {"p_a_gewinnt": round(p_a, 4), "bevorzugt": bevorzugt}

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "feature_names": list(FEATURE_NAMEN),
            "weights": [round(float(w), 6) for w in self.weights],
            "weights_prior": [round(float(w), 6) for w in self.weights_prior],
            "weights_learned": [round(float(w), 6) for w in self.weights_learned],
            "n_labels": self.n_labels,
            "pseudo_count": self.pseudo_count,
            "log_loss": round(self.log_loss, 6),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OhrPriorFit:
        """Artefakt laden — Schema fail-closed (§V6 (copilot-instructions.md)).

        Schema `/1` hatte ROHWERTE (defektlast positiv gewichtet = „mehr
        Defekte besser"). Es wird bewusst NICHT interpretiert, sondern
        abgelehnt: ein altes Artefakt muss neu kalibriert werden.
        """
        if d.get("schema") != SCHEMA:
            raise ValueError(f"unerwartetes Ohr-Prior-Schema {d.get('schema')!r} (erwartet {SCHEMA!r})")
        return cls(
            weights=np.asarray(d["weights"], dtype=np.float64),
            weights_prior=np.asarray(d["weights_prior"], dtype=np.float64),
            weights_learned=np.asarray(d["weights_learned"], dtype=np.float64),
            n_labels=int(d["n_labels"]),
            pseudo_count=int(d["pseudo_count"]),
            log_loss=float(d["log_loss"]),
            seed=int(d["seed"]),
        )


def fit_ohr_prior(
    pairs: list[tuple[np.ndarray, np.ndarray, int]],
    sr: int,
    pseudo_count: int = PSEUDO_COUNT_DEFAULT,
    seed: int = FIT_SEED,
) -> OhrPriorFit:
    """Bradley-Terry-Fit auf A/B-Urteilen mit Pseudo-Count-Prior.

    pairs: (audio_a, audio_b, label) mit label = 1 wenn A gewonnen, 0 wenn B.
    Deterministisch: Gradient Descent, feste Schritte, kein Sampling.
    """
    w = OHR_PRIOR_WEIGHTS.copy()
    n = len(pairs)
    fit = OhrPriorFit(pseudo_count=pseudo_count, seed=seed)
    if n == 0:
        fit.log_loss = 0.0
        return fit  # w_eff = Prior (n=0)
    xs = np.stack([feature_diff(a, b, sr) for a, b, _ in pairs])
    ys = np.asarray([float(label) for _, _, label in pairs], dtype=np.float64)
    lam = float(pseudo_count) / float(n)  # Ridge gegen den Prior
    lr = 0.05
    for _ in range(3000):
        p = _sigmoid(xs @ w)
        grad = xs.T @ (p - ys) / n + lam * (w - OHR_PRIOR_WEIGHTS)
        w = w - lr * grad
        w = np.clip(w, -5.0, 5.0)
    p = _sigmoid(xs @ w)
    eps = 1e-9
    log_loss = float(-np.mean(ys * np.log(p + eps) + (1.0 - ys) * np.log(1.0 - p + eps)))
    w_eff = (n * w + pseudo_count * OHR_PRIOR_WEIGHTS) / (n + pseudo_count)
    fit.weights_learned = w.copy()
    fit.weights = w_eff
    fit.n_labels = n
    fit.log_loss = log_loss
    return fit


def save_ohr_prior(fit: OhrPriorFit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fit.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_ohr_prior(path: Path) -> OhrPriorFit:
    return OhrPriorFit.from_dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = [
    "FEATURE_NAMEN",
    "FEATURE_RICHTUNG",
    "LOUDNESS_BAND_PHON",
    "LOUDNESS_ZIEL_PHON",
    "OHR_PRIOR_WEIGHTS",
    "OhrPriorFit",
    "band_qualitaet",
    "feature_vector",
    "feature_diff",
    "fit_ohr_prior",
    "save_ohr_prior",
    "load_ohr_prior",
    "qualitaets_vektor",
]
