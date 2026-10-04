"""golden_ear_corpus — Golden-Ear-Korpus: Hörbarkeitsgrenzen pro Defektklasse.

Punkt 6 der Wohlklang-Roadmap (Vorgabe: „Wahrheit durch Ohren absichern").
Dieses Modul ist das kalibrierende Fundament der Hör-Instanz
(:mod:`backend.core.dsp.audibility_targets`): Für jede klassische
Defektklasse existieren

  1. ein deterministischer, geseedeter **Defekt-Generator** (Schellack-Sprung,
     Schellack-Knister, Bandklemmung, Digital-Dropout-Kette, Bandrauschen),
  2. eine **Rückführungs-Kalibrierung** ``inject_at_margin``: der Defekt wird
     auf eine Ziel-Hörbarkeits-Marge (Fassaden-Semantik) dosiert — Round-Trip
     injizieren → messen verankert jede Klasse gegen Messpfad-Bias,
  3. eine **Referenzrestaurierung** ``reference_pair`` (defective, clean,
     Defektstellen) als Never-worsen-Goldstandard für jede Reparaturfamilie,
  4. die dokumentierte **Hörbarkeits-Grenze** pro Klasse
     (``GOLDEN_EAR_THRESHOLDS``): Zielwert ist Marge 0 = „ab Maske hörbar"
     (Hörordnung §4). Status „vorbelegung_hoerpanel_offen" heißt ehrlich:
     Literatur-Vorbelegung — die Messung durch das kalibrierte Hörpanel
     (MUSHRA/ABX, GO_NO_GO_DECISION_PROTOCOL.md) ersetzt sie und deckt
     klassen-spezifische Fassaden-Abweichungen τ auf.

Warum das: Metriken sind Zeugen, das Ohr entscheidet. Ohne dokumentierte
Hörbarkeitsgrenzen pro Defektklasse optimiert eine Restaurierung gegen ihr
eigenes Messmodell statt gegen Gehörte. Der Korpus macht jede künftige
Verbesserung an menschlichen Grenzen messbar — nie an Mess-Null
(Wohlklang-Roadmap, klassischer Fehler 1).

Abgrenzung (Vorhandensein-Prüfung 2026-10-03): ``scripts/build_mushra_stimuli.py``
baut das Hörpanel-Stimuli-Set aus echten Trägern (Schnitt, LUFS-Abgleich,
Seed 42) und injiziert KEINE Defekte — dieser Korpus ist die komplementäre
synthetische Kalibrierungsschicht mit dosierbaren Defektklassen; die
62-DefectTypes-Taxonomie bleibt in defect_scanner kanonisch.

Mono-Vertrag (1-D); deterministisch (§G5 (GEBOTE.md)), NaN-sicher
(§0a (copilot-instructions.md)), fail-closed (§V6 (VERBOTEN.md)).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

#: Geprüfte Defektklassen (62-DefectTypes-Kanon, hier die klassischen
#: Hör-Kalibrierklassen der Golden-Ear-Suite).
DEFECT_CLASSES: tuple[str, ...] = (
    "shellac_sprung",
    "shellac_knister",
    "bandklemmung",
    "digital_dropout_kette",
    "bandrauschen",
)


@dataclass(frozen=True)
class GoldenEarThreshold:
    """Dokumentierte Hörbarkeits-Grenze einer Defektklasse.

    ``expected_margin_db`` ist die Fassaden-Marge, bei der Hörer den Defekt
    nach Literatur gerade wahrnehmen (0 = Fassade deckt sich mit „ab Maske
    hörbar"); ``tolerance_db`` ist die erlaubte Abweichung bis zur
    Kalibrierungs-Verfehlung. ``status`` unterscheidet ehrlich zwischen
    Vorbelegung und echter Hörpanel-Messung.
    """

    defect_class: str
    expected_margin_db: float = 0.0
    tolerance_db: float = 3.0
    source: str = "Hörordnung §4 / ISO 11172-3 (Literatur-Vorbelegung)"
    status: str = "vorbelegung_hoerpanel_offen"


GOLDEN_EAR_THRESHOLDS: dict[str, GoldenEarThreshold] = {
    name: GoldenEarThreshold(defect_class=name) for name in DEFECT_CLASSES
}


@dataclass
class InjectionResult:
    """Defekt-Injektion: defective = clean + scale·defect."""

    defective: np.ndarray
    defect_residual: np.ndarray
    region_mask: np.ndarray
    scale: float
    measured_margin_db: float
    target_margin_db: float
    defect_class: str = ""
    meta: dict[str, float] = field(default_factory=dict)


def _unit_defect(
    defect_class: str, clean: np.ndarray, sr: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministischer Defekt-Wellenform-Vorschlag (Peak 1) + Regionen-Maske.

    additiv: defective = clean + scale·defect — für dip-/gap-artige Klassen
    ist ``defect`` der negative Energie-Chunk (fehlendes Material).
    """
    n = int(clean.size)
    d = np.zeros(n, dtype=np.float64)
    region = np.zeros(n, dtype=bool)

    def _burst(center: int, width: int, decay_hz: float = 4000.0) -> None:
        s = max(0, center - width)
        e = min(n, center + 2 * width)
        seg = np.arange(e - s) / sr
        noise = rng.standard_normal(e - s)
        d[s:e] += np.exp(-decay_hz * seg) * noise
        region[s:e] = True

    if defect_class == "shellac_sprung":
        # 1–3 ms Sprünge: HF-tilt-Abkling-Impulse (Schellack-Sprung-Signatur)
        for _ in range(int(rng.integers(1, 4))):
            _burst(int(rng.integers(50, n - 50)), max(8, int(0.001 * sr)), 3500.0)
    elif defect_class == "shellac_knister":
        # dichte Impulse 0.5–2 ms
        for _ in range(int(rng.integers(12, 24))):
            _burst(int(rng.integers(50, n - 50)), max(6, int(0.0005 * sr)), 6000.0)
    elif defect_class == "bandklemmung":
        # 15–30 ms Dip (Bandklemmung) + Splice-Kante
        width = int(rng.uniform(0.015, 0.030) * sr)
        c = int(rng.integers(width, n - width))
        w = np.hanning(2 * width)
        d[c - width : c + width] -= clean[c - width : c + width] * w
        region[c - width : c + width] = True
        _burst(c + width, max(8, int(0.001 * sr)), 3500.0)
    elif defect_class == "digital_dropout_kette":
        # Kette aus 4–8 kurzen 2–8-ms-Aussetzern (digitaler Ketten-Ausfall)
        c = int(rng.integers(int(0.1 * n), int(0.6 * n)))
        for _ in range(int(rng.integers(4, 9))):
            width = int(rng.uniform(0.002, 0.008) * sr)
            e = min(n, c + width)
            d[c:e] -= clean[c:e]
            region[c:e] = True
            _burst(e, max(6, int(0.0005 * sr)), 6000.0)
            c = e + int(rng.uniform(0.004, 0.020) * sr)
            if c >= n - 32:
                break
    elif defect_class == "bandrauschen":
        # stationäres Bandrauschen +15 dB über Noise-Floor, 500-ms-Region
        width = min(n, int(0.5 * sr))
        c = int(rng.integers(width, n - width)) if n > 2 * width else 0
        seg = rng.standard_normal(width)
        d[c : c + width] += seg
        region[c : c + width] = True
    else:  # fail-closed (§V6 (VERBOTEN.md))
        raise ValueError(f"unbekannte Defektklasse: {defect_class}")

    peak = float(np.max(np.abs(d))) if d.size else 0.0
    if peak <= 1e-12:
        logger.warning("golden_ear_corpus: leere Defekt-Waveform für %s (§V6 (VERBOTEN.md))", defect_class)
        return d, region
    return d / peak, region


def inject_at_margin(
    clean: np.ndarray,
    sr: int,
    defect_class: str,
    target_margin_db: float,
    seed: int = 20261003,
    iterations: int = 6,
) -> InjectionResult:
    """Dosiert den Defekt auf eine Ziel-Hörbarkeits-Marge (Rückführung).

    Deterministische Newton-Iteration über die Fassade
    (``audibility_targets.is_audible``): gemessene Marge → Amplituden-
    Korrektur, geclippt, beste Annäherung gewinnt. Round-Trip-Stabilität
    (injizierte ≈ gemessene Marge) ist der Kalibrierungs-Regressionstest
    pro Defektklasse — er fängt Messpfad-Bias je Klasse (z. B. Fenster- oder
    Maskierungseffekte) ohne Hörpanel auf.
    """
    sig = np.nan_to_num(np.asarray(clean, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(seed)
    d, region = _unit_defect(defect_class, sig, sr, rng)

    from backend.core.dsp.audibility_targets import residual_audibility as is_audible

    scale = 0.1
    best = (float("inf"), 0.1, 0.0)
    measured = MARGIN_SILENCE_SENTINEL
    for _ in range(max(1, int(iterations))):
        residual = scale * d
        measured = float(is_audible(residual, sig, sr)["margin_db"])
        drift = abs(measured - float(target_margin_db))
        if drift < best[0]:
            best = (drift, scale, measured)
        if abs(measured - float(target_margin_db)) < 0.25:
            break
        step = 10.0 ** ((float(target_margin_db) - measured) / 20.0)
        scale = float(np.clip(scale * step, 1e-5, 5.0))
    _, best_scale, measured = best
    return InjectionResult(
        defective=np.asarray(sig + best_scale * d, dtype=np.float64),
        defect_residual=np.asarray(best_scale * d, dtype=np.float64),
        region_mask=region,
        scale=float(best_scale),
        measured_margin_db=float(measured),
        target_margin_db=float(target_margin_db),
        defect_class=defect_class,
    )


#: Fail-closed-Sentinel analog audibility_targets.MARGIN_SILENCE_DB
MARGIN_SILENCE_SENTINEL = -200.0


def reference_pair(
    clean: np.ndarray,
    sr: int,
    defect_class: str,
    margin_db: float = 6.0,
    seed: int = 20261003,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, InjectionResult]:
    """Never-worsen-Goldstandard: (defective, clean_reference, region, injection).

    Die ideale Restaurierung von ``defective`` IST ``clean_reference``; jede
    Reparaturkette wird daran gemessen (witness_chain Frage A an den
    Defektstellen, Frage B auf dem Residual). Hörbares Musikmaterial entfernt
    zu haben ist auch hier NIE OK.
    """
    inj = inject_at_margin(clean, sr, defect_class, margin_db, seed=seed)
    return inj.defective, np.asarray(clean, dtype=np.float64), inj.region_mask, inj


def calibration_report(
    clean: np.ndarray,
    sr: int,
    margins_db: tuple[float, ...] = (-6.0, 0.0, 4.0),
    seed: int = 20261003,
) -> list[dict[str, float | str | bool]]:
    """Rückführungs-Report je Klasse: Ziel vs. gemessene Marge + Verdikt.

    Grundlage des Golden-Ear-Kalibrierberichts, den das Hörpanel später um
    die menschliche Schwelle je Klasse ergänzt (Status-Felder in
    ``GOLDEN_EAR_THRESHOLDS``).
    """
    rows: list[dict[str, float | str | bool]] = []
    for name in DEFECT_CLASSES:
        for target in margins_db:
            inj = inject_at_margin(clean, sr, name, float(target), seed=seed)
            audible = bool(inj.measured_margin_db > 0.0)
            rows.append(
                {
                    "defect_class": name,
                    "target_margin_db": float(target),
                    "measured_margin_db": round(inj.measured_margin_db, 2),
                    "drift_db": round(inj.measured_margin_db - float(target), 2),
                    "audible": audible,
                    "threshold_status": GOLDEN_EAR_THRESHOLDS[name].status,
                }
            )
    return rows


__all__ = [
    "DEFECT_CLASSES",
    "GOLDEN_EAR_THRESHOLDS",
    "GoldenEarThreshold",
    "InjectionResult",
    "calibration_report",
    "inject_at_margin",
    "reference_pair",
]
