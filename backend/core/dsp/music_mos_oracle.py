"""Musik-natives MOS-Orakel (KP-2, Pfad A — KEIN Training).

Ersetzt rollenmäßig das sprachtrainierte UTMOSv2-Orakel (BVCC) im
Restaurierungs-Regelkreis durch **musiknative** MOS-Schätzer aus lokalen
Assets (§III.11 (copilot-instructions.md)):

- **MuQ + A1-Eval-Head** (`plugins/muq_plugin.estimate_muq_mos`,
  MuQ-Eval 2026: SRCC 0.957 gegen menschliche MOS auf frozen Features) —
  der einzige richtungsvalidierte Musik-Zeuge (3/3 MUSDB-Paare,
  `scripts/validate_muq_plugin_direction.py`).
- **SingMOS Pro** (`plugins/versa_plugin.get_versa_plugin().score`,
  wav2vec2-large-ft Gesangs-MOS, ONNX `models/singmos/singmos_pro.onnx`).

Rolle: **report-only Zeuge** im Witness-Pfad. Dieses Modul misst und
protokolliert das Musik-MOS-DELTA (Referenz → Kandidat) IMMER — es übt
NIE ein Veto-Recht aus, flippt keine Flags und mutiert kein Kalibrierungs-
Profil (Hörordnung §8: Metriken sind Zeugen, die Hör-Instanz entscheidet;
AGENTS.md §3: Guard-Kalibrierung delta-basiert). Die Veto-Freigabe hängt
weiterhin am musik-kalibrierten MOS (`music_model_flags.use_utmos_music`,
F8) und steht nach Kalibrierung + menschlicher Hörabnahme aus (§v10.802
(copilot-instructions.md), D-K4-4 `.github/SOTA_DEFICIT_REGISTER.md`).

Determinismus (§G5 (copilot-instructions.md)): keine Zeitquellen in der
Entscheidungslogik; die Inferenz selbst ist modell-deterministisch
(gleiche Gewichte + gleicher Input). Silent-Failure (§V6 (copilot-instructions.md)): jedes fehlende Modell/Head liefert einen
neutralen Wert NUR mit ``logger.warning`` + Begründung — ein fehlender
Messwert darf nie bestrafen und nie eine Verbesserung vortäuschen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# MOS-Skala 1..5. Delta-Schwelle analog zum UTMOS-Witness (utmos_delta_gate)
# — die musiknative Kalibrierung (Kalibrier-Schwelle je Degradationsklasse)
# steht noch aus (KP-2 §3/Pfad A, docs/SOTA_TRAINING_KONZEPT.md).
_MOS_MIN = 1.0
_MOS_MAX = 5.0


@dataclass
class MusicMOSWitness:
    """Deterministisches Ergebnis des report-only Musik-MOS-Orakels.

    Ein Ergebnis ist immer ein ZEUGNIS — ``applied`` ist konstant False.
    Die Felder transportieren nur die gemessenen Scores/Deltas für die
    Protokollierung (report-only, Hörordnung §8).
    """

    stage: str = ""
    oracle: str = "music_mos"
    muq_mos_ref: float | None = None
    muq_mos_candidate: float | None = None
    muq_delta: float | None = None
    singmos_ref: float | None = None
    singmos_candidate: float | None = None
    singmos_delta: float | None = None
    composite_delta: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        """Report-only: ein Musik-MOS-Zeuge übt nie ein Veto aus."""
        return False


def _clip_mos(value: float | None) -> float | None:
    """Clip auf die MOS-Skala [1, 5]; None bleibt None (kein Messwert)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return float(np.clip(f, _MOS_MIN, _MOS_MAX))


def _downmix(audio: np.ndarray) -> np.ndarray:
    """Stereo-Layout-Invariante (§11 Spec 02): (C, N) und (N, C) → Mono float32.

    ``audio[:, 0]``/``mean(axis=0)`` ohne Layout-Check kollabiert auf
    C Samples (Produktionsbefund: Export ``(2,)``) — deshalb beide Layouts
    explizit bedienen und auf float32 normalisieren.
    """
    arr: np.ndarray = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    res: np.ndarray
    if arr.ndim <= 1:
        res = arr.astype(np.float32)
    elif arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
        # kanonisch channels-first (C, N); (N, C) ebenfalls bedienen (wie muq_plugin).
        res = arr.mean(axis=0).astype(np.float32)
    else:
        res = arr.mean(axis=1).astype(np.float32)
    return res


def _muq_mos(audio: np.ndarray, sr: int) -> float | None:
    """MuQ+A1-Head MOS 1–5; None wenn Head/Modell/GPU nicht verfügbar (§V6 (copilot-instructions.md))."""
    try:
        from plugins.muq_plugin import estimate_muq_mos  # pylint: disable=import-outside-toplevel

        mos = estimate_muq_mos(audio, sr)
        if mos is None:
            logger.warning(
                "MuQ+A1-MOS nicht messbar (Head/Modell/GPU fehlt) — Musik-MOS-Zeuge übersprungen "
                "(§V6 (copilot-instructions.md))"
            )
            return None
        return _clip_mos(mos)
    except Exception as _exc:  # pylint: disable=broad-except
        logger.warning(
            "MuQ+A1-MOS-Schätzung fehlgeschlagen (%s) — Musik-MOS-Zeuge neutral (§V6 (copilot-instructions.md))",
            _exc,
        )
        return None


def _singmos_pro(audio: np.ndarray, sr: int) -> float | None:
    """SingMOS-Pro MOS 1–5 über das VERSA-ONNX; None bei nicht-vokalem Material (§V6 (copilot-instructions.md)).

    ``versa_plugin.score()`` verlangt 48 kHz — vorher deterministisch
    resamplen (§G5 (copilot-instructions.md), resample_poly).

    Akzeptiert AUSSCHLIESSLICH ``model_used == "singmos_pro"`` (reines
    SingMOS-Pro-ONNX). ``singmos_pqs_blend`` und ``pqs_dsp_fallback`` sind
    NICHT-musiknative Schätzer (DSP-Proxies) und werden als kein Messwert
    behandelt — der Musik-MOS-Zeuge darf ein DSP-Proxy nicht als
    musiknatives SingMOS-Pro ausgeben.
    """
    try:
        from plugins.versa_plugin import get_versa_plugin  # pylint: disable=import-outside-toplevel

        audio48 = audio
        sr48 = sr
        if sr != 48000:
            from math import gcd  # pylint: disable=import-outside-toplevel

            from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

            g = gcd(sr, 48000)
            audio48 = resample_poly(audio, 48000 // g, sr // g)
            sr48 = 48000
        plugin = get_versa_plugin()
        # versa_plugin.score() verlangt 48 kHz (assert sr == 48_000).
        result = plugin.score(audio48, sr48)
        if result.model_used != "singmos_pro":
            # §V6 (copilot-instructions.md): SingMOS Pro nicht ladbar oder
            # vocal-gated → PQS-DSP-Fallback/Blend. Für den musiknativen
            # MOS-Zeugen ist das ein anderer (NICHT-musiknativer) Schätzer —
            # als Zeugenwert ungeeignet, deshalb None + Begründung.
            logger.warning(
                "SingMOS Pro nicht verfügbar (model_used=%s) — SingMOS-Zeuge übersprungen "
                "(§V6 (copilot-instructions.md))",
                result.model_used,
            )
            return None
        return _clip_mos(result.mos)
    except Exception as _exc:  # pylint: disable=broad-except
        logger.warning(
            "SingMOS-Pro-Schätzung fehlgeschlagen (%s) — SingMOS-Zeuge neutral (§V6 (copilot-instructions.md))",
            _exc,
        )
        return None


def _composite_delta(muq_delta: float | None, singmos_delta: float | None) -> float | None:
    """Kalibrierter Verbund-Delta; None wenn keine Orakel-Komponente gemessen hat.

    Deterministischer Mittelwert der vorhandenen Komponenten-Deltas. Die
    eigentliche Schwellen-Kalibrierung (KP-2 §3/Pfad A) steht aus; bis dahin
    ist dies ein reiner Beobachtungswert ohne Veto-Semantik.
    """
    deltas = [d for d in (muq_delta, singmos_delta) if d is not None]
    if not deltas:
        return None
    return round(float(np.mean(deltas)), 4)


def music_mos_witness(
    reference: np.ndarray,
    candidate: np.ndarray,
    sr: int,
    stage: str = "music_mos_gate",
) -> MusicMOSWitness:
    """Berechnet das report-only Musik-MOS-Delta (Referenz → Kandidat).

    Pure Funktion (mutiert nichts): misst MuQ+A1 und SingMOS Pro auf beiden
    Signalen, bildet die Deltas und liefert ein ``MusicMOSWitness`` zurück.
    ``applied`` ist immer False — kein Veto, kein Flag-Flip, keine
    Profil-Mutation. Rückgabe ist NIE None (auch ohne Messwert), damit der
    Aufrufer den Zeugen-Status immer protokollieren kann.
    """
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    w = MusicMOSWitness(stage=stage, oracle="music_mos")

    mos_ref_muq = _muq_mos(ref, sr)
    mos_cand_muq = _muq_mos(cand, sr)
    mos_ref_sing = _singmos_pro(ref, sr)
    mos_cand_sing = _singmos_pro(cand, sr)

    w.muq_mos_ref = mos_ref_muq
    w.muq_mos_candidate = mos_cand_muq
    w.singmos_ref = mos_ref_sing
    w.singmos_candidate = mos_cand_sing

    if mos_ref_muq is not None and mos_cand_muq is not None:
        w.muq_delta = round(mos_cand_muq - mos_ref_muq, 4)
    if mos_ref_sing is not None and mos_cand_sing is not None:
        w.singmos_delta = round(mos_cand_sing - mos_ref_sing, 4)

    w.composite_delta = _composite_delta(w.muq_delta, w.singmos_delta)
    if w.composite_delta is None:
        w.reasons.append("kein musiknativer MOS-Messwert verfügbar (MuQ+A1 und/oder SingMOS Pro fehlt)")
    return w


def music_mos_witness_log(witness: MusicMOSWitness) -> str:
    """Formatiert einen Musik-MOS-Zeugen-Befund als EINE Log-Zeile (report-only)."""
    parts: list[str] = []
    for label, ref, cand, delta in (
        ("muq", witness.muq_mos_ref, witness.muq_mos_candidate, witness.muq_delta),
        ("singmos", witness.singmos_ref, witness.singmos_candidate, witness.singmos_delta),
    ):
        if delta is not None:
            parts.append(f"{label}_delta={delta:+.3f} (ref={ref:.2f}, cand={cand:.2f})")
    comp = witness.composite_delta
    comp_s = f"{comp:+.3f}" if comp is not None else "n/a"
    reasons = "; ".join(witness.reasons) if witness.reasons else "—"
    return (
        f"musikMOS zeuge (report-only, stage={witness.stage}): composite_delta={comp_s}, "
        f"{', '.join(parts) if parts else 'keine Komponente gemessen'}, reasons=[{reasons}]"
    )


def measure_and_log_music_mos(
    reference: np.ndarray,
    candidate: np.ndarray,
    sr: int,
    stage: str = "feedback_chain_music_mos",
) -> MusicMOSWitness:
    """Report-only Convenience: misst UND protokolliert das Musik-MOS-Delta.

    Kein Veto, kein Flag-Flip (§III.11 (copilot-instructions.md)) — der
    Zeuge wird über ``logger.info`` „immer gemessen und protokolliert“
    (Hörordnung §8). Rückgabe ist der Witness für optionale Metadaten-
    Anreicherung durch den Aufrufer.
    """
    witness = music_mos_witness(reference, candidate, sr, stage=stage)
    logger.info("%s", music_mos_witness_log(witness))
    return witness
