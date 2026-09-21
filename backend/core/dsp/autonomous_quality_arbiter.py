#!/usr/bin/env python3
"""Autonomer Never-worsen-Arbiter — referenzfreie Eingabe-vs-Ausgabe-Bewertung.

§v10.26 (2026-09-20): Aurik erkennt Verschlechterung eigenständig und wählt
autonom das beste Restaurierungsresultat für Wohlklang und Natürlichkeit —
ohne externe Eingriffe (Hörordnung: Never-worsen; Metriken sind Zeugen).

Produktionsbefund, der diesen Arbiter begründet: Quality-Mode-Kette auf
`corpus/cassette/damaged/cassette_hiphop_1980s_hiss.wav` → seg-SNR −14,4 dB,
ViSQOL 4,86→2,33 gegenüber der Eingabe. Einzelfälle wie dieser müssen am
Pipeline-Ende erkannt und autonom aufgelöst werden.

Metriken: `BlindQualityEstimator` (§G55a–g: spektrale Natürlichkeit,
Dynamik-Gesundheit, Noise-Floor-Kontinuität, HF-Präsenz, Stereo-Natürlichkeit,
Transienten-Dichte, MERT-perzeptuell). Deterministisch (§G5 (GEBOTE.md)) —
reine DSP-Merkmale + fester MERT-Pfad, keine Zeitstempel in der Logik.

Auflösungsleiter (autonom):
  1. output_score ≥ input_score − Marge → Ausgabe übernehmen.
  2. Sonst: Verschlechterung → Rücksprung auf die EINGABE (bit-identisch,
     Passthrough) — die Invariante „nie schlechter als das Original“ gilt
     immer, auch wenn der Pipeline-Ausgang abgelehnt wird.

Verwendung:
    from backend.core.dsp.autonomous_quality_arbiter import arbitrate
    decision = arbitrate(input_audio, output_audio, sr, material_key)
    final_audio = output_audio if not decision.reverted else input_audio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Margen (Punkte auf der 0–100-Skala des BlindQualityEstimator). Konservativ:
# nur eindeutige Regressionen lösen den Rücksprung aus.
_OVERALL_MARGIN = 4.0
_HF_MARGIN = 12.0
_TRANSIENT_MARGIN = 12.0
_SPECTRAL_MARGIN = 8.0
_DYNAMIC_MARGIN = 15.0
_NOISE_CONT_MARGIN = 12.0


@dataclass
class ArbitrationResult:
    """Entscheidung des autonomen Never-worsen-Arbiters."""

    chosen: str  # "output" | "input"
    reverted: bool
    input_score: float
    output_score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class LadderResolution:
    """Ergebnis der SOTA-Parameter-Retry-Leiter (§v10.26).

    status: "accepted" | "retry_balanced" | "reverted"
    chosen_audio: das autonom gewählte beste Resultat (Eingabe, Erstlauf
        oder Retry) — garantiert nie schlechter als die Eingabe.
    """

    status: str
    chosen_audio: np.ndarray
    input_score: float
    first_score: float
    retry_score: float | None = None
    reasons: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def metadata(self) -> dict[str, float | str]:
        """Maschinenlesbare Zusammenfassung für das Ergebnis-Metadatum."""
        md: dict[str, float | str] = {
            "status": self.status,
            "input": float(self.input_score),
            "first_out": float(self.first_score),
        }
        if self.retry_score is not None:
            md["retry_out"] = float(self.retry_score)
        return md


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Mono-Downmix für die Merkmalsextraktion.

    Stereo-Layout-Invariante (AGENTS.md): bedient BOTH channels-first (C, N)
    UND channels-last (N, C) — sonst kollabiert der BlindQualityEstimator auf
    C Samples und liefert degeneriert 50,0 (n < 4096-Fallback).
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        if a.shape[0] == 2:
            a = a.mean(axis=0)  # channels-first (C, N)
        elif a.shape[1] == 2:
            a = a.mean(axis=1)  # channels-last (N, C)
        else:
            a = a.mean(axis=0) if a.shape[0] < a.shape[1] else a.mean(axis=1)
    return cast(np.ndarray, np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))


def arbitrate(
    input_audio: np.ndarray,
    output_audio: np.ndarray,
    sr: int = 48000,
    material_key: str = "unknown",
    era_decade: int | None = None,
    *,
    margin: float = _OVERALL_MARGIN,
) -> ArbitrationResult:
    """Vergleicht Eingabe und Ausgabe referenzfrei; autonom entschieden.

    Returns:
        ArbitrationResult — bei `reverted=True` ist die EINGABE das beste
        Resultat (Ausgabe wird verworfen, §V6 (copilot-instructions.md)-Warnung mit Scores + Gründen).
    """
    from backend.core.blind_reference_free_quality import (  # deferred: MERT-Load
        BlindQualityEstimator,
    )

    estimator = BlindQualityEstimator(sr=sr, material_key=material_key, era_decade=era_decade)
    in_score = estimator.estimate(_to_mono(input_audio))
    out_score = estimator.estimate(_to_mono(output_audio))

    reasons = _compare(in_score, out_score, margin)
    reverted = bool(reasons)
    if reverted:
        logger.warning(
            "§v10.26 Never-worsen-Arbiter: Verschlechterung erkannt (%s) — "
            "autonomer Rücksprung auf die Eingabe (Eingabe=%.1f, Ausgabe=%.1f).",
            "; ".join(reasons),
            in_score.overall,
            out_score.overall,
        )
    else:
        logger.info(
            "§v10.26 Never-worsen-Arbiter: Ausgabe akzeptiert (Eingabe=%.1f, Ausgabe=%.1f).",
            in_score.overall,
            out_score.overall,
        )
    return ArbitrationResult(
        chosen="input" if reverted else "output",
        reverted=reverted,
        input_score=float(in_score.overall),
        output_score=float(out_score.overall),
        reasons=reasons,
    )


def _compare(in_score: Any, out_score: Any, margin: float) -> list[str]:
    """Vergleicht zwei Scores; liefert die Gründe für eine Verschlechterung.

    Erwartet Score-Objekte mit den Feldern des BlindQualityScore (§G55a–g).
    `Any`, weil der Scorer injizierbar ist (Tests nutzen Fake-Objekte).
    """
    reasons: list[str] = []
    if out_score.overall < in_score.overall - margin:
        reasons.append(f"overall {out_score.overall:.1f} < {in_score.overall:.1f}-{margin}")
    if out_score.hf_presence < in_score.hf_presence - _HF_MARGIN:
        reasons.append(f"hf {out_score.hf_presence:.1f} < {in_score.hf_presence:.1f}-{_HF_MARGIN}")
    if out_score.transient_density < in_score.transient_density - _TRANSIENT_MARGIN:
        reasons.append(
            f"transients {out_score.transient_density:.1f} < {in_score.transient_density:.1f}-{_TRANSIENT_MARGIN}"
        )
    if out_score.spectral_naturalness < in_score.spectral_naturalness - _SPECTRAL_MARGIN:
        reasons.append(
            f"spectral {out_score.spectral_naturalness:.1f} < {in_score.spectral_naturalness:.1f}-{_SPECTRAL_MARGIN}"
        )
    if out_score.dynamic_range_health < in_score.dynamic_range_health - _DYNAMIC_MARGIN:
        reasons.append(
            f"dynamic {out_score.dynamic_range_health:.1f} < {in_score.dynamic_range_health:.1f}-{_DYNAMIC_MARGIN}"
        )
    if out_score.noise_floor_continuity < in_score.noise_floor_continuity - _NOISE_CONT_MARGIN:
        reasons.append(
            f"noise_floor {out_score.noise_floor_continuity:.1f} < "
            f"{in_score.noise_floor_continuity:.1f}-{_NOISE_CONT_MARGIN}"
        )
    return reasons


def resolve_never_worsen(
    input_audio: np.ndarray,
    first_audio: np.ndarray,
    retry_fn: Any,
    sr: int = 48000,
    material_key: str = "unknown",
    era_decade: int | None = None,
    *,
    scorer: Any | None = None,
    margin: float = _OVERALL_MARGIN,
) -> LadderResolution:
    """SOTA-Parameter-Retry-Leiter (§v10.26): autonom das beste Resultat wählen.

    Ablauf:
      1. Erstlauf referenzfrei gegen die Eingabe prüfen.
      2. Verschlechterung → `retry_fn()` aufrufen (z. B. balanced-Pass mit
         anderen Parametern) und den Retry erneut prüfen.
      3. Retry nicht schlechter als die Eingabe → Retry übernehmen
         (`status="retry_balanced"`).
      4. Sonst → Eingabe (`status="reverted"`, bit-identischer Passthrough).

    Args:
        input_audio:  ursprüngliche Eingabe (Referenz).
        first_audio:  Ausgabe des Erstlaufs.
        retry_fn:     Callable ohne Argumente → Retry-Audio oder None
                      (Retry fehlgeschlagen/unverfügbar).
        scorer:       Optional injizierbarer Scorer callable(audio)->Score mit
                      den Feldern overall/hf_presence/transient_density/
                      spectral_naturalness/dynamic_range_health/
                      noise_floor_continuity (Default: BlindQualityEstimator).
    """
    if scorer is None:
        from backend.core.blind_reference_free_quality import (  # deferred: MERT-Load
            BlindQualityEstimator,
        )

        estimator = BlindQualityEstimator(sr=sr, material_key=material_key, era_decade=era_decade)
        scorer = estimator.estimate

    in_score = scorer(_to_mono(input_audio))
    first_score = scorer(_to_mono(first_audio))
    first_reasons = _compare(in_score, first_score, margin)
    if not first_reasons:
        return LadderResolution(
            status="accepted",
            chosen_audio=np.asarray(first_audio, dtype=np.float32),
            input_score=float(in_score.overall),
            first_score=float(first_score.overall),
            note="Never-worsen-Arbiter: Ausgabe akzeptiert.",
        )

    logger.warning(
        "§v10.26 Never-worsen: Erstlauf schlechter als Eingabe (%s) — autonome Parameter-Wiederholung.",
        "; ".join(first_reasons),
    )
    retry_audio: np.ndarray | None = None
    retry_score: Any | None = None
    try:
        _candidate = retry_fn()
        if _candidate is not None:
            retry_audio = np.asarray(_candidate, dtype=np.float32)
            retry_score = scorer(_to_mono(retry_audio))
    except Exception as retry_exc:  # pylint: disable=broad-except — §V6 (copilot-instructions.md): Eingabe bleibt Netz
        logger.warning("§v10.26 Parameter-Wiederholung fehlgeschlagen: %s", retry_exc)
        retry_audio = None

    if retry_audio is not None and retry_score is not None and not _compare(in_score, retry_score, margin):
        return LadderResolution(
            status="retry_balanced",
            chosen_audio=retry_audio,
            input_score=float(in_score.overall),
            first_score=float(first_score.overall),
            retry_score=float(retry_score.overall),
            reasons=first_reasons,
            note=(
                "Never-worsen-Arbiter: Verschlechterung autonom erkannt — "
                "Balanced-Retry lieferte das bessere Resultat (aus eigener Kraft)."
            ),
        )

    logger.warning(
        "§v10.26 Never-worsen-Arbiter: Rücksprung auf die Eingabe (Eingabe=%.1f, Erstlauf=%.1f).",
        in_score.overall,
        first_score.overall,
    )
    return LadderResolution(
        status="reverted",
        chosen_audio=np.asarray(input_audio, dtype=np.float32),
        input_score=float(in_score.overall),
        first_score=float(first_score.overall),
        retry_score=float(retry_score.overall) if retry_score is not None else None,
        reasons=first_reasons,
        note=(
            f"Never-worsen-Arbiter: Verschlechterung autonom erkannt ({first_reasons[0]}) — auf Eingabe zurückgesetzt."
        ),
    )
