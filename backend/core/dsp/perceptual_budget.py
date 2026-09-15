"""§SOTA-R1 (Wahrnehmungs-Budget-Bilanz) — JND-Budget je Messung + Summenbericht.

Roadmap R1 (2026-09-14): „Jede Phase verbraucht ein JND-Budget; die Kette
bilanziert die Gesamt-Hörbarkeit (wie ein Wahrnehmungs-Wasserzeichen)."
Damit wird „Klangtreu" messbar statt Absichtserklärung.

Dieses Modul misst den Wahrnehmungs-Delta zwischen zwei Signalen in
JND-Einheiten (``hearing_jnd``, PSY-A8) für vier robuste Fähigkeiten:
``level_broadband`` (RMS-Pegel, 1 dB), ``loudness_ratio`` (Zwicker-Loudness,
0,25 LU), ``iacc`` (interaurale Bündelung, 0,08 — nur Stereo),
``frequency_1khz`` (Spektral-Centroid-Verschiebung, 0,2 % relativ).

Deterministisch (§G5 (GEBOTE.md)), layout-sicher, NaN-geschützt (§0a).
Per-Phase-Aufruf: ``measure_perceptual_budget(pre, post, sr, phase=...)``;
Ketten-Bilanz: ``summarize_budget([...])``.

Autor: Aurik Testing Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from backend.core.dsp.hearing_jnd import jnd

logger = logging.getLogger(__name__)


@dataclass
class PerceptualBudgetReport:
    """Wahrnehmungs-Budget einer Einzelmessung (Phase- oder Pipeline-Ebene)."""

    phase: str | None = None
    jnd_units: dict[str, float] = field(default_factory=dict)  # verbrauchte JND-Einheiten
    audible_changes: dict[str, bool] = field(default_factory=dict)  # über JND hörbar
    total_jnd_units: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "jnd_units": {k: round(v, 3) for k, v in self.jnd_units.items()},
            "audible_changes": dict(self.audible_changes),
            "total_jnd_units": round(self.total_jnd_units, 3),
        }


def _to_mono(x: np.ndarray) -> np.ndarray:
    """Layout-sicherer Mono-Downmix (Stereo-Invariante, AGENTS.md §3).

    Bedient channels-first (2, N), channels-last (N, 2) und Mono (N,) —
    ``mean(axis=0)`` auf channels-last würde auf C Samples kollabieren
    (Produktionsbefund: Export (2,)).
    """
    arr = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            return np.asarray(arr.mean(axis=0), dtype=np.float64)  # type: ignore[no-any-return]
        return np.asarray(arr.mean(axis=1), dtype=np.float64)  # type: ignore[no-any-return]
    return np.asarray(arr.ravel(), dtype=np.float64)  # type: ignore[no-any-return]


def _rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12))


def _spectral_centroid_hz(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    denom = float(np.sum(spec)) + 1e-12
    return float(np.sum(freqs * spec) / denom)


def _iacc(x_stereo: np.ndarray) -> float | None:
    """Interaurale Kreuzkorrelation bei Lag 0 (IACC-Proxy), nur Stereo."""
    if x_stereo.ndim != 2 or min(x_stereo.shape) != 2:
        return None
    if x_stereo.shape[0] == 2:
        l, r = x_stereo[0], x_stereo[1]
    else:
        l, r = x_stereo[:, 0], x_stereo[:, 1]
    denom = float(np.sqrt(np.dot(l, l) * np.dot(r, r))) + 1e-12
    return float(np.dot(l, r) / denom)


def measure_perceptual_budget(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    phase: str | None = None,
) -> PerceptualBudgetReport:
    """Misst den Wahrnehmungs-Delta pre→post in JND-Einheiten (fail-closed).

    Fehler in einer Einzelgröße entfernen nur diese Größe (fail-closed,
    §V6 (copilot-instructions.md): die Bilanz bleibt nutzbar).
    """
    units: dict[str, float] = {}
    audible: dict[str, bool] = {}
    try:
        pre_m = _to_mono(pre)
        post_m = _to_mono(post)
        n = min(len(pre_m), len(post_m))
        if n >= 8:
            pre_m, post_m = pre_m[:n], post_m[:n]
            # Pegel (Breitband)
            level_delta_db = abs(_rms_db(post_m) - _rms_db(pre_m))
            units["level_broadband"] = level_delta_db / jnd("level_broadband")
            audible["level_broadband"] = level_delta_db > jnd("level_broadband")
            # Lautheit (Zwicker)
            try:
                from backend.core.dsp.zwicker_loudness import compute_loudness_sone as _zl

                n_pre = float(_zl(pre_m, sr))
                n_post = float(_zl(post_m, sr))
                loudness_delta_lu = abs(n_post - n_pre)
                units["loudness_ratio"] = loudness_delta_lu / jnd("loudness_ratio")
                audible["loudness_ratio"] = loudness_delta_lu > jnd("loudness_ratio")
            except Exception as _zl_exc:
                logger.debug("Wahrnehmungsbilanz: Zwicker-Lautheit nicht verfügbar (%s)", _zl_exc)
            # Spektral-Centroid (Frequenz-JND relativ)
            c_pre = _spectral_centroid_hz(pre_m, sr)
            c_post = _spectral_centroid_hz(post_m, sr)
            if c_pre > 1.0:
                rel_shift = abs(c_post - c_pre) / c_pre
                units["frequency_1khz"] = rel_shift / jnd("frequency_1khz")
                audible["frequency_1khz"] = rel_shift > jnd("frequency_1khz")
            # IACC (Stereo)
            i_pre = _iacc(np.asarray(pre))
            i_post = _iacc(np.asarray(post))
            if i_pre is not None and i_post is not None:
                iacc_delta = abs(i_post - i_pre)
                units["iacc"] = iacc_delta / jnd("iacc")
                audible["iacc"] = iacc_delta > jnd("iacc")
    except Exception as _exc:
        logger.warning("Wahrnehmungsbilanz: Messung fehlgeschlagen (%s) — leere Bilanz", _exc)
    return PerceptualBudgetReport(
        phase=phase,
        jnd_units=units,
        audible_changes=audible,
        total_jnd_units=float(sum(units.values())),
    )


def summarize_budget(reports: list[PerceptualBudgetReport]) -> dict[str, object]:
    """Summenbericht über eine Kette von Einzelbilanzen (Wahrnehmungs-Wasserzeichen).

    Returns:
        total_jnd_units (über alle Fähigkeiten), per_fähigkeit-Summen,
        top_consumers (größte Einzelverbraucher je Fähigkeit), n_reports.
    """
    totals: dict[str, float] = {}
    top: dict[str, tuple[str, float]] = {}
    n_reports = 0
    for r in reports:
        if r is None:
            continue
        n_reports += 1
        for kind, v in r.jnd_units.items():
            totals[kind] = totals.get(kind, 0.0) + v
            if kind not in top or v > top[kind][1]:
                top[kind] = (r.phase or "unbekannt", v)
    return {
        "n_reports": n_reports,
        "total_jnd_units": round(float(sum(totals.values())), 3),
        "per_kind_jnd_units": {k: round(v, 3) for k, v in sorted(totals.items())},
        "top_consumers": {k: {"phase": p, "jnd_units": round(v, 3)} for k, (p, v) in top.items()},
    }
