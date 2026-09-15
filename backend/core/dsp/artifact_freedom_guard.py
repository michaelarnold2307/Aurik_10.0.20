"""§SOTA-P1 (2026-09-15) — af-Never-worsen-Guard (delta-basiert, billige Komponenten).

Roadmap P1-Folge-Slice: „gezielte Never-worsen-Fixes in 07/17/19/38“. Der
Diagnose-Lauf (``scripts/artifact_freedom_diagnosis.py``) zeigte, dass diese
Phasen den ``ArtifactDetector.overall_score`` (af) absenken — die Sub-Score-
Analyse lokalisiert den Mechanismus auf die KOMPONENTEN click + pre_echo
(phase_07: pre_echo 0,20→0,03; phase_19: click 0,74→0,27; phase_17/38:
click-Degradation). Der volle af-Scan kostet ≈0,72× RT (dominant:
``_detect_spectral_holes`` mit 14,4 s je 20 s) — produktionsuntauglich als
Per-Phase-Guard. Die billigen Komponenten ``_detect_clicks`` (0,11 s) +
``_detect_pre_echo`` (0,05 s) kosten zusammen ≈0,008× RT und bilden den
beobachteten Schaden ab.

Guard-Vertrag (deterministisch, layout-sicher, §V6 (copilot-instructions.md)
-fail-open, Hörordnung §4 Never-worsen):

- Proxy = (0,30·click + 0,25·pre_echo) / 0,55 — die af-Gewichte der beiden
  billigen Komponenten, auf 1 renormiert.
- Fällt der Proxy um mehr als ``tolerance`` (zentral, kein phasen-individueller
  Schwellwert — §V7 (copilot-instructions.md) Workaround-Verbot), wird
  proportional Richtung Input zurückgeblendet:
  ``wet = clip(1 − (Überschuss / tolerance), 0, 1)``.
- Rückgabe: (Audio, Metadaten) — ``af_guard_applied``, ``af_guard_wet``,
  ``af_proxy_pre/post``, ``af_delta``.

Autor: Aurik Testing Team
"""

from __future__ import annotations

import logging

import numpy as np

from backend.core.artifact_detector import ArtifactDetector

logger = logging.getLogger(__name__)

_CLICK_WEIGHT = 0.30
_PREECHO_WEIGHT = 0.25
_PROXY_SUM = _CLICK_WEIGHT + _PREECHO_WEIGHT
DEFAULT_TOLERANCE = 0.02


def _to_mono(x: np.ndarray) -> np.ndarray:
    """Layout-sicherer Mono-Downmix (Stereo-Invariante, AGENTS.md §3)."""
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            return np.asarray(arr.mean(axis=0), dtype=np.float32)  # type: ignore[no-any-return]
        return np.asarray(arr.mean(axis=1), dtype=np.float32)  # type: ignore[no-any-return]
    return np.asarray(arr.ravel(), dtype=np.float32)  # type: ignore[no-any-return]


def fast_af_proxy(mono: np.ndarray, sample_rate: int) -> float:
    """Billiger af-Proxy: nur click + pre_echo (≈0,008× RT je Messung)."""
    detector = ArtifactDetector(int(sample_rate))
    _click_count, click_score = detector._detect_clicks(mono)
    pre_echo_score = detector._detect_pre_echo(mono)
    proxy = (_CLICK_WEIGHT * float(click_score) + _PREECHO_WEIGHT * float(pre_echo_score)) / _PROXY_SUM
    return float(np.clip(proxy, 0.0, 1.0))


def af_fast_never_worsen(
    audio: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    tolerance: float = DEFAULT_TOLERANCE,
    enabled: bool = True,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Delta-basierter Never-worsen auf den billigen af-Proxy (fail-open)."""
    meta: dict[str, float | bool] = {
        "af_guard_applied": False,
        "af_guard_wet": 1.0,
        "af_proxy_pre": 1.0,
        "af_proxy_post": 1.0,
        "af_delta": 0.0,
    }
    if not enabled:
        return processed, meta
    try:
        pre_m = _to_mono(np.asarray(audio))
        post_m = _to_mono(np.asarray(processed))
        if len(pre_m) < 512 or len(post_m) < 512:
            return processed, meta  # zu kurz für die Detektoren — Passthrough
        proxy_pre = fast_af_proxy(pre_m, sample_rate)
        proxy_post = fast_af_proxy(post_m, sample_rate)
        delta = proxy_post - proxy_pre
        meta["af_proxy_pre"] = round(proxy_pre, 4)
        meta["af_proxy_post"] = round(proxy_post, 4)
        meta["af_delta"] = round(delta, 4)
        if delta >= -float(tolerance):
            return processed, meta
        excess = -(delta + float(tolerance))
        wet = float(np.clip(1.0 - excess / float(tolerance), 0.0, 1.0))
        out = audio + wet * (processed - audio)
        out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        meta["af_guard_applied"] = True
        meta["af_guard_wet"] = round(wet, 4)
        logger.warning(
            "§SOTA-P1 af-Never-worsen: Proxy %.3f → %.3f (Δ%+.3f) über Toleranz %.3f — wet=%.2f",
            proxy_pre,
            proxy_post,
            delta,
            float(tolerance),
            wet,
        )
        return out.astype(np.float32), meta
    except Exception as _afg_exc:  # §V6 (copilot-instructions.md): nie blockierend
        logger.debug("§SOTA-P1 af-Never-worsen nicht anwendbar: %s", _afg_exc)
        return processed, meta
