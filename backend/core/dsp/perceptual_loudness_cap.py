"""§SOTA-PSY-A7 (2026-09-15) — wahrnehmungs-basierter Loudness-Cap (Rollout-Helfer).

Muster: phase_47-``_perceptual_loudness_cap`` (Wave 2), hier als gemeinsamer
Helfer für die Phasen 10 (Compression), 11 (Limiting) und 40
(Loudness-Normalization): Diese Phasen dürfen die Kurzzeit-Lautheit
(peak STL, Sone) nicht über den Input hinaus anheben. Überschreitet der
Output die erlaubte Schwelle (Input-Lautheit × Headroom) um mehr als die
konservative Marge (``max(0,15 Sone, 5 % der Schwelle)``), wird die
Bearbeitung proportional Richtung Input zurückgenommen (linearer Blend —
Never-worsen, Hörordnung §4/§8a). Deterministisch, layout-sicher,
§V6 (copilot-instructions.md)-fail-open.

``headroom_lin`` = erlaubter Uniform-Gain über den Input:
- 10/11/47: 1,0 — nie über den Input anheben,
- phase_40: 10^(gain_db/20) — die Ziel-LUFS-Anhebung ist legitim,
  Kurzzeit-Pumping darüber (über den Uniform-Gain hinaus) wird gekappt.

Rückgabe: (Audio, Metadaten) — ``loudness_cap_applied``, ``loudness_cap_wet``,
``peak_stl_before_sone``, ``peak_stl_after_sone``.

Autor: Aurik Testing Team
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _to_mono(x: np.ndarray) -> np.ndarray:
    """Layout-sicherer Mono-Downmix (Stereo-Invariante, AGENTS.md §3)."""
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            return np.asarray(arr.mean(axis=0), dtype=np.float32)  # type: ignore[no-any-return]
        return np.asarray(arr.mean(axis=1), dtype=np.float32)  # type: ignore[no-any-return]
    return np.asarray(arr.ravel(), dtype=np.float32)  # type: ignore[no-any-return]


def perceptual_loudness_cap(
    audio: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    phase_label: str = "PSY-A7",
    headroom_lin: float = 1.0,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """peak-STL-Never-worsen mit interner Messung (ERB-Kurzzeit-Lautheit)."""
    meta: dict[str, float | bool] = {
        "loudness_cap_applied": False,
        "loudness_cap_wet": 1.0,
        "peak_stl_before_sone": 0.0,
        "peak_stl_after_sone": 0.0,
    }
    try:
        from backend.core.dsp.temporal_loudness import temporal_loudness

        _pre = temporal_loudness(_to_mono(np.asarray(audio)), int(sample_rate))
        _post = temporal_loudness(_to_mono(np.asarray(processed)), int(sample_rate))
        _stl_before = float(_pre.peak_stl_sone)
        _stl_after = float(_post.peak_stl_sone)
        meta["peak_stl_before_sone"] = round(_stl_before, 3)
        meta["peak_stl_after_sone"] = round(_stl_after, 3)
        _ceiling_stl = _stl_before * max(1.0, float(headroom_lin))
        _margin = max(0.15, 0.05 * max(_ceiling_stl, 1e-9))
        if _stl_after <= _ceiling_stl + _margin:
            return processed, meta
        _cap_wet = float(np.clip((_ceiling_stl + _margin) / max(_stl_after, 1e-9), 0.0, 1.0))
        capped = audio + _cap_wet * (processed - audio)
        capped = np.clip(np.nan_to_num(capped, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        logger.warning(
            "%s §SOTA-PSY-A7 Loudness-Cap: peak STL %.2f → %.2f Sone über Marge %.2f — wet=%.2f",
            phase_label,
            _stl_before,
            _stl_after,
            _margin,
            _cap_wet,
        )
        meta["loudness_cap_applied"] = True
        meta["loudness_cap_wet"] = round(_cap_wet, 3)
        meta["loudness_cap_headroom_lin"] = round(max(1.0, float(headroom_lin)), 4)
        return capped.astype(np.float32), meta
    except Exception as _plc_exc:  # §V6 (copilot-instructions.md): nie blockierend
        logger.debug("%s §SOTA-PSY-A7 Loudness-Cap nicht anwendbar: %s", phase_label, _plc_exc)
        return processed, meta
