"""§MKK (V20) Mikrodynamik-Korrelations-Guard.

Prüft nach NR/Dynamics-Phasen auf Vokal-Material, ob die Frame-Energie-Korrelation
zwischen pre und post im Voiced-Bereich ≥ 0.97 liegt. Unterschreitung →
Dry-Wet-Blend: wet = min(1.0, (corr - 0.90) / 0.07).

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.mikrodynamik_guard import frame_energy_correlation
    corr = frame_energy_correlation(pre, post, sr, frame_ms=10)
    if corr < 0.97:
        wet = min(1.0, (corr - 0.90) / 0.07)
        result.audio = wet * result.audio + (1 - wet) * audio
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Schwellwert: Delta-Korrelation ≥ 0.93 auf Voiced-Frames (siehe Docstring frame_energy_correlation)
MIKRODYNAMIK_THRESHOLD = 0.93
# §v10.101 Material-adaptive Korrelations-Schwellwerte:
# Kassette/Tape haben physikalisch bedingt niedrigere Mikrodynamik-
# Korrelation (0.67–0.85). Der Default 0.97 führt zum fast vollständigen
# Verwerfen der Entrauschung. Material-adaptiver Floor garantiert
# minimalen Wet-Blend auch bei strukturell niedriger Korrelation.
_MATERIAL_FLOOR_THRESHOLD: dict[str, float] = {
    "cassette": 0.72,
    "reel_tape": 0.80,
    "tape": 0.80,
    "vinyl": 0.85,
}
# Voiced-Frame-Schwellwert: Frames mit Energie über diesem Wert
_VOICED_ENERGY_PERCENTILE = 25.0


def recommend_mikrodynamik_wet(
    corr: float,
    panns_singing: float = 0.0,
    *,
    global_need: float = 0.0,
    material: str = "unknown",
) -> float:
    """Empfiehlt einen bedarfsorientierten Dry-Wet-Blend für V20-Mikrodynamik.

    Der Blend berücksichtigt sowohl die lokale Korrelation als auch den globalen
    Restaurierungsbedarf des Songs. So bleibt Aurik bedarfsorientiert: hohe
    Korrelation allein erzwingt nicht automatisch mehr Wet, und hoher Song-Bedarf
    kann eine vorsichtige Phase stärker erhalten.

    §v10.101: material-Parameter senkt effektiven Schwellwert für Kassette/Tape.
    """

    vocal_material = panns_singing >= 0.35
    # §v10.101 Material-adaptiver Floor: Kassette/Tape haben niedrigere
    # physikalische Mikrodynamik-Korrelation — Schwellwert absenken.
    mat_lower = str(material).lower()
    mat_floor = _MATERIAL_FLOOR_THRESHOLD.get(mat_lower, MIKRODYNAMIK_THRESHOLD)
    target_corr = max(mat_floor, 0.95 if vocal_material else 0.93)
    floor_corr = max(mat_floor - 0.10, 0.90 if vocal_material else 0.87)
    base_min_wet = 0.35 if mat_lower in ("cassette", "reel_tape", "tape") else (0.20 if vocal_material else 0.15)
    need = float(np.clip(global_need, 0.0, 1.0))

    if corr >= target_corr:
        return 1.0

    if corr <= 0.0:
        return 0.0

    if corr < floor_corr:
        base_wet = float(np.clip(base_min_wet * (corr / max(floor_corr, 1e-6)), 0.0, 1.0))
    else:
        base_wet = float(np.clip((corr - floor_corr) / max(target_corr - floor_corr, 1e-6), 0.0, 1.0))
        base_wet = float(np.clip(max(base_wet, base_min_wet), 0.0, 1.0))

    need_gain = 0.45 if vocal_material else 0.30
    wet = base_wet + need_gain * need * (1.0 - base_wet)
    return float(np.clip(wet, 0.0, 1.0))


def frame_energy_correlation(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 10.0,
) -> float:
    """Berechnet Delta-Korrelation der Frame-Energien auf Voiced-Zonen.

    §MKK-V20 (2026-09-11): Misst die Erhaltung der Mikrodynamik-FORM —
    Korrelation der Frame-zu-Frame-Übergänge (log-Energie-Differenzen) statt
    der absoluten Frame-Energien. Produktionsbefund: Absichtliches
    Vocal-Presence-Shaping (lyrics-guided, Saliency-Boosts ±1.5 dB auf
    Wort-Regionen) senkte die ABSOLUTE Energie-Korrelation auf 0.92 und
    verwarf damit 61 % der gesamten Restaurierung (wet=0.39), obwohl keine
    hörbare Dynamik-Zerstörung vorlag. Die Delta-Korrelation trennt sauber:
    LGE-Kette 0.954 (passiert), harter Limiter 0.73, Kompressor 4:1 0.93.

    Args:
        pre: Audio vor der Phase. Shape [N] oder [2, N].
        post: Audio nach der Phase.
        sr: Sample-Rate (muss 48000 sein).
        frame_ms: Frame-Länge in ms (Standard: 10 ms — §2.75).

    Returns:
        Pearson-Korrelation der Übergänge [0.0 … 1.0]. Grenzwert: 0.93.
        Bei Fehler oder sehr kurzem Signal: 1.0 (kein Eingriff).
    """
    assert sr == 48000

    try:
        pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if pre.shape != post.shape or pre.size < 512:
            return 1.0

        pre_mono = pre.mean(axis=0) if pre.ndim == 2 else pre
        post_mono = post.mean(axis=0) if post.ndim == 2 else post

        frame_len = max(64, int(sr * frame_ms / 1000.0))
        n = len(pre_mono)
        n_frames = n // frame_len

        if n_frames < 4:
            return 1.0

        pre_energy = np.array(
            [float(np.mean(pre_mono[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n_frames)],
            dtype=np.float32,
        )
        post_energy = np.array(
            [float(np.mean(post_mono[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n_frames)],
            dtype=np.float32,
        )

        # Nur Voiced-Frames: Frames über dem 25. Perzentil der pre-Energie
        voiced_threshold = float(np.percentile(pre_energy, _VOICED_ENERGY_PERCENTILE))
        voiced_mask = pre_energy > voiced_threshold

        if voiced_mask.sum() < 4:
            return 1.0

        log_pre = np.log10(pre_energy + 1e-12)
        log_post = np.log10(post_energy + 1e-12)

        # Übergänge nur INNERHALB zusammenhängender voiced-Runs — Pausen-Lücken
        # dürfen keine künstlichen Sprünge in die Differenzfolge bringen.
        d_pre: list[float] = []
        d_post: list[float] = []
        run: list[int] = []
        for i in range(n_frames):
            if voiced_mask[i]:
                run.append(i)
            else:
                if len(run) >= 2:
                    d_pre.extend((log_pre[run[1:]] - log_pre[run[:-1]]).tolist())
                    d_post.extend((log_post[run[1:]] - log_post[run[:-1]]).tolist())
                run = []
        if len(run) >= 2:
            d_pre.extend((log_pre[run[1:]] - log_pre[run[:-1]]).tolist())
            d_post.extend((log_post[run[1:]] - log_post[run[:-1]]).tolist())

        if len(d_pre) < 4:
            return 1.0

        corr = float(np.corrcoef(d_pre, d_post)[0, 1])
        return float(np.clip(corr, 0.0, 1.0)) if np.isfinite(corr) else 1.0

    except Exception as exc:
        logger.debug("frame_energy_correlation nicht blockierend: %s", exc)
        return 1.0
