"""
§v10 Grenzwert-Optimizer — Pleasantness-Maximum an physikalischen Limits.

Ein Toningenieur weiß genau, wie weit er gehen darf. Aurik kennt die
physikalischen Grenzen jedes Mediums und tastet sie systematisch ab,
um das Pleasantness-Maximum zu finden — nicht mehr, nicht weniger.

Physikalische Limits pro Medium:
  Shellac:    8 kHz Ceiling, SNR ~35 dB, Mono, Knistern inhärent
  Vinyl:     16 kHz, SNR ~55 dB, Stereo, leichte Rumpel
  Tape:      18 kHz, SNR ~60 dB, Stereo, Hiss inhärent
  Cassette:  15 kHz, SNR ~50 dB, Stereo, Dropout-Risiko
  CD/Digital: 22 kHz, SNR >90 dB, Stereo, kein inhärenter Defekt

Strategie:
  1. Definiere physikalisches Limit pro Dimension (Frequenz, SNR, Dynamik, Stereo)
  2. Starte 20% UNTERHALB des Limits (sicherer Startpunkt)
  3. Erhöhe in 10%-Schritten bis Pleasantness nicht mehr steigt
  4. Stoppe wenn: ΔP < 0.01 ODER physikalisches Limit erreicht
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Physikalische Limits pro Medium
MEDIA_LIMITS: dict[str, dict[str, float]] = {
    "shellac": {
        "freq_ceiling_hz": 8000.0,
        "snr_max_db": 35.0,
        "nr_max_reduction_db": 12.0,
        "stereo_width_max": 1.0,  # Mono
        "dynamic_range_max_db": 30.0,
        "eq_max_boost_db": 3.0,
    },
    "wax_cylinder": {
        "freq_ceiling_hz": 5000.0,
        "snr_max_db": 25.0,
        "nr_max_reduction_db": 8.0,
        "stereo_width_max": 1.0,
        "dynamic_range_max_db": 20.0,
        "eq_max_boost_db": 2.0,
    },
    "vinyl": {
        "freq_ceiling_hz": 16000.0,
        "snr_max_db": 55.0,
        "nr_max_reduction_db": 18.0,
        "stereo_width_max": 1.30,
        "dynamic_range_max_db": 50.0,
        "eq_max_boost_db": 4.0,
    },
    "reel_tape": {
        "freq_ceiling_hz": 18000.0,
        "snr_max_db": 60.0,
        "nr_max_reduction_db": 15.0,
        "stereo_width_max": 1.20,
        "dynamic_range_max_db": 55.0,
        "eq_max_boost_db": 3.5,
    },
    "cassette": {
        "freq_ceiling_hz": 15000.0,
        "snr_max_db": 50.0,
        "nr_max_reduction_db": 14.0,
        "stereo_width_max": 1.15,
        "dynamic_range_max_db": 45.0,
        "eq_max_boost_db": 3.0,
    },
    "cd_digital": {
        "freq_ceiling_hz": 20000.0,
        "snr_max_db": 90.0,
        "nr_max_reduction_db": 6.0,  # CD hat wenig Rauschen
        "stereo_width_max": 1.20,
        "dynamic_range_max_db": 80.0,
        "eq_max_boost_db": 2.0,  # CD ist bereits gut abgemischt
    },
    "dat": {
        "freq_ceiling_hz": 20000.0,
        "snr_max_db": 90.0,
        "nr_max_reduction_db": 4.0,
        "stereo_width_max": 1.20,
        "dynamic_range_max_db": 80.0,
        "eq_max_boost_db": 1.5,
    },
}


@dataclass
class BoundaryResult:
    """Ergebnis der Grenzwert-Optimierung."""

    dimension: str
    start_value: float
    best_value: float
    limit_value: float
    best_pleasantness: float
    steps_taken: int
    reason_stopped: str  # "limit_reached" oder "pleasantness_peak"


def optimize_to_boundary(
    pleasantness_fn,  # Callable: f(value) -> float
    dimension: str,
    start_value: float,
    limit_value: float,
    step_size: float | None = None,
    max_steps: int = 10,
) -> BoundaryResult:
    """Tastet eine Dimension von start bis limit ab, sucht Pleasantness-Maximum.

    Args:
        pleasantness_fn: Funktion die einen Parameter-Wert nimmt und P zurückgibt
        dimension: Name der Dimension (für Logging)
        start_value: Startwert (20% unter Limit)
        limit_value: Physikalisches Limit
        step_size: Schrittgröße (default: 10% der Spanne)
        max_steps: Maximale Schritte

    Returns:
        BoundaryResult mit optimalem Wert
    """
    if step_size is None:
        step_size = (limit_value - start_value) * 0.10

    best_value = start_value
    best_p = pleasantness_fn(start_value)
    current_value = start_value
    steps = 0

    logger.info(
        "BoundaryOpt %s: start=%.2f limit=%.2f step=%.2f initial_P=%.3f",
        dimension,
        start_value,
        limit_value,
        step_size,
        best_p,
    )

    while steps < max_steps:
        next_value = min(current_value + step_size, limit_value)
        if next_value <= current_value + step_size * 0.1:
            break  # Zu kleine Schritte → am Limit

        next_p = pleasantness_fn(next_value)
        steps += 1

        if next_p > best_p + 0.005:
            # Verbesserung → weiter
            best_p = next_p
            best_value = next_value
            current_value = next_value
            logger.debug("  step %d: %.2f → P=%.3f (↑)", steps, next_value, next_p)
        elif next_p < best_p - 0.01:
            # Verschlechterung → Peak gefunden, stopp
            logger.info(
                "  step %d: %.2f → P=%.3f (↓) — Peak bei %.2f (P=%.3f)",
                steps,
                next_value,
                next_p,
                best_value,
                best_p,
            )
            return BoundaryResult(
                dimension=dimension,
                start_value=start_value,
                best_value=best_value,
                limit_value=limit_value,
                best_pleasantness=best_p,
                steps_taken=steps,
                reason_stopped="pleasantness_peak",
            )
        else:
            # Neutral → weitermachen aber nicht als best speichern
            current_value = next_value
            logger.debug("  step %d: %.2f → P=%.3f (→)", steps, next_value, next_p)

    # Limit erreicht oder max steps
    return BoundaryResult(
        dimension=dimension,
        start_value=start_value,
        best_value=best_value,
        limit_value=limit_value,
        best_pleasantness=best_p,
        steps_taken=steps,
        reason_stopped="limit_reached",
    )


def get_media_limits(material: str) -> dict[str, float]:
    """Gibt die physikalischen Limits für ein Medium zurück."""
    material_key = material.lower().strip().replace("-", "_").replace(" ", "_")
    for key in MEDIA_LIMITS:
        if key in material_key or material_key in key:
            return dict(MEDIA_LIMITS[key])
    # Default: konservativ (Vinyl-ähnlich)
    logger.debug("BoundaryOpt: Material '%s' nicht erkannt, verwende vinyl-defaults", material)
    return dict(MEDIA_LIMITS["vinyl"])
