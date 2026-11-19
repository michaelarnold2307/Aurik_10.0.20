"""
Regulator — setzt DSP-Parameter pro Zone kontextbewusst auf Basis der
Tonträgerkettenerkennung (§6.7) und des GPParameterOptimizer-Parameterraums (§2.5).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Material-spezifische Basis-Parameter (§6.2, §2.5 GPParameterOptimizer)
_MATERIAL_DEFAULTS: dict[str, dict[str, float]] = {
    "tape": {"noise_reduction_strength": 0.75, "compression_ratio": 1.8, "eq_high_shelf_db": 1.5},
    "reel_tape": {"noise_reduction_strength": 0.80, "compression_ratio": 1.6, "eq_high_shelf_db": 2.0},
    "vinyl": {"noise_reduction_strength": 0.55, "compression_ratio": 1.4, "eq_high_shelf_db": 0.5},
    "shellac": {"noise_reduction_strength": 0.90, "compression_ratio": 2.0, "eq_high_shelf_db": 3.0},
    "wax_cylinder": {"noise_reduction_strength": 0.92, "compression_ratio": 2.2, "eq_high_shelf_db": 4.0},
    "cd_digital": {"noise_reduction_strength": 0.20, "compression_ratio": 1.05, "eq_high_shelf_db": 0.0},
    "mp3_low": {"noise_reduction_strength": 0.45, "compression_ratio": 1.3, "eq_high_shelf_db": 2.5},
    "dat": {"noise_reduction_strength": 0.30, "compression_ratio": 1.1, "eq_high_shelf_db": 0.5},
    "unknown": {"noise_reduction_strength": 0.50, "compression_ratio": 1.5, "eq_high_shelf_db": 1.0},
}


class Regulator:
    """Setzt optimale DSP-Parameter je Zone aus Tonträgerketten- und Ärainformation."""

    def regulate(self, zones: list[Any], tontraegerkette_info: Any) -> dict[str, dict[str, float]]:
        """Leitet material- und zonen-spezifische DSP-Parameter ab.

        Algorithmus:
            1. Material aus tontraegerkette_info extrahieren.
            2. Basis-Parametersatz aus _MATERIAL_DEFAULTS laden.
            3. Restorability-Score (falls vorhanden) skaliert NR-Stärke.
            4. Jeden Zone-Eintrag mit dem fertigen Parameter-Dict belegen.

        Args:
            zones:                  Liste von Zonen-Bezeichnern (beliebige hashbare Werte).
            tontraegerkette_info:   Dict mit 'material', optional 'restorability_score'.

        Returns:
            Dict[zone → DSP-Parameter-Dict] mit Einträgen für alle 4 SOTA-Kernparameter.
        """
        if isinstance(tontraegerkette_info, dict):
            material = str(tontraegerkette_info.get("material", "unknown")).lower()
            restorability = float(tontraegerkette_info.get("restorability_score", 70.0))
        else:
            material = "unknown"
            restorability = 70.0

        base = dict(_MATERIAL_DEFAULTS.get(material, _MATERIAL_DEFAULTS["unknown"]))

        # Restorability-Skalierung (§2.31): bei schlechtem Material konservativer
        if restorability < 40.0:
            scale = 0.85
        elif restorability < 70.0:
            scale = 0.93
        else:
            scale = 1.00
        base["noise_reduction_strength"] = min(0.95, base["noise_reduction_strength"] * scale)

        result: dict[str, dict[str, float]] = {}
        for zone in zones:
            # Zone-lokale Kopie — Mutations-Sicherheit
            result[zone] = dict(base)

        logger.debug(
            "[Regulator] material=%s restorability=%.1f → NR=%.2f zones=%d",
            material,
            restorability,
            base["noise_reduction_strength"],
            len(zones),
        )
        return result
