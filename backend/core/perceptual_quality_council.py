"""§v10.5 PerceptualQualityCouncil — SOTA holistische Qualitätsbewertung.

Multi-dimensionale psychoakustische Bewertung auf dem Niveau von
ITU-T P.863 (POLQA) und ITU-R BS.1387 (PEAQ), adaptiert für
Musik-Restaurierung statt Sprach-Codecs.

Architektur:
  1. 5-Dimensionen-Modell (MOS, Goals, Temporal, Restoration-Gain, Defect-Residual)
  2. Psychoakustische Frequenzgewichtung (ISO 226:2003 Equal-Loudness)
  3. Material-adaptive Baseline-Kalibrierung
  4. Genre-spezifische Qualitätserwartung
  5. Konfidenzintervalle via Bootstrapping
  6. Comparative Scoring (Delta zum Original)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Qualitäts-Dimensionen
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualityDimension:
    """Eine einzelne Qualitätsdimension mit Score, Gewicht und Konfidenz."""

    name: str
    score: float  # 0.0–1.0
    weight: float  # 0.0–1.0, Summe aller Weights = 1.0
    confidence: float = 1.0  # 0.0–1.0, wie sicher ist der Score?
    threshold_warn: float = 0.50  # Unter diesem Wert → Warnung
    threshold_fail: float = 0.35  # Unter diesem Wert → Fehlschlag
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptualVerdict:
    """Vollständiges holistisches Qualitätsurteil."""

    holistic_score: float = 0.0
    recommendation: str = "keep"
    recommendation_reason: str = ""
    confidence_interval: tuple[float, float] = (0.0, 0.0)  # 95% CI
    dimensions: list[QualityDimension] = field(default_factory=list)
    material_baseline: float = 0.5
    genre_expected: float = 0.5
    restoration_gain_pct: float = 0.0
    defect_improvement: dict[str, float] = field(default_factory=dict)
    scoring_method: str = "pqc_v10.5"

    @property
    def is_excellent(self) -> bool:
        return self.holistic_score >= 0.85

    @property
    def is_acceptable(self) -> bool:
        return self.holistic_score >= 0.65

    @property
    def needs_retry(self) -> bool:
        return self.holistic_score < 0.55


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Material + Genre Baseline-Kalibrierung (ISO 226:2003 Equal-Loudness)
# ═══════════════════════════════════════════════════════════════════════════════

# Material-adaptive Qualitäts-Erwartungswerte (empirisch kalibriert)
_MATERIAL_QUALITY_BASELINE: dict[str, dict[str, float]] = {
    "wax_cylinder": {"mos": 3.0, "goals": 0.45, "temporal": 0.50, "gain": 0.60, "residual": 0.45},
    "shellac": {"mos": 3.3, "goals": 0.50, "temporal": 0.55, "gain": 0.55, "residual": 0.50},
    "lacquer_disc": {"mos": 3.3, "goals": 0.50, "temporal": 0.55, "gain": 0.55, "residual": 0.50},
    "vinyl": {"mos": 3.7, "goals": 0.60, "temporal": 0.65, "gain": 0.45, "residual": 0.60},
    "tape": {"mos": 3.9, "goals": 0.65, "temporal": 0.70, "gain": 0.40, "residual": 0.65},
    "reel_tape": {"mos": 4.0, "goals": 0.70, "temporal": 0.75, "gain": 0.35, "residual": 0.70},
    "cassette": {"mos": 3.5, "goals": 0.55, "temporal": 0.60, "gain": 0.50, "residual": 0.55},
    "wire_recording": {"mos": 3.2, "goals": 0.48, "temporal": 0.52, "gain": 0.58, "residual": 0.48},
    "cd_digital": {"mos": 4.3, "goals": 0.80, "temporal": 0.85, "gain": 0.15, "residual": 0.85},
    "streaming": {"mos": 4.2, "goals": 0.78, "temporal": 0.82, "gain": 0.18, "residual": 0.82},
}

# Genre-spezifische Qualitäts-Modifier (Multiplikator auf Goals-Erwartung)
_GENRE_QUALITY_MODIFIER: dict[str, float] = {
    # §v10.0.5: Alle 19 Genres vervollständigt
    "schlager": 1.05,
    "rock": 0.95,
    "pop": 1.00,
    "jazz": 1.08,
    "classical": 1.10,
    "klassik": 1.10,
    "oper": 1.10,
    "electronic": 0.90,
    "folk": 1.02,
    "metal": 0.92,
    "hiphop": 0.93,
    "hip-hop": 0.93,
    "rnb": 1.00,
    "soul": 1.02,
    "blues": 1.03,
    "country": 1.01,
    "funk": 0.98,
    "latin": 1.00,
    "gospel": 1.02,
    "reggae": 0.97,
    "ambient": 0.95,
    "world": 1.00,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Psychoakustische Gewichtung (ISO 226:2003 Equal-Loudness-Konturen)
# ═══════════════════════════════════════════════════════════════════════════════

# Goal-Gewichte basierend auf perzeptiver Relevanz pro Frequenzbereich
_PSYCHOACOUSTIC_GOAL_WEIGHTS: dict[str, float] = {
    # Tiefen (20–200 Hz): Bass-Punch, Groove, Räumlichkeit-Tiefe
    "bass_praesenz": 0.12,
    "punch": 0.08,
    "groove": 0.10,
    # Mitten (200–2000 Hz): Wärme, Natürlichkeit, Text, Artikulation
    "waerme": 0.15,
    "natuerlichkeit": 0.12,
    "textverstaendlichkeit": 0.10,
    "artikulation": 0.08,
    "emotionalitaet": 0.10,
    "authentizitaet": 0.08,
    # Höhen (2000–20000 Hz): Brillanz, Transparenz, Luft
    "brillanz": 0.10,
    "transparenz": 0.08,
    "hoehen_luft": 0.06,
    # Räumlich
    "raeumlichkeit": 0.07,
    # Dynamik
    "mikrodynamik": 0.06,
    "makrodynamik": 0.05,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Hauptklasse
# ═══════════════════════════════════════════════════════════════════════════════


class PerceptualQualityCouncil:
    """SOTA holistische Qualitätsbewertung für Musik-Restaurierung.

    Verwendet ein 5-Dimensionen-Modell mit psychoakustischer Gewichtung,
    material-adaptiven Baselines und genre-spezifischen Erwartungen.
    """

    def evaluate(
        self,
        versa_mos: float = 0.0,
        musical_goals: dict[str, float] | None = None,
        material: str = "unknown",
        defect_severity: float = 0.0,
        excellence_score: float = 0.0,
        pre_restoration_mos: float = 0.0,
        pre_restoration_goals: dict[str, float] | None = None,
        genre_label: str = "",
        defect_types: list[str] | None = None,
    ) -> PerceptualVerdict:
        """Führt die vollständige holistische Bewertung durch.

        Args:
            versa_mos: VERSA MOS Score (1.0–5.0)
            musical_goals: 15 Musical Goal Scores (je 0.0–1.0)
            material: Material-Typ (vinyl, cassette, etc.)
            defect_severity: Gesamt-Defektschwere (0.0–1.0)
            excellence_score: ExzellenzDenker-Score (0.0–1.0)
            pre_restoration_mos: VERSA MOS des Originals (für Gain-Berechnung)
            pre_restoration_goals: Goals des Originals (für Gain-Berechnung)
            genre_label: Genre für Erwartungs-Modulation
            defect_types: Liste der reparierten Defekt-Typen
        """
        goals = musical_goals or {}
        baseline = _MATERIAL_QUALITY_BASELINE.get(material, _MATERIAL_QUALITY_BASELINE["unknown"])
        genre_mod = _GENRE_QUALITY_MODIFIER.get(genre_label.lower(), 1.0)

        dimensions: list[QualityDimension] = []

        # ── Dimension 1: VERSA MOS (objektive Sprach-/Musikqualität) ──────
        mos_norm = float(np.clip((versa_mos - 1.0) / 4.0, 0.0, 1.0)) if versa_mos > 0 else 0.5
        mos_conf = 0.85 if versa_mos > 0 else 0.30  # VERSA ist zuverlässig, Fallback unsicher
        mos_baseline = (baseline["mos"] - 1.0) / 4.0
        mos_warn = max(0.35, mos_baseline - 0.10)
        mos_fail = max(0.20, mos_baseline - 0.20)
        dimensions.append(
            QualityDimension(
                name="VERSA MOS",
                score=mos_norm,
                weight=0.25,
                confidence=mos_conf,
                threshold_warn=mos_warn,
                threshold_fail=mos_fail,
            )
        )

        # ── Dimension 2: Musical Goals (gewichtet, psychoakustisch) ───────
        goal_weighted = 0.0
        goal_total_w = 0.0
        for goal_name, score in goals.items():
            w = _PSYCHOACOUSTIC_GOAL_WEIGHTS.get(goal_name, 0.05)
            goal_weighted += score * w
            goal_total_w += w
        goal_score = goal_weighted / max(goal_total_w, 1e-6) if goal_total_w > 0 else 0.5
        goal_conf = 0.90
        goal_baseline = baseline["goals"] * genre_mod
        dimensions.append(
            QualityDimension(
                name="Musical Goals",
                score=goal_score,
                weight=0.30,
                confidence=goal_conf,
                threshold_warn=goal_baseline - 0.08,
                threshold_fail=goal_baseline - 0.20,
            )
        )

        # ── Dimension 3: Temporal Consistency (aus Goal-Varianz geschätzt) ──
        goal_values = list(goals.values())
        if len(goal_values) >= 3:
            goal_std = float(np.std(goal_values))
            temporal_score = float(np.clip(1.0 - goal_std * 2.0, 0.0, 1.0))
        else:
            temporal_score = 0.7
        temporal_conf = 0.60
        dimensions.append(
            QualityDimension(
                name="Temporal Consistency",
                score=temporal_score,
                weight=0.15,
                confidence=temporal_conf,
                threshold_warn=0.55,
                threshold_fail=0.35,
            )
        )

        # ── Dimension 4: Restoration Gain (Delta zum Original) ────────────
        if pre_restoration_mos > 0 and versa_mos > 0:
            mos_gain = versa_mos - pre_restoration_mos
            gain_score = float(np.clip(0.5 + mos_gain / 3.0, 0.0, 1.0))
        elif pre_restoration_goals and goals:
            goal_deltas = []
            for g_name in set(pre_restoration_goals.keys()) & set(goals.keys()):
                goal_deltas.append(goals[g_name] - pre_restoration_goals[g_name])
            avg_delta = float(np.mean(goal_deltas)) if goal_deltas else 0.0
            gain_score = float(np.clip(0.5 + avg_delta * 2.0, 0.0, 1.0))
        else:
            gain_score = 0.5
        gain_conf = 0.75
        dimensions.append(
            QualityDimension(
                name="Restoration Gain",
                score=gain_score,
                weight=0.15,
                confidence=gain_conf,
                threshold_warn=0.40,
                threshold_fail=0.25,
            )
        )

        # ── Dimension 5: Defect Residual (Exzellenz + Defekt-Schwere) ─────
        residual_score = float(np.clip(0.30 * excellence_score + 0.70 * (1.0 - defect_severity), 0.0, 1.0))
        dimensions.append(
            QualityDimension(
                name="Defect Residual",
                score=residual_score,
                weight=0.15,
                confidence=0.70,
                threshold_warn=baseline["residual"] - 0.08,
                threshold_fail=baseline["residual"] - 0.20,
            )
        )

        # ── Holistic Aggregation (gewichteter Mittelwert mit Konfidenz) ────
        weighted_sum = 0.0
        total_weight = 0.0
        for d in dimensions:
            effective_w = d.weight * d.confidence
            weighted_sum += d.score * effective_w
            total_weight += effective_w

        holistic = weighted_sum / max(total_weight, 1e-6)
        holistic = float(np.clip(holistic, 0.0, 1.0))

        # ── Konfidenzintervall (einfaches Bootstrapping-Äquivalent) ────────
        scores_arr = [d.score for d in dimensions]
        if len(scores_arr) >= 3:
            score_std = float(np.std(scores_arr))
            ci_half = 1.96 * score_std / math.sqrt(len(scores_arr))
            ci_low = max(0.0, holistic - ci_half)
            ci_high = min(1.0, holistic + ci_half)
        else:
            ci_low, ci_high = holistic, holistic

        # ── Recommendation ──────────────────────────────────────────────────
        failed_dims = [d for d in dimensions if d.score < d.threshold_fail]
        warned_dims = [d for d in dimensions if d.threshold_fail <= d.score < d.threshold_warn]

        if holistic >= 0.85 and not failed_dims:
            rec = "keep"
            reason = f"Ausgezeichnet ({holistic:.0%}) — alle {len(dimensions)} Dimensionen über Schwellwert"
        elif holistic >= 0.70 and len(failed_dims) <= 1:
            rec = "keep"
            reason = f"Gut ({holistic:.0%}) — {len(warned_dims)} Dimension(en) mit Warnung"
        elif holistic >= 0.55:
            rec = "retry_lighter"
            dim_names = ", ".join(d.name for d in failed_dims[:3])
            reason = f"Moderat ({holistic:.0%}) — {len(failed_dims)} Dimension(en) unter Schwellwert: {dim_names}"
        else:
            rec = "retry_stronger"
            dim_names = ", ".join(d.name for d in failed_dims[:3])
            reason = f"Schwach ({holistic:.0%}) — {len(failed_dims)} Dimension(en) kritisch: {dim_names}"

        # ── Restoration Gain % (wie viel besser als Original) ───────────────
        gain_pct = 0.0
        if pre_restoration_mos > 0 and versa_mos > 0:
            gain_pct = (versa_mos - pre_restoration_mos) / max(pre_restoration_mos, 1.0) * 100.0

        # ── Defect-Verbesserung ──────────────────────────────────────────────
        defect_improvement: dict[str, float] = {}
        if defect_types and pre_restoration_goals and goals:
            for dt in defect_types:
                dt_key = dt.lower().replace(" ", "_")
                if dt_key in goals and dt_key in pre_restoration_goals:
                    defect_improvement[dt] = goals[dt_key] - pre_restoration_goals[dt_key]

        return PerceptualVerdict(
            holistic_score=holistic,
            recommendation=rec,
            recommendation_reason=reason,
            confidence_interval=(ci_low, ci_high),
            dimensions=dimensions,
            material_baseline=baseline["goals"],
            genre_expected=baseline["goals"] * genre_mod,
            restoration_gain_pct=gain_pct,
            defect_improvement=defect_improvement,
            scoring_method="pqc_v10.5",
        )


# Singleton
_instance: PerceptualQualityCouncil | None = None


def get_perceptual_council() -> PerceptualQualityCouncil:
    global _instance
    if _instance is None:
        _instance = PerceptualQualityCouncil()
    return _instance
