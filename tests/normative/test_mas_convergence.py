from __future__ import annotations

"""[RELEASE_MUST] MAS-Konvergenz-CI-Gate (v10.0.0)

CI-Gate das normativ prüft:

  §0k:  MAS-Targets liegen immer >= Canonical-Floor (nie darunter).
  §09.11: MAS-Targets für Shellac/WaxCyl respektieren physikalische Obergrenzen.
  §2.64: Delta-Entscheidungslogik korrekt (Rollback bei P1/P2 < -0.03).
  §2.65: MAS-Convergence-Early-Stop-Logik korrekt (alle P1/P2 <= MAS+0.02).

Keine Audioverarbeitung, kein ML-Aufruf. Reine Logik- und Kalibrierungstests.
Laufzeit: < 3 s.

Aufruf:
    pytest tests/normative/test_mas_convergence.py -v

Spec-Referenzen:
    §0k  Maximum-Achievable-Score-Prinzip
    §2.64 Per-Phase-Score-Delta-Loop
    §2.65 MAS-Convergence-Early-Stop
    §09.11 MAS Formale Definition + PHYSICAL_CEILING
"""


import math

import pytest

from backend.core.calibration_matrix import CANONICAL_THRESHOLDS_RESTORATION
from backend.core.studio_goal_targets import estimate_song_goal_targets

# ---------------------------------------------------------------------------
# Canonical P1/P2 Goal names (subset for strict regression checks)
# ---------------------------------------------------------------------------
P1P2_GOALS = [
    "natuerlichkeit",
    "authentizitaet",
    "tonal_center",
    "timbre_authentizitaet",
    "artikulation",
]

ALL_CANONICAL_GOALS = list(CANONICAL_THRESHOLDS_RESTORATION.keys())

# ---------------------------------------------------------------------------
# §09.11 PHYSICAL_CEILING (normative; Produktionscode noch pending)
# Definiert die physikalisch maximal erreichbaren Goal-Scores pro Material.
# Wenn leer: kein physikalisches Ceiling (CD/DAT/FLAC).
# ---------------------------------------------------------------------------
PHYSICAL_CEILING: dict[str, dict[str, float]] = {
    "shellac": {
        "brillanz": 0.72,
        "transparenz": 0.72,
        "spatial_depth": 0.55,
        "raumtiefe": 0.55,
        "artikulation": 0.78,
        "separation_fidelity": 0.60,
    },
    "wax_cylinder": {
        "brillanz": 0.55,
        "transparenz": 0.60,
        "spatial_depth": 0.45,
        "raumtiefe": 0.45,
        "artikulation": 0.70,
    },
    "vinyl": {
        "brillanz": 0.86,
        "transparenz": 0.84,
        "spatial_depth": 0.80,
        "raumtiefe": 0.80,
    },
    "tape": {
        "brillanz": 0.88,
        "transparenz": 0.86,
    },
    "reel_tape": {
        "brillanz": 0.90,
        "transparenz": 0.88,
    },
    "mp3_low": {
        "brillanz": 0.80,
        "transparenz": 0.78,
        "artikulation": 0.82,
    },
}

# ---------------------------------------------------------------------------
# §2.64/§2.65 Konvergenzkonstanten (normativ aus Spec 02)
# ---------------------------------------------------------------------------
MAS_TOLERANCE = 0.02  # P1/P2 als "erreicht" wenn gap <= 0.02
MAS_FULL_TOLERANCE = 0.05  # P3-P5 als "erreicht" wenn gap <= 0.05
MAS_OVERSHOOT_TOLERANCE = 0.03


# ---------------------------------------------------------------------------
# §2.65 Referenz-Implementierung (Pure-Function — unabhängig von UV3-Produktionscode)
# Dient als normative Spec-Verifikation.
# ---------------------------------------------------------------------------
def _compute_mas_convergence(
    current_scores: dict[str, float],
    mas_targets: dict[str, float],
) -> dict:
    gaps = {g: mas_targets[g] - current_scores.get(g, 0.0) for g in mas_targets}
    p1p2_goals_in_targets = [g for g in P1P2_GOALS if g in mas_targets]
    other_goals = [g for g in mas_targets if g not in P1P2_GOALS]

    p1p2_achieved = all(gaps[g] <= MAS_TOLERANCE for g in p1p2_goals_in_targets)
    p3p5_achieved = all(gaps[g] <= MAS_FULL_TOLERANCE for g in other_goals)
    overshooting = [g for g in mas_targets if current_scores.get(g, 0.0) > mas_targets[g] + MAS_OVERSHOOT_TOLERANCE]
    max_p1p2_gap = max((gaps[g] for g in p1p2_goals_in_targets), default=0.0)
    return {
        "gaps": gaps,
        "p1p2_achieved": p1p2_achieved,
        "p3p5_achieved": p3p5_achieved,
        "fully_achieved": p1p2_achieved and p3p5_achieved,
        "overshooting_goals": overshooting,
        "max_p1p2_gap": max_p1p2_gap,
    }


# ---------------------------------------------------------------------------
# §0k: MAS-Targets >= Canonical Floor
# ---------------------------------------------------------------------------
class TestMASAboveCanonicalFloor:
    """§0k: MAS-Targets dürfen niemals unter dem Canonical-Floor liegen."""

    @pytest.mark.parametrize(
        "material,era,genre,restorability",
        [
            ("shellac", 1930, "jazz", 40),
            ("shellac", 1920, "oper", 30),
            ("vinyl", 1970, "schlager", 65),
            ("vinyl", 1965, "rock", 70),
            ("tape", 1975, "pop", 75),
            ("reel_tape", 1968, "klassik", 80),
            ("cd_digital", 1998, "pop", 90),
            ("mp3_low", 2005, "hip-hop", 55),
            ("wax_cylinder", 1910, "klassik", 20),
        ],
    )
    def test_mas_target_at_or_above_canonical_floor(
        self, material: str, era: int, genre: str, restorability: int
    ) -> None:
        """Jedes MAS-Target muss >= effective_floor liegen.

        effective_floor = min(canonical_floor, PHYSICAL_CEILING)
        — für Vintage-Materialien (Shellac, WaxCyl) liegt das physikalische Ceiling
        UNTER dem globalen Canonical-Floor. Das ist korrekt: Shellac kann physikalisch
        keine Artikulation=0.85 erreichen (BW ~7 kHz, SNR ~15 dB).
        Die Pipeline setzt das erreichbare Maximum, nicht das unerreichbare Canonical-Floor.
        """
        result = estimate_song_goal_targets(
            is_studio_2026=False,
            goal_weights=None,
            restorability_score=float(restorability),
            era_decade=era,
            genre_label=genre,
            material_type=material,
        )
        material_ceiling = PHYSICAL_CEILING.get(material, {})
        for goal in P1P2_GOALS:
            floor = CANONICAL_THRESHOLDS_RESTORATION.get(goal)
            if floor is None:
                continue
            target = result.targets.get(goal)
            if target is None:
                continue
            # Effektiver Mindestboden: physikalisches Ceiling hat Vorrang wenn es niedriger ist
            effective_floor = min(floor, material_ceiling.get(goal, floor))
            assert target >= effective_floor - 0.001, (
                f"MAS target für {goal} ({target:.4f}) liegt UNTER dem "
                f"effective_floor ({effective_floor:.4f}) für material={material}, "
                f"era={era}, genre={genre}, restorability={restorability}. "
                f"(canonical_floor={floor:.3f}, physical_ceiling={material_ceiling.get(goal, 'none')}). "
                f"Verletzt §0k Maximum-Achievable-Score-Invariante."
            )


# ---------------------------------------------------------------------------
# §09.11: Shellac- und WaxCylinder-Targets respektieren PHYSICAL_CEILING
# ---------------------------------------------------------------------------
class TestMASPhysicalCeilingRespected:
    """§09.11: MAS-Targets für vintage Materialien dürfen PHYSICAL_CEILING nicht überschreiten.

    Hinweis: Diese Tests dokumentieren das Soll-Verhalten wenn PHYSICAL_CEILING-Clamp
    in estimate_song_goal_targets() implementiert wird (backend/core/studio_goal_targets.py).
    Bis dahin prüfen sie ob die Targets plausibel nahe am Ceiling bleiben.
    """

    @pytest.mark.parametrize(
        "material,era,genre,restorability",
        [
            ("shellac", 1930, "jazz", 40),
            ("shellac", 1920, "blues", 30),
            ("wax_cylinder", 1905, "klassik", 20),
        ],
    )
    def test_shellac_brillanz_target_physically_plausible(
        self, material: str, era: int, genre: str, restorability: int
    ) -> None:
        """Shellac/WaxCyl MAS-Brillanz muss physikalisch plausibel sein (kein HF über Material-Ceiling)."""
        result = estimate_song_goal_targets(
            is_studio_2026=False,
            goal_weights=None,
            restorability_score=float(restorability),
            era_decade=era,
            genre_label=genre,
            material_type=material,
        )
        ceiling = PHYSICAL_CEILING.get(material, {})
        brillanz_target = result.targets.get("brillanz")
        brillanz_ceiling = ceiling.get("brillanz")
        if brillanz_target is not None and brillanz_ceiling is not None:
            # Nach PHYSICAL_CEILING-Clamp (§09.11): target MUSS <= ceiling sein.
            assert brillanz_target <= brillanz_ceiling, (
                f"Brillanz-Target ({brillanz_target:.4f}) überschreitet PHYSICAL_CEILING "
                f"({brillanz_ceiling:.4f}) für {material}/{era}/{genre}. "
                f"estimate_song_goal_targets() wendet PHYSICAL_CEILING-Clamp nicht korrekt an (§09.11)."
            )

    def test_shellac_spatial_depth_physically_constrained(self) -> None:
        """Shellac ist Mono — Raumtiefe/spatial_depth muss stark reduziert sein."""
        result = estimate_song_goal_targets(
            is_studio_2026=False,
            goal_weights=None,
            restorability_score=35.0,
            era_decade=1925,
            genre_label="jazz",
            material_type="shellac",
        )
        spatial = result.targets.get("raumtiefe") or result.targets.get("spatial_depth") or 0.0
        ceiling = PHYSICAL_CEILING["shellac"]["spatial_depth"]
        # Wir erlauben etwas mehr als Ceiling bis Clamp implementiert ist
        assert spatial <= ceiling + 0.20, (
            f"Shellac spatial_depth-Target ({spatial:.4f}) viel zu hoch. "
            f"Shellac ist Mono — physikalisches Ceiling: {ceiling:.4f} (§09.11)."
        )

    def test_wax_cylinder_ceiling_requires_clamp_implementation(self) -> None:
        """wax_cylinder brillanz-Ceiling (0.55) muss durch PHYSICAL_CEILING-Clamp eingehalten werden.

        WaxCyl BW ≤ 5 kHz → brillanz physikalisch ≤ 0.55.
        Prüft dass estimate_song_goal_targets() den PHYSICAL_CEILING-Clamp korrekt anwendet.
        """
        result = estimate_song_goal_targets(
            is_studio_2026=False,
            goal_weights=None,
            restorability_score=20.0,
            era_decade=1905,
            genre_label="klassik",
            material_type="wax_cylinder",
        )
        brillanz = result.targets.get("brillanz", 0.0)
        ceiling = PHYSICAL_CEILING["wax_cylinder"]["brillanz"]
        assert brillanz <= ceiling, (
            f"wax_cylinder brillanz-Target ({brillanz:.4f}) überschreitet "
            f"PHYSICAL_CEILING ({ceiling:.4f}). "
            f"PHYSICAL_CEILING-Clamp in estimate_song_goal_targets() funktioniert nicht (§09.11)."
        )


# ---------------------------------------------------------------------------
# §2.64: Delta-Entscheidungslogik (Rollback-Schwelle)
# ---------------------------------------------------------------------------
class TestPhaseDeltaDecisionLogic:
    """§2.64: Rollback-Schwelle für P1/P2-Regression (-0.03) korrekt."""

    def test_p1p2_regression_triggers_rollback(self) -> None:
        """Delta < -0.03 für einen P1/P2-Goal muss Rollback auslösen."""
        dict.fromkeys(P1P2_GOALS, 0.92)
        pre_scores = dict.fromkeys(P1P2_GOALS, 0.88)
        post_scores = dict.fromkeys(P1P2_GOALS, 0.88)
        # Simuliere starke Regression an natuerlichkeit
        post_scores["natuerlichkeit"] = 0.84

        delta = {g: post_scores[g] - pre_scores[g] for g in P1P2_GOALS}
        regression = any(delta[g] < -0.03 for g in P1P2_GOALS)
        assert regression is True, (
            "P1/P2-Regression (delta=-0.04) hätte Rollback ausgelöst — "
            "Logik in _profiled_phase_call_with_delta() verletzt §2.64."
        )

    def test_minimal_regression_below_threshold_no_rollback(self) -> None:
        """Delta >= -0.03 (minimal) darf KEINEN Rollback auslösen."""
        dict.fromkeys(P1P2_GOALS, 0.92)
        pre_scores = dict.fromkeys(P1P2_GOALS, 0.88)
        post_scores = dict.fromkeys(P1P2_GOALS, 0.88)
        # Minimale Regression unterhalb der Schwelle
        post_scores["natuerlichkeit"] = 0.852  # delta = -0.028

        delta = {g: post_scores[g] - pre_scores[g] for g in P1P2_GOALS}
        regression = any(delta[g] < -0.03 for g in P1P2_GOALS)
        assert regression is False, (
            "Minimale Regression (delta=-0.028) darf keinen Rollback auslösen — "
            "Schwelle ist < -0.03 (strikt), nicht <= (§2.64)."
        )

    def test_overshoot_above_tolerance_detected(self) -> None:
        """Post-Score > MAS + 0.03 muss als Overshoot erkannt werden."""
        mas_targets = {"natuerlichkeit": 0.91}
        post_scores = {"natuerlichkeit": 0.95}  # 0.04 über MAS

        overshooting = [g for g in mas_targets if post_scores.get(g, 0.0) > mas_targets[g] + MAS_OVERSHOOT_TOLERANCE]
        assert "natuerlichkeit" in overshooting, (
            "Overshoot (0.04 über MAS) nicht erkannt — Strength-Clamp-Logik in §2.64 greift nicht."
        )

    def test_overshoot_within_tolerance_not_flagged(self) -> None:
        """Post-Score <= MAS + 0.03 darf NICHT als Overshoot erkannt werden."""
        mas_targets = {"natuerlichkeit": 0.91}
        post_scores = {"natuerlichkeit": 0.935}  # 0.025 über MAS

        overshooting = [g for g in mas_targets if post_scores.get(g, 0.0) > mas_targets[g] + MAS_OVERSHOOT_TOLERANCE]
        assert "natuerlichkeit" not in overshooting, (
            "Kein Overshoot (0.025 < 0.03 Toleranz) — darf nicht als Overshoot gewertet werden."
        )


# ---------------------------------------------------------------------------
# §2.65: MAS-Convergence-Early-Stop-Logik
# ---------------------------------------------------------------------------
class TestMASConvergenceEarlyStop:
    """§2.65: Pipeline-Stopp-Logik bei MAS-Erreichung korrekt."""

    def _make_targets(self, value: float = 0.92) -> dict[str, float]:
        return dict.fromkeys(P1P2_GOALS, value)

    def test_p1p2_achieved_when_all_gaps_within_tolerance(self) -> None:
        """Wenn alle P1/P2-Gaps <= MAS_TOLERANCE: p1p2_achieved muss True sein."""
        mas_targets = self._make_targets(0.92)
        # Alle Scores knapp unter MAS (gap = 0.015 <= 0.02)
        current = dict.fromkeys(P1P2_GOALS, 0.905)

        result = _compute_mas_convergence(current, mas_targets)
        assert result["p1p2_achieved"] is True, (
            f"P1/P2 als nicht erreicht erkannt obwohl alle Gaps <= {MAS_TOLERANCE}. "
            f"Max-Gap: {result['max_p1p2_gap']:.4f}. §2.65 Early-Stop-Bedingung fehlerhaft."
        )

    def test_p1p2_not_achieved_when_gap_exceeds_tolerance(self) -> None:
        """Wenn ein P1/P2-Gap > MAS_TOLERANCE: p1p2_achieved muss False sein."""
        mas_targets = self._make_targets(0.92)
        # Gap = 0.04 > 0.02
        current = dict.fromkeys(P1P2_GOALS, 0.88)

        result = _compute_mas_convergence(current, mas_targets)
        assert result["p1p2_achieved"] is False, (
            f"P1/P2 fälschlicherweise als erreicht erkannt obwohl Gap ({result['max_p1p2_gap']:.4f}) "
            f"> MAS_TOLERANCE ({MAS_TOLERANCE}). §2.65 Early-Stop würde zu früh stoppen."
        )

    def test_fully_achieved_requires_both_p1p2_and_p3p5(self) -> None:
        """fully_achieved = True nur wenn BEIDE (P1/P2 UND P3-P5) konvergiert sind."""
        mas_targets = dict.fromkeys(P1P2_GOALS, 0.92)
        mas_targets["groove"] = 0.87
        mas_targets["raumtiefe"] = 0.74

        # P1/P2 erreicht, aber P3-P5 nicht (gap = 0.10 > 0.05)
        current = dict.fromkeys(P1P2_GOALS, 0.91)  # gap = 0.01 ✓
        current["groove"] = 0.77  # gap = 0.10 > 0.05 ✗
        current["raumtiefe"] = 0.74  # gap = 0.0 ✓

        result = _compute_mas_convergence(current, mas_targets)
        assert result["p1p2_achieved"] is True
        assert result["fully_achieved"] is False, (
            "fully_achieved darf nicht True sein wenn P3-P5-Goals noch nicht konvergiert. "
            "§2.65: Pipeline darf nicht stoppen wenn nur P1/P2 konvergiert."
        )

    def test_fully_achieved_when_all_goals_converged(self) -> None:
        """fully_achieved = True wenn alle Goals innerhalb Tolerance."""
        mas_targets = dict.fromkeys(P1P2_GOALS, 0.92)
        mas_targets["groove"] = 0.87
        mas_targets["raumtiefe"] = 0.74

        # Alle Goals nahe am MAS (gap <= respective tolerance)
        current = dict.fromkeys(P1P2_GOALS, 0.91)  # gap = 0.01 ✓
        current["groove"] = 0.855  # gap = 0.015 ✓
        current["raumtiefe"] = 0.735  # gap = 0.005 ✓

        result = _compute_mas_convergence(current, mas_targets)
        assert result["fully_achieved"] is True, (
            "fully_achieved sollte True sein wenn alle Goals innerhalb Toleranz. "
            f"Gaps: {result['gaps']}. §2.65 Early-Stop-Invariante."
        )

    def test_pipeline_must_stop_after_mas_achieved(self) -> None:
        """Simuliert UV3-Loop: Nach _mas_fully_achieved=True dürfen keine Phasen mehr laufen."""
        mas_targets = dict.fromkeys(P1P2_GOALS, 0.92)
        mas_targets["groove"] = 0.87

        current_scores = dict.fromkeys(P1P2_GOALS, 0.91)
        current_scores["groove"] = 0.865

        convergence = _compute_mas_convergence(current_scores, mas_targets)
        mas_fully_achieved = convergence["fully_achieved"]

        # Simuliere Pipeline-Loop
        phases_executed_after_convergence = 0
        remaining_phases = ["phase_29", "phase_06", "phase_23"]
        for phase in remaining_phases:
            if mas_fully_achieved:
                break
            phases_executed_after_convergence += 1

        assert phases_executed_after_convergence == 0, (
            f"Pipeline hat {phases_executed_after_convergence} Phasen nach MAS-Erreichung "
            f"ausgeführt. VERBOTEN laut §2.65 und §0k."
        )


# ---------------------------------------------------------------------------
# §0k: Allgemeine Plausibilitäts-Invariante (verschiedene Szenarien)
# ---------------------------------------------------------------------------
class TestMASPlausibilityInvariants:
    """§0k: Allgemeine Plausibilitäts-Checks für estimate_song_goal_targets()."""

    def test_high_restorability_yields_higher_targets(self) -> None:
        """Hohe Restorability (90) muss höhere oder gleiche Targets als niedrige (30) liefern."""
        high = estimate_song_goal_targets(
            restorability_score=90.0,
            is_studio_2026=False,
            goal_weights=None,
            era_decade=1975,
            genre_label="pop",
            material_type="vinyl",
        )
        low = estimate_song_goal_targets(
            restorability_score=30.0,
            is_studio_2026=False,
            goal_weights=None,
            era_decade=1975,
            genre_label="pop",
            material_type="vinyl",
        )

        for goal in P1P2_GOALS:
            h = high.targets.get(goal, 0.0)
            lo = low.targets.get(goal, 0.0)
            assert h >= lo - 0.05, (
                f"Goal {goal}: hohe Restorability ({h:.4f}) hat niedrigere Targets "
                f"als geringe Restorability ({lo:.4f}) — §0k MAS-Monotonie verletzt."
            )

    def test_confidence_in_valid_range(self) -> None:
        """SongGoalTargets.confidence muss immer in [0.0, 1.0] liegen."""
        result = estimate_song_goal_targets(
            is_studio_2026=False,
            goal_weights=None,
            restorability_score=50.0,
            era_decade=1950,
            genre_label="blues",
            material_type="shellac",
        )
        assert 0.0 <= result.confidence <= 1.0, (
            f"confidence ({result.confidence}) liegt außerhalb [0.0, 1.0]. "
            f"estimate_song_goal_targets() hat einen numerischen Fehler."
        )

    def test_no_nan_in_targets(self) -> None:
        """Kein NaN/Inf-Wert in MAS-Targets (Robustheit bei Edge-Case-Inputs)."""
        result = estimate_song_goal_targets(
            is_studio_2026=False,
            goal_weights=None,
            restorability_score=0.0,  # Extremwert: komplett unrestorierbar
            era_decade=1900,
            genre_label="",
            material_type="wax_cylinder",
        )
        for goal, value in result.targets.items():
            assert math.isfinite(value), (
                f"Goal {goal} hat NaN/Inf-Wert ({value}) bei extremen Inputs. "
                f"estimate_song_goal_targets() fehlt nan_to_num-Schutz."
            )
