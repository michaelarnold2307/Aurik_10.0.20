"""
tests/unit/test_adaptive_phase_rescheduler.py — §2.78 AdaptivePhaseRescheduler

Prüft:
- Grundlegende Importierbarkeit und Singleton-Verhalten
- Injektion einer Recovery-Phase wenn Goal-Lücke > GAP_THRESHOLD
- Keine Injektion wenn Goal-Lücke ≤ GAP_THRESHOLD (§2.45 Minimal-Intervention)
- §0a-Guard: phase_21/35/42 nie in Restoration injiziert
- §MAS-Guard: Caller prüft _mas_fully_achieved (reset_session verhindert Doppel-Injektion)
- Keine Doppel-Injektion derselben Phase in einer Session
- Session-Limit: max 3 Injektionen pro Session
- §_NEVER_SKIP: diese Phasen werden nicht injiziert (unnötig — laufen immer)
- Phase bereits im Plan: wird nicht erneut injiziert
- Phase bereits ausgeführt: wird nicht erneut injiziert
- Non-blocking: kaputte Inputs liefern leeres RescheduleResult
"""

from unittest.mock import patch

import pytest


def _get_rescheduler():
    from backend.core.adaptive_phase_rescheduler import (
        AdaptivePhaseRescheduler,
        get_adaptive_phase_rescheduler,
    )

    return AdaptivePhaseRescheduler, get_adaptive_phase_rescheduler


def _fresh_rescheduler():
    """Frische (nicht-Singleton) Instanz für isolierte Tests."""
    Cls, _ = _get_rescheduler()
    r = Cls()
    return r


@pytest.mark.unit
class TestImportAndSingleton:
    def test_import(self):
        _get_rescheduler()

    def test_singleton_identity(self):
        _, factory = _get_rescheduler()
        r1 = factory()
        r2 = factory()
        assert r1 is r2

    def test_reschedule_result_dataclass(self):
        from backend.core.adaptive_phase_rescheduler import RescheduleResult

        result = RescheduleResult()
        assert result.new_phases_to_append == []
        assert result.goal_gaps_found == {}


class TestGapThreshold:
    """§2.45 Minimal-Intervention: nur bei signifikanter Lücke injizieren."""

    def test_no_injection_when_no_gap(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.85},
            song_goal_targets={"brillanz": 0.80},  # Score > Target → kein Gap
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
        )
        assert result.new_phases_to_append == []

    def test_no_injection_when_gap_below_threshold(self):
        """Gap = 0.04 < GAP_THRESHOLD(0.05) → keine Injektion."""
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.76},
            song_goal_targets={"brillanz": 0.80},  # gap = 0.04
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
        )
        assert result.new_phases_to_append == []

    def test_injection_when_gap_above_threshold(self):
        """Gap = 0.10 > GAP_THRESHOLD(0.05) → Injektion."""
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.70},
            song_goal_targets={"brillanz": 0.80},  # gap = 0.10
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
        )
        # phase_06_frequency_restoration ist Primär-Recovery für brillanz
        assert len(result.new_phases_to_append) >= 1
        assert "phase_06_frequency_restoration" in result.new_phases_to_append
        assert "brillanz" in result.goal_gaps_found


class TestRestorationForbiddenGuard:
    """§0a: phase_21/35/42 niemals in Restoration injiziert."""

    def test_no_forbidden_phase_injected_in_restoration(self):
        r = _fresh_rescheduler()
        from backend.core.adaptive_phase_rescheduler import _RESTORATION_FORBIDDEN

        # Teste alle Goals — keine verbotene Phase darf erscheinen
        _all_goals = [
            "natuerlichkeit",
            "authentizitaet",
            "brillanz",
            "transparenz",
            "emotionalitaet",
            "micro_dynamics",
            "waerme",
            "bass_kraft",
            "spatial_depth",
            "tonal_center",
            "timbre",
            "artikulation",
            "groove",
            "separation_fidelity",
            "transient_energie",
        ]
        _targets = dict.fromkeys(_all_goals, 0.9)
        _scores = dict.fromkeys(_all_goals, 0.7)
        result = r.reschedule(
            current_goal_scores=_scores,
            song_goal_targets=_targets,
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
        )
        for p in result.new_phases_to_append:
            assert p not in _RESTORATION_FORBIDDEN, f"§0a-Verletzung: {p} in Restoration injiziert"


class TestNoDuplicateInjection:
    """Phase die bereits im Plan oder ausgeführt ist, wird nicht nochmals injiziert."""

    def test_no_injection_if_phase_already_in_plan(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.70},
            song_goal_targets={"brillanz": 0.80},
            selected_phases=["phase_06_frequency_restoration"],  # bereits im Plan
            executed=set(),
            is_studio_2026=False,
        )
        assert "phase_06_frequency_restoration" not in result.new_phases_to_append

    def test_no_injection_if_phase_already_executed(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.70},
            song_goal_targets={"brillanz": 0.80},
            selected_phases=[],
            executed={"phase_06_frequency_restoration"},  # bereits ausgeführt
            is_studio_2026=False,
        )
        assert "phase_06_frequency_restoration" not in result.new_phases_to_append

    def test_no_duplicate_injection_across_calls(self):
        """Selbe Phase wird in einer Session nicht zweimal injiziert."""
        r = _fresh_rescheduler()
        _kwargs = {
            "current_goal_scores": {"brillanz": 0.70},
            "song_goal_targets": {"brillanz": 0.80},
            "selected_phases": [],
            "executed": set(),
            "is_studio_2026": False,
        }
        result1 = r.reschedule(**_kwargs)
        inj = result1.new_phases_to_append

        # Zweiter Aufruf: Phase ist jetzt in selected_phases (simuliert Append)
        result2 = r.reschedule(
            current_goal_scores={"brillanz": 0.70},
            song_goal_targets={"brillanz": 0.80},
            selected_phases=list(inj),
            executed=set(),
            is_studio_2026=False,
        )
        for p in result2.new_phases_to_append:
            assert p not in inj, f"Doppel-Injektion: {p}"


class TestSessionLimit:
    """Max MAX_INJECTIONS_PER_SESSION (3) Injektionen pro Session."""

    def test_session_limit_respected(self):
        from backend.core.adaptive_phase_rescheduler import MAX_INJECTIONS_PER_SESSION

        r = _fresh_rescheduler()
        _all_goals = [
            "brillanz",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "micro_dynamics",
        ]
        _targets = dict.fromkeys(_all_goals, 0.9)
        _scores = dict.fromkeys(_all_goals, 0.6)
        _injected_total: list[str] = []
        for _ in range(5):
            result = r.reschedule(
                current_goal_scores=_scores,
                song_goal_targets=_targets,
                selected_phases=list(_injected_total),
                executed=set(),
                is_studio_2026=False,
            )
            _injected_total.extend(result.new_phases_to_append)
        assert len(_injected_total) <= MAX_INJECTIONS_PER_SESSION

    def test_reset_session_clears_limit(self):
        from backend.core.adaptive_phase_rescheduler import MAX_INJECTIONS_PER_SESSION

        r = _fresh_rescheduler()
        _targets = {"brillanz": 0.90}
        _scores = {"brillanz": 0.60}
        # Limite verbrauchen
        for _ in range(MAX_INJECTIONS_PER_SESSION + 2):
            r.reschedule(
                current_goal_scores=_scores,
                song_goal_targets=_targets,
                selected_phases=[],
                executed=set(),
                is_studio_2026=False,
            )
        r.reset_session()
        # Nach Reset: wieder möglich
        result = r.reschedule(
            current_goal_scores=_scores,
            song_goal_targets=_targets,
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
        )
        assert len(result.new_phases_to_append) >= 1


class TestNeverSkipNotInjected:
    """§2.52 _NEVER_SKIP-Phasen werden nicht injiziert (unnötig — laufen immer)."""

    def test_never_skip_not_in_result(self):
        from backend.core.adaptive_phase_rescheduler import _NEVER_SKIP

        r = _fresh_rescheduler()
        _all_goals = [
            "natuerlichkeit",
            "authentizitaet",
            "brillanz",
            "transparenz",
            "emotionalitaet",
            "micro_dynamics",
            "waerme",
            "bass_kraft",
        ]
        result = r.reschedule(
            current_goal_scores=dict.fromkeys(_all_goals, 0.6),
            song_goal_targets=dict.fromkeys(_all_goals, 0.9),
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
        )
        for p in result.new_phases_to_append:
            assert p not in _NEVER_SKIP, f"_NEVER_SKIP-Phase injiziert: {p}"


class TestNonBlocking:
    """Kaputte Inputs müssen leeres RescheduleResult liefern (non-blocking)."""

    def test_empty_scores_no_crash(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={},
            song_goal_targets={"brillanz": 0.80},
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
        )
        assert isinstance(result.new_phases_to_append, list)

    def test_empty_targets_no_crash(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.70},
            song_goal_targets={},
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
        )
        assert result.new_phases_to_append == []

    def test_none_values_in_scores_no_crash(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": None, "natuerlichkeit": 0.70},  # type: ignore
            song_goal_targets={"brillanz": 0.80, "natuerlichkeit": 0.90},
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
        )
        assert isinstance(result.new_phases_to_append, list)


class TestAdaptiveGapThreshold:
    """_adaptive_gap_threshold() — restorability-adaptiver GAP_THRESHOLD."""

    def test_high_restorability_returns_max_threshold(self):
        """rest=100 → 0.05 (Standard)."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_gap_threshold

        assert _adaptive_gap_threshold(100.0) == pytest.approx(0.05, abs=1e-6)

    def test_zero_restorability_returns_min_threshold(self):
        """rest=0 → 0.020 (aggressivste Erkennung, v10.0.0: abgesenkt für sensitivere Recovery)."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_gap_threshold

        assert _adaptive_gap_threshold(0.0) == pytest.approx(0.020, abs=1e-6)

    def test_midpoint_restorability(self):
        """rest=50 → 0.035 (lineare Mitte, v10.0.0: neue Formel 0.020+0.030×rest/100)."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_gap_threshold

        assert _adaptive_gap_threshold(50.0) == pytest.approx(0.035, abs=1e-6)

    def test_clip_out_of_range(self):
        """Werte außerhalb [0, 100] werden geclippt."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_gap_threshold

        assert _adaptive_gap_threshold(-50.0) == pytest.approx(0.020, abs=1e-6)  # geclippt auf min 0.020 (v10.0.0)
        assert _adaptive_gap_threshold(200.0) == pytest.approx(0.05, abs=1e-6)

    def test_monotone_increasing(self):
        """Threshold steigt monoton mit restorability."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_gap_threshold

        vals = [_adaptive_gap_threshold(r) for r in range(0, 101, 10)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1], f"Nicht monoton bei rest={i * 10}"


class TestAdaptiveMaxInjections:
    """_adaptive_max_injections() — restorability-adaptives Injektionslimit."""

    def test_high_restorability_returns_3(self):
        """rest=100 → 3 (Standard)."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_max_injections

        assert _adaptive_max_injections(100.0) == 3

    def test_rest_51_returns_3(self):
        """rest > 50 → 3."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_max_injections

        assert _adaptive_max_injections(51.0) == 3

    def test_rest_50_returns_4(self):
        """rest ≤ 50 → 4."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_max_injections

        assert _adaptive_max_injections(50.0) == 4

    def test_rest_25_returns_5(self):
        """rest ≤ 25 → 5."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_max_injections

        assert _adaptive_max_injections(25.0) == 5

    def test_rest_zero_returns_5(self):
        """rest=0 → 5 (sehr schwierige Songs)."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_max_injections

        assert _adaptive_max_injections(0.0) == 5

    def test_clip_negative(self):
        """Negative Werte → 5 (wie rest=0)."""
        from backend.core.adaptive_phase_rescheduler import _adaptive_max_injections

        assert _adaptive_max_injections(-10.0) == 5


class TestRestorabilityAdaptiveBehavior:
    """Integrations-Tests: restorability_score steuert Injektion korrekt."""

    def test_low_restorability_injects_more(self):
        """rest=10 erlaubt bis zu 5 Injektionen; rest=100 nur 3."""
        r_low = _fresh_rescheduler()
        r_high = _fresh_rescheduler()
        _all_goals = [
            "brillanz",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
            "micro_dynamics",
            "groove",
        ]
        _targets = dict.fromkeys(_all_goals, 0.9)
        _scores = dict.fromkeys(_all_goals, 0.5)

        _inj_low: list[str] = []
        _inj_high: list[str] = []
        for _ in range(6):
            res_low = r_low.reschedule(
                current_goal_scores=_scores,
                song_goal_targets=_targets,
                selected_phases=list(_inj_low),
                executed=set(),
                is_studio_2026=False,
                restorability_score=10.0,
            )
            _inj_low.extend(res_low.new_phases_to_append)

            res_high = r_high.reschedule(
                current_goal_scores=_scores,
                song_goal_targets=_targets,
                selected_phases=list(_inj_high),
                executed=set(),
                is_studio_2026=False,
                restorability_score=100.0,
            )
            _inj_high.extend(res_high.new_phases_to_append)

        # Beide respektieren ihr Limit; low darf mehr injizieren
        assert len(_inj_low) <= 5
        assert len(_inj_high) <= 3
        assert len(_inj_low) >= len(_inj_high)  # low ≥ high (mindestens gleich viele)

    def test_low_restorability_lower_gap_threshold(self):
        """rest=0 → gap_threshold=0.025; gap=0.03 reicht für Injektion."""
        r = _fresh_rescheduler()
        # gap = 0.80 - 0.77 = 0.03 (über 0.025, aber unter 0.05)
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.77},
            song_goal_targets={"brillanz": 0.80},
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
            restorability_score=0.0,
        )
        assert len(result.new_phases_to_append) >= 1

    def test_high_restorability_gap_0_03_no_injection(self):
        """rest=100 → gap_threshold=0.05; gap=0.03 reicht NICHT für Injektion."""
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.77},
            song_goal_targets={"brillanz": 0.80},
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
            restorability_score=100.0,
        )
        assert result.new_phases_to_append == []


class TestGoalConfidenceAdaptiveBehavior:
    """Niedrige Goal-Konfidenz soll Injektionen konservativer machen."""

    def test_low_confidence_blocks_borderline_gap(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.72},
            song_goal_targets={"brillanz": 0.78},  # gap = 0.06
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
            restorability_score=100.0,  # Basis-Schwelle 0.05
            goal_confidence={"brillanz": 0.20},
        )
        assert result.new_phases_to_append == []

    def test_high_confidence_allows_same_gap(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.72},
            song_goal_targets={"brillanz": 0.78},  # gap = 0.06
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
            restorability_score=100.0,
            goal_confidence={"brillanz": 0.95},
        )
        assert "phase_06_frequency_restoration" in result.new_phases_to_append

    def test_global_low_confidence_reduces_injection_budget(self):
        r_low = _fresh_rescheduler()
        r_high = _fresh_rescheduler()
        goals = [
            "brillanz",
            "natuerlichkeit",
            "authentizitaet",
            "transparenz",
            "micro_dynamics",
            "groove",
        ]
        targets = dict.fromkeys(goals, 0.9)
        scores = dict.fromkeys(goals, 0.5)

        low_conf_injected: list[str] = []
        high_conf_injected: list[str] = []

        for _ in range(6):
            low_res = r_low.reschedule(
                current_goal_scores=scores,
                song_goal_targets=targets,
                selected_phases=list(low_conf_injected),
                executed=set(),
                is_studio_2026=False,
                restorability_score=20.0,
                goal_confidence=dict.fromkeys(goals, 0.2),
            )
            low_conf_injected.extend(low_res.new_phases_to_append)

            high_res = r_high.reschedule(
                current_goal_scores=scores,
                song_goal_targets=targets,
                selected_phases=list(high_conf_injected),
                executed=set(),
                is_studio_2026=False,
                restorability_score=20.0,
                goal_confidence=dict.fromkeys(goals, 0.95),
            )
            high_conf_injected.extend(high_res.new_phases_to_append)

        assert len(low_conf_injected) <= len(high_conf_injected)

    def test_higher_confidence_goal_wins_when_gap_is_equal(self):
        """Bei gleichem Gap soll das verlässlichere Recovery-Goal zuerst kommen."""
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.72, "spatial_depth": 0.72},
            song_goal_targets={"brillanz": 0.82, "spatial_depth": 0.82},
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
            restorability_score=80.0,
            transfer_chain=[],
            material_type="vinyl",
            goal_confidence={"brillanz": 0.20, "spatial_depth": 0.95},
        )

        assert result.new_phases_to_append
        assert result.new_phases_to_append[0] == "phase_46_spatial_enhancement"


class TestCoalitionPriority:
    """Goals mit gemeinsamer Recovery-Phase sollen früher kommen."""

    def test_shared_recovery_phase_boosts_predictive_priority(self):
        r = _fresh_rescheduler()

        def _fake_recovery(goal: str, is_studio_2026: bool = False, transfer_chain=None):
            mapping = {
                "brillanz": ["phase_brillanz"],
                "spatial_depth": ["phase_spatial"],
                "waerme": ["phase_warmup", "phase_spatial"],
            }
            return list(mapping.get(goal, []))

        with patch("backend.core.adaptive_phase_rescheduler._get_goal_recovery_phases", side_effect=_fake_recovery):
            result = r.reschedule(
                current_goal_scores={"brillanz": 0.70, "spatial_depth": 0.70, "waerme": 0.70},
                song_goal_targets={"brillanz": 0.80, "spatial_depth": 0.80, "waerme": 0.80},
                selected_phases=["phase_warmup"],
                executed=set(),
                is_studio_2026=False,
                restorability_score=80.0,
                goal_confidence={"brillanz": 0.80, "spatial_depth": 0.80, "waerme": 0.80},
            )

        assert result.new_phases_to_append[:2] == ["phase_spatial", "phase_brillanz"]


class TestUncertaintyBudgetAndRiskPricing:
    """Geschlossener Regelkreis: Unsicherheit und Risiko steuern Injektionen."""

    def test_high_uncertainty_budget_blocks_borderline_gap(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.72},
            song_goal_targets={"brillanz": 0.77},  # gap = 0.05 (borderline)
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
            restorability_score=80.0,
            goal_confidence={"brillanz": 0.95},
            uncertainty_budget=0.92,
        )
        assert result.new_phases_to_append == []

    def test_low_uncertainty_budget_allows_same_gap(self):
        r = _fresh_rescheduler()
        result = r.reschedule(
            current_goal_scores={"brillanz": 0.72},
            song_goal_targets={"brillanz": 0.77},
            selected_phases=[],
            executed=set(),
            is_studio_2026=False,
            restorability_score=80.0,
            goal_confidence={"brillanz": 0.95},
            uncertainty_budget=0.10,
        )
        assert "phase_06_frequency_restoration" in result.new_phases_to_append

    def test_counterfactual_safe_phase_survives_risk_filter(self):
        r = _fresh_rescheduler()

        def _fake_recovery(goal: str, is_studio_2026: bool = False, transfer_chain=None):
            if goal == "natuerlichkeit":
                return ["phase_55_diffusion_inpainting"]
            return []

        with patch("backend.core.adaptive_phase_rescheduler._get_goal_recovery_phases", side_effect=_fake_recovery):
            result = r.reschedule(
                current_goal_scores={"natuerlichkeit": 0.60},
                song_goal_targets={"natuerlichkeit": 0.76},
                selected_phases=[],
                executed=set(),
                is_studio_2026=False,
                transfer_chain=["vinyl", "mp3_low"],
                material_type="vinyl",
                restorability_score=60.0,
                goal_confidence={"natuerlichkeit": 0.75},
                uncertainty_budget=0.55,
            )

        assert "phase_55_diffusion_inpainting" in result.new_phases_to_append

    def test_rare_chain_can_unlock_one_extra_injection_slot(self):
        r = _fresh_rescheduler()

        with patch("backend.core.adaptive_phase_rescheduler._adaptive_max_injections", return_value=3):
            goals = ["brillanz", "natuerlichkeit", "authentizitaet", "transparenz", "micro_dynamics", "groove"]
            targets = dict.fromkeys(goals, 0.90)
            scores = dict.fromkeys(goals, 0.50)
            injected: list[str] = []
            for _ in range(4):
                res = r.reschedule(
                    current_goal_scores=scores,
                    song_goal_targets=targets,
                    selected_phases=list(injected),
                    executed=set(),
                    is_studio_2026=False,
                    transfer_chain=["wire_recording", "cassette", "vinyl", "mp3_low"],
                    material_type="wire_recording",
                    restorability_score=30.0,
                    goal_confidence=dict.fromkeys(goals, 0.85),
                    uncertainty_budget=0.35,
                )
                injected.extend(res.new_phases_to_append)

        assert len(injected) <= 4
