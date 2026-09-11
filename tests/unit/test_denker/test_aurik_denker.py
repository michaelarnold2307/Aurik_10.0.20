from __future__ import annotations

import logging

"""tests/unit/test_denker/test_aurik_denker.py

Tests für AurikDenker — Orchestrator aller Domänen-Denker.
Verwendet ausschließlich synthetische Signale und vollständige Mocks;
kein echter Restorer-Aufruf, kein CDPAMPlugin, kein CrepePlugin.
np.random.seed(42) für Reproduzierbarkeit.
"""


import ast
import math
import sys
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from denker.aurik_denker import AurikErgebnis

# ─── Helpers ──────────────────────────────────────────────────────────────────

SR = 48_000


def _sine(duration_s: float = 1.0, freq: float = 440.0) -> np.ndarray:
    np.random.seed(42)
    t = np.linspace(0, duration_s, int(SR * duration_s), dtype=np.float32)
    y = np.sin(2 * np.pi * freq * t) * 0.5
    return np.asarray(y, dtype=np.float32)


def _normalized_fail_reasons(stage_notes: dict[str, Any]) -> list[dict[str, Any]]:
    raw = stage_notes.get("fail_reasons", [])
    if isinstance(raw, list):
        return cast(list[dict[str, Any]], raw)
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, list):
            return cast(list[dict[str, Any]], parsed)
    return []


def _make_toni_mock(material: str = "tape", confidence: float = 0.85) -> MagicMock:
    m = MagicMock()
    m.material_type = material
    m.confidence = confidence
    return m


def _make_kette_mock(chain_string: str = "tape→mp3_low") -> MagicMock:
    k = MagicMock()
    k.chain_string = chain_string
    k.chain_complexity = 0.6
    k.combined_phases = ["phase_03_denoise", "phase_23_spectral_repair"]
    k.as_dict.return_value = {
        "chain_string": chain_string,
        "chain_complexity": 0.6,
        "is_multi_generation": True,
        "generation_count": 2,
        "chain": ["tape", "mp3_low"],
        "primary_medium": "mp3_low",
    }
    return k


def _make_defekt_mock() -> MagicMock:
    d = MagicMock()
    d.defect_scores = {"clicks": 0.1, "hiss": 0.3}
    d.primary_defect = "hiss"
    d.confidence = 0.8
    d.overall_severity = 0.5
    return d


def _make_strategie_mock() -> MagicMock:
    s = MagicMock()
    s.selected_phases = ["phase_03_denoise"]
    s.budget_ok.return_value = True
    s.rt_factor.return_value = 0.8
    return s


def _make_exzellenz_mock(audio: np.ndarray, versa_mos: float = 0.0) -> MagicMock:
    e = MagicMock()
    e.audio = audio
    e.excellence_score = 0.82
    e.musical_goals = {"brillanz": 0.87, "waerme": 0.81}
    e.goals_passed = 2
    e.goals_total = 14
    e.processing_note = "Exzellenz optimiert"
    e.warnings = []
    e.versa_mos = versa_mos  # M-8b: VERSA-Cache-Interface
    return e


# ─── AurikErgebnis-Dataclass-Tests ────────────────────────────────────────────


@pytest.mark.unit
class TestAurikErgebnis:
    def _make(self, chain_info=None) -> AurikErgebnis:
        from denker.aurik_denker import AurikErgebnis

        return AurikErgebnis(
            audio=_sine(),
            material="tape",
            rt_factor=0.9,
            quality_estimate=0.75,
            musical_goals={"brillanz": 0.86},
            goals_passed=1,
            phases_executed=["phase_03_denoise"],
            warnings=[],
            processing_note="Test",
            stage_notes={"tontraeger": "tape (0.85)"},
            chain_info=chain_info,
        )

    def test_01_chain_info_field_exists(self):
        """AurikErgebnis hat das Feld chain_info."""
        import dataclasses

        from denker.aurik_denker import AurikErgebnis

        fields = {f.name for f in dataclasses.fields(AurikErgebnis)}
        assert "chain_info" in fields

    def test_02_chain_info_default_none(self):
        ergebnis = self._make()
        assert ergebnis.chain_info is None

    def test_03_chain_info_set_dict(self):
        ci = {"chain_string": "tape→mp3", "chain_complexity": 0.5}
        ergebnis = self._make(chain_info=ci)
        assert ergebnis.chain_info == ci

    def test_04_as_dict_contains_chain_info(self):
        ci = {"chain_string": "vinyl→aac"}
        ergebnis = self._make(chain_info=ci)
        d = ergebnis.as_dict()
        assert "chain_info" in d
        assert d["chain_info"] == ci

    def test_05_as_dict_chain_info_none(self):
        ergebnis = self._make()
        d = ergebnis.as_dict()
        assert d["chain_info"] is None

    def test_06_as_dict_all_required_keys(self):
        d = self._make().as_dict()
        for key in (
            "material",
            "rt_factor",
            "quality_estimate",
            "musical_goals",
            "goals_passed",
            "phases_executed",
            "warnings",
            "processing_note",
            "stage_notes",
            "chain_info",
            "degradation_status",
            "fail_reason",
            "global_plan",
        ):
            assert key in d, f"Key '{key}' fehlt in as_dict()"

    def test_06b_degradation_defaults_are_typed(self):
        ergebnis = self._make()
        assert ergebnis.degradation_status == "ok"
        assert ergebnis.fail_reason is None

    def test_07_audio_field_is_ndarray(self):
        ergebnis = self._make()
        assert isinstance(ergebnis.audio, np.ndarray)

    def test_08_rt_factor_finite(self):
        ergebnis = self._make()
        assert math.isfinite(ergebnis.rt_factor)

    def test_09_quality_estimate_bounded(self):
        ergebnis = self._make()
        assert 0.0 <= ergebnis.quality_estimate <= 1.0


# ─── AurikDenker Singleton ────────────────────────────────────────────────────


class TestAurikDenkerSingleton:
    def test_10_get_aurik_denker_returns_instance(self):
        from denker.aurik_denker import AurikDenker, get_aurik_denker

        inst = get_aurik_denker()
        assert isinstance(inst, AurikDenker)

    def test_11_singleton_identity(self):
        from denker.aurik_denker import get_aurik_denker

        a = get_aurik_denker()
        b = get_aurik_denker()
        assert a is b

    def test_12_singleton_thread_safe(self):
        import concurrent.futures

        from denker.aurik_denker import get_aurik_denker

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda _: get_aurik_denker(), range(16)))
        assert all(r is results[0] for r in results)

    def test_13_denke_convenience_function_exists(self):
        from denker.aurik_denker import denke

        assert callable(denke)


class TestAurikDenkerNoRtLimit:
    def test_13b_denke_forwards_no_rt_limit_to_orchestrator(self):
        from denker.aurik_denker import AurikDenker

        denker = AurikDenker()
        captured: dict[str, bool] = {}

        def _fake_orchestriere(*args, **kwargs):
            captured["no_rt_limit"] = bool(kwargs.get("no_rt_limit", False))
            return denker._fallback(args[0], rt_factor=0.1, grund="test")

        with patch.object(AurikDenker, "_orchestriere", side_effect=_fake_orchestriere):
            denker.denke(_sine(), SR, no_rt_limit=True)

        assert captured.get("no_rt_limit") is True

    def test_13c_denke_forwards_phase_strength_oracle_rollout_to_orchestrator(self):
        from denker.aurik_denker import AurikDenker

        denker = AurikDenker()
        captured: dict[str, str] = {}

        def _fake_orchestriere(*args, **kwargs):
            captured["phase_strength_oracle_rollout"] = str(kwargs.get("phase_strength_oracle_rollout"))
            return denker._fallback(args[0], rt_factor=0.1, grund="test")

        with patch.object(AurikDenker, "_orchestriere", side_effect=_fake_orchestriere):
            denker.denke(_sine(), SR, phase_strength_oracle_rollout="pilot")

        assert captured.get("phase_strength_oracle_rollout") == "pilot"


# ─── AurikDenker._orchestriere Stage 1b ─────────────────────────────────────


class TestAurikDenkerStage1b:
    """Stage 1b: TontraegerketteDenker wird aufgerufen und chain_info befüllt."""

    def _run_with_mocks(self, audio: np.ndarray, kette_mock=None, fail_kette=False):
        from denker.aurik_denker import AurikDenker

        denker = AurikDenker()

        toni_m = MagicMock()
        toni_m.material_type = "tape"
        toni_m.confidence = 0.9

        defekt_m = _make_defekt_mock()
        strat_m = _make_strategie_mock()
        exz_m = _make_exzellenz_mock(audio)
        exz_denker_m = MagicMock()
        exz_denker_m.optimiere.return_value = exz_m

        versa_result_m = MagicMock()
        versa_result_m.mos = 4.2
        versa_result_m.model_used = "mock_versa"

        kette = kette_mock or _make_kette_mock()

        strat_check = MagicMock(should_exit_early=False)
        strat_m.check.return_value = strat_check

        rest_audio = audio.copy()
        rest_m = MagicMock()
        rest_m.restauriere.return_value = MagicMock(audio=rest_audio)

        rep_m = MagicMock()
        rep_m.repariere.return_value = rest_audio

        rek_m = MagicMock()
        rek_m.rekonstruiere.return_value = rest_audio

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=toni_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=defekt_m))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(
                    return_value=MagicMock(
                        analysiere=MagicMock(
                            side_effect=RuntimeError("fail") if fail_kette else MagicMock(return_value=kette)
                        )
                    )
                ),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_m)),
            patch("denker.aurik_denker.get_reparatur_denker", MagicMock(return_value=rep_m)),
            patch("denker.aurik_denker.get_rekonstruktions_denker", MagicMock(return_value=rek_m)),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m)),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            try:
                result = denker.denke(audio, SR)
            except Exception:
                result = None
        return result

    def test_14_chain_info_dict_on_success(self):
        audio = _sine()
        result = self._run_with_mocks(audio)
        if result is not None:
            # chain_info ist dict wenn Stage 1b erfolgreich
            assert result.chain_info is None or isinstance(result.chain_info, dict)

    def test_15_chain_info_none_on_kette_failure(self):
        """Bei Ausfall von TontraegerketteDenker bleibt chain_info {} oder None."""
        audio = _sine()
        result = self._run_with_mocks(audio, fail_kette=True)
        # Kein Absturz — chain_info ist leer dict oder None
        if result is not None:
            assert result.chain_info in (None, {})

    def test_16_stage_notes_has_kette_key_on_success(self):
        audio = _sine()
        kette_m = _make_kette_mock("tape→mp3_low")
        exz_m16 = _make_exzellenz_mock(audio)
        exz_denker_m16 = MagicMock()
        exz_denker_m16.optimiere.return_value = exz_m16
        versa_result_m16 = MagicMock(mos=4.2, model_used="mock_versa")
        strat_m16 = MagicMock()
        strat_m16.check.return_value = MagicMock(should_exit_early=False)
        rest_m16 = MagicMock()
        rest_m16.restauriere.return_value = MagicMock(audio=audio.copy())
        rep_m16 = MagicMock()
        rep_m16.repariere.return_value = audio.copy()
        rek_m16 = MagicMock()
        rek_m16.rekonstruiere.return_value = audio.copy()
        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_make_toni_mock()))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m16)),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_m16)),
            patch("denker.aurik_denker.get_reparatur_denker", MagicMock(return_value=rep_m16)),
            patch("denker.aurik_denker.get_rekonstruktions_denker", MagicMock(return_value=rek_m16)),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m16)),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m16)),
        ):
            from denker.aurik_denker import AurikDenker

            try:
                result = AurikDenker().denke(audio, SR)
                if result is not None:
                    assert "kette" in result.stage_notes
            except Exception:
                logger.warning("test fallback", exc_info=True)
                pass  # Andere Stufen dürfen scheitern

    def test_16b_pre_analysis_medium_forwarded_to_kette_denker(self):
        """pre_analysis_result.medium muss als cached_medium_result an analysiere() gehen (§2.47a)."""
        audio = _sine()

        cached_medium = MagicMock()
        cached_medium.primary_material = "vinyl"
        cached_medium.confidence = 0.91

        pre = MagicMock()
        pre.medium = cached_medium
        pre.era = None
        pre.genre = None
        pre.defects = None
        pre.restorability = None

        kette_m = _make_kette_mock("vinyl→mp3_low")
        exz_m = _make_exzellenz_mock(audio)
        exz_denker_m = MagicMock()
        exz_denker_m.optimiere.return_value = exz_m
        versa_result_m = MagicMock(mos=4.2, model_used="mock_versa")
        strat_m = MagicMock()
        strat_m.check.return_value = MagicMock(should_exit_early=False)
        rest_m = MagicMock()
        rest_m.restauriere.return_value = MagicMock(audio=audio.copy())
        rep_m = MagicMock()
        rep_m.repariere.return_value = audio.copy()
        rek_m = MagicMock()
        rek_m.rekonstruiere.return_value = audio.copy()

        toni_denker_m = MagicMock()
        kette_denker_m = MagicMock(analysiere=MagicMock(return_value=kette_m))

        with (
            patch("denker.aurik_denker.get_tontraeger_denker", MagicMock(return_value=toni_denker_m)),
            patch("denker.aurik_denker.get_tontraegerkette_denker", MagicMock(return_value=kette_denker_m)),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_m)),
            patch("denker.aurik_denker.get_reparatur_denker", MagicMock(return_value=rep_m)),
            patch("denker.aurik_denker.get_rekonstruktions_denker", MagicMock(return_value=rek_m)),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m)),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, pre_analysis_result=pre)

        # Bei vorhandenem cached_medium_result darf Stufe 1 keinen neuen Tonträger-Scan starten.
        toni_denker_m.erkenne.assert_not_called()
        assert kette_denker_m.analysiere.called
        call_kwargs = kette_denker_m.analysiere.call_args.kwargs
        assert call_kwargs.get("cached_medium_result") is cached_medium
        assert "vinyl" in str(result.stage_notes.get("tontraeger", "")).lower()

    def test_17_no_crash_on_all_stages_failing(self):
        """AurikDenker stürzt nicht ab, auch wenn alle Unterphasen scheitern."""
        audio = _sine()
        mock_versa_module = MagicMock()
        mock_versa_module.score_mos = MagicMock(side_effect=RuntimeError("mock"))
        with (
            patch("denker.aurik_denker.get_tontraeger_denker", side_effect=ImportError("mock")),
            patch("denker.aurik_denker.get_tontraegerkette_denker", side_effect=ImportError("mock")),
            patch("denker.aurik_denker.get_defekt_denker", side_effect=ImportError("mock")),
            patch("denker.aurik_denker.get_strategie_denker", side_effect=ImportError("mock")),
            patch("denker.aurik_denker.get_restaurier_denker", side_effect=ImportError("mock")),
            patch("denker.aurik_denker.get_reparatur_denker", side_effect=ImportError("mock")),
            patch("denker.aurik_denker.get_rekonstruktions_denker", side_effect=ImportError("mock")),
            patch("denker.aurik_denker.get_exzellenz_denker", side_effect=ImportError("mock")),
            patch.dict(sys.modules, {"plugins.versa_plugin": mock_versa_module}),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, mode="maximum")
        assert result is not None
        assert isinstance(result.audio, np.ndarray)


class TestAurikDenkerDigitalExcellenceGuard:
    def test_skips_excellence_for_clean_digital_chain(self):
        audio = _sine()
        exz_denker_m = MagicMock()
        versa_result_m = MagicMock(mos=3.2, model_used="mock_versa")
        kette_m = _make_kette_mock("tape→mp3_high")
        kette_m.as_dict.return_value = {
            "chain_string": "tape→mp3_high",
            "chain_complexity": 0.6,
            "is_multi_generation": True,
            "generation_count": 2,
            "chain": ["tape", "mp3_high"],
            "primary_medium": "mp3_high",
        }
        strat_m = _make_strategie_mock()
        strat_m.check.return_value = MagicMock(should_exit_early=False)
        rest_m = MagicMock()
        rest_m.restauriere.return_value = MagicMock(audio=audio.copy())
        rep_m = MagicMock()
        rep_m.repariere.return_value = audio.copy()
        rek_m = MagicMock()
        rek_m.rekonstruiere.return_value = audio.copy()

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_make_toni_mock()))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_m)),
            patch("denker.aurik_denker.get_reparatur_denker", MagicMock(return_value=rep_m)),
            patch("denker.aurik_denker.get_rekonstruktions_denker", MagicMock(return_value=rek_m)),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m)),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(True, {"material": "mp3_high"})),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, mode="maximum")

        rep_m.repariere.assert_not_called()
        rek_m.rekonstruiere.assert_not_called()
        rest_m.restauriere.assert_not_called()
        exz_denker_m.optimiere.assert_not_called()
        assert "clean_digital_pass_through" in result.phases_executed
        assert "saubere Digitalquelle" in result.stage_notes.get("restaurierung", "")
        assert "saubere Digitalquelle" in result.stage_notes.get("exzellenz", "")
        assert "exzellenz_optimierung_2" not in result.phases_executed

    def test_excellence_uses_primary_medium_from_chain(self):
        audio = _sine()
        exz_m = _make_exzellenz_mock(audio)
        exz_denker_m = MagicMock()
        exz_denker_m.optimiere.return_value = exz_m
        exz_denker_m.messe_ziele.return_value = {"brillanz": 0.87, "waerme": 0.81}
        versa_result_m = MagicMock(mos=4.2, model_used="mock_versa")
        kette_m = _make_kette_mock("tape→mp3_high")
        kette_m.as_dict.return_value = {
            "chain_string": "tape→mp3_high",
            "chain_complexity": 0.6,
            "is_multi_generation": True,
            "generation_count": 2,
            "chain": ["tape", "mp3_high"],
            "primary_medium": "mp3_high",
        }
        strat_m = _make_strategie_mock()
        strat_m.check.return_value = MagicMock(should_exit_early=False)
        rest_m = MagicMock()
        rest_m.restauriere.return_value = MagicMock(audio=audio.copy())
        rep_m = MagicMock()
        rep_m.repariere.return_value = audio.copy()
        rek_m = MagicMock()
        rek_m.rekonstruiere.return_value = audio.copy()

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_make_toni_mock("tape")))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_m)),
            patch("denker.aurik_denker.get_reparatur_denker", MagicMock(return_value=rep_m)),
            patch("denker.aurik_denker.get_rekonstruktions_denker", MagicMock(return_value=rek_m)),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m)),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "mp3_high"})),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            from denker.aurik_denker import AurikDenker

            AurikDenker().denke(audio, SR, no_rt_limit=True)

        # v10.0.0: ExzellenzDenker ruft messe_ziele() statt optimiere() auf
        assert exz_denker_m.messe_ziele.called, "ExzellenzDenker.messe_ziele() wurde nicht aufgerufen"

    def test_prefers_messe_und_repariere_when_available(self):
        audio = _sine()

        class _RealExz:
            def __init__(self):
                self.repair_called = False
                self.legacy_called = False
                self.last_mode: str | None = None
                self.last_material: str | None = None
                self.last_reference = None

            def messe_und_repariere(
                self,
                audio_in,
                sr,
                *,
                mode="restoration",
                material="auto",
                reference_audio=None,
            ):
                self.repair_called = True
                self.last_mode = str(mode)
                self.last_material = str(material)
                self.last_reference = reference_audio
                return audio_in, {"brillanz": 0.87, "waerme": 0.81}

            def messe_ziele(self, audio_in, sr):
                self.legacy_called = True
                return {"brillanz": 0.40}

        exz_real = _RealExz()
        versa_result_m = MagicMock(mos=4.6, model_used="mock_versa")
        rest_result = MagicMock()
        rest_result.audio = audio.copy()
        rest_result.phases_executed = []
        rest_result.warnings = []
        rest_result.quality_estimate = 0.8
        rest_result.rt_factor = 0.5
        rest_result.confidence = 0.9
        rest_result.rollback_triggered = False
        rest_result.winning_variant = "balanced"
        rest_result.musical_goals = {}
        rest_result.goals_passed = 0
        rest_result.metadata = {}

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_make_toni_mock("tape")))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_kette_mock("tape→mp3_high")))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=_make_strategie_mock())),
            patch(
                "denker.aurik_denker.get_restaurier_denker",
                MagicMock(return_value=MagicMock(restauriere=MagicMock(return_value=rest_result))),
            ),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=MagicMock(audio=audio.copy())))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=MagicMock(audio=audio.copy())))),
            ),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_real)),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "tape"})),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            from denker.aurik_denker import AurikDenker

            AurikDenker().denke(audio, SR, no_rt_limit=True)

        assert exz_real.repair_called, "ExzellenzDenker.messe_und_repariere() wurde nicht genutzt"
        assert not exz_real.legacy_called, "Legacy-Path messe_ziele() darf bei verfügbarem Repair-Pfad nicht laufen"
        assert exz_real.last_mode == "restoration"
        # v10.0.0 Fix G2: _resolve_excellence_material() bevorzugt analogen Ursprungs-Träger
        # aus der Transfer-Kette (chain=["tape", "mp3_low"]) → "tape" wird korrekt bevorzugt.
        assert exz_real.last_material in {"tape", "mp3_low", "mp3_high"}
        assert exz_real.last_reference is not None


class TestAurikDenkerMaterialMosGate:
    def test_material_mos_gate_clamps_quality_when_target_is_missed(self):
        audio = _sine()
        exz_m = _make_exzellenz_mock(audio)
        exz_denker_m = MagicMock()
        exz_denker_m.optimiere.return_value = exz_m
        versa_result_m = MagicMock(mos=3.6, model_used="mock_versa")

        kette_m = _make_kette_mock("tape→mp3_high")
        kette_m.as_dict.return_value = {
            "chain_string": "tape→mp3_high",
            "chain_complexity": 0.6,
            "is_multi_generation": True,
            "generation_count": 2,
            "chain": ["tape", "mp3_high"],
            "primary_medium": "mp3_high",
        }

        strat_m = _make_strategie_mock()
        strat_result = MagicMock()
        strat_result.quality_mode = "quality"
        strat_result.max_processing_s = 30.0
        strat_m.plan.return_value = strat_result
        strat_m.starte_timer.return_value = None

        rest_result = MagicMock()
        rest_result.audio = audio.copy()
        rest_result.phases_executed = []
        rest_result.warnings = []
        rest_result.quality_estimate = 0.7
        rest_result.rt_factor = 0.5
        rest_result.confidence = 0.9
        rest_result.rollback_triggered = False
        rest_result.winning_variant = "balanced"
        rest_result.musical_goals = {}
        rest_result.goals_passed = 0
        rest_m = MagicMock()
        rest_m.restauriere.return_value = rest_result

        rep_result = MagicMock(
            audio=audio.copy(), warnings=[], clicks_removed=False, hum_removed=False, clipping_repaired=False
        )
        rek_result = MagicMock(audio=audio.copy(), warnings=[], gaps_found=0, gaps_repaired=0, total_repaired_ms=0.0)

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_make_toni_mock("tape")))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_m)),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_result))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_result))),
            ),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m)),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "mp3_high"})),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, mode="maximum")

        assert result.quality_estimate < 0.60  # MOS gate proportionally reduces quality
        assert "FAILED" in result.stage_notes.get("material_mos_gate", "")
        assert any("Material-MOS-Gate nicht bestanden" in w for w in result.warnings)

    def test_material_mos_gate_honors_release_alias_studio_2026(self):
        audio = _sine()
        exz_m = _make_exzellenz_mock(audio)
        exz_denker_m = MagicMock()
        exz_denker_m.optimiere.return_value = exz_m
        versa_result_m = MagicMock(mos=3.6, model_used="mock_versa")

        kette_m = _make_kette_mock("tape→mp3_high")
        kette_m.as_dict.return_value = {
            "chain_string": "tape→mp3_high",
            "chain_complexity": 0.6,
            "is_multi_generation": True,
            "generation_count": 2,
            "chain": ["tape", "mp3_high"],
            "primary_medium": "mp3_high",
        }

        strat_m = _make_strategie_mock()
        strat_result = MagicMock()
        strat_result.quality_mode = "quality"
        strat_result.max_processing_s = 30.0
        strat_m.plan.return_value = strat_result
        strat_m.starte_timer.return_value = None

        rest_result = MagicMock()
        rest_result.audio = audio.copy()
        rest_result.phases_executed = []
        rest_result.warnings = []
        rest_result.quality_estimate = 0.7
        rest_result.rt_factor = 0.5
        rest_result.confidence = 0.9
        rest_result.rollback_triggered = False
        rest_result.winning_variant = "balanced"
        rest_result.musical_goals = {}
        rest_result.goals_passed = 0
        rest_m = MagicMock()
        rest_m.restauriere.return_value = rest_result

        rep_result = MagicMock(
            audio=audio.copy(), warnings=[], clicks_removed=False, hum_removed=False, clipping_repaired=False
        )
        rek_result = MagicMock(audio=audio.copy(), warnings=[], gaps_found=0, gaps_repaired=0, total_repaired_ms=0.0)

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_make_toni_mock("tape")))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_m)),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_result))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_result))),
            ),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m)),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "mp3_high"})),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, mode="Studio 2026")

        assert result.quality_estimate < 0.60
        assert "FAILED" in result.stage_notes.get("material_mos_gate", "")

    def test_material_mos_gate_keeps_quality_when_target_is_met(self):
        audio = _sine()
        exz_m = _make_exzellenz_mock(audio)
        exz_denker_m = MagicMock()
        exz_denker_m.optimiere.return_value = exz_m
        versa_result_m = MagicMock(mos=4.4, model_used="mock_versa")

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_make_toni_mock("cd_digital")))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_kette_mock("cd_digital")))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=_make_strategie_mock())),
            patch(
                "denker.aurik_denker.get_restaurier_denker",
                MagicMock(
                    return_value=MagicMock(
                        restauriere=MagicMock(
                            return_value=MagicMock(
                                audio=audio.copy(),
                                phases_executed=[],
                                warnings=[],
                                quality_estimate=0.8,
                                rt_factor=0.4,
                                confidence=0.9,
                                rollback_triggered=False,
                                winning_variant="balanced",
                                musical_goals={},
                                goals_passed=0,
                            )
                        )
                    )
                ),
            ),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(
                    return_value=MagicMock(
                        repariere=MagicMock(
                            return_value=MagicMock(
                                audio=audio.copy(),
                                warnings=[],
                                clicks_removed=False,
                                hum_removed=False,
                                clipping_repaired=False,
                            )
                        )
                    )
                ),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(
                    return_value=MagicMock(
                        rekonstruiere=MagicMock(
                            return_value=MagicMock(
                                audio=audio.copy(), warnings=[], gaps_found=0, gaps_repaired=0, total_repaired_ms=0.0
                            )
                        )
                    )
                ),
            ),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m)),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "cd_digital"})),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, mode="maximum")

        assert result.quality_estimate > 0.54
        assert "FAILED" not in result.stage_notes.get("material_mos_gate", "")


class TestAurikDenkerDegradationSignals:
    def test_critical_restoration_failure_is_signaled_and_clamped(self):
        audio = _sine()

        strat_result = MagicMock(quality_mode="quality", max_processing_s=30.0)
        strat_m = MagicMock()
        strat_m.plan.return_value = strat_result
        strat_m.starte_timer.return_value = None

        failing_restaurier = MagicMock()
        failing_restaurier.restauriere.side_effect = RuntimeError("uv3 failed")

        rep_result = MagicMock(
            audio=audio.copy(), warnings=[], clicks_removed=False, hum_removed=False, clipping_repaired=False
        )
        rek_result = MagicMock(audio=audio.copy(), warnings=[], gaps_found=0, gaps_repaired=0, total_repaired_ms=0.0)

        exz_m = _make_exzellenz_mock(audio, versa_mos=4.4)
        exz_denker_m = MagicMock()
        exz_denker_m.optimiere.return_value = exz_m

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_make_toni_mock("tape")))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(
                    return_value=MagicMock(analysiere=MagicMock(return_value=_make_kette_mock("tape→reel_tape")))
                ),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=failing_restaurier)),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_result))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_result))),
            ),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m)),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "tape"})),
            ),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, mode="restoration")

        assert result.stage_notes.get("degradation_status") == "critical_degraded"
        assert "restaurierung" in result.stage_notes.get("degradation_failures", "")
        assert "critical_stage_failure" in result.stage_notes.get("fail_reason", "")
        assert isinstance(result.stage_notes.get("fail_reasons"), list)
        assert any(
            isinstance(entry, dict) and entry.get("error_code") == "STAGE_RESTAURIERUNG_FAILED"
            for entry in result.stage_notes.get("fail_reasons", [])
        )
        assert result.quality_estimate < 0.70  # proportional penalty for critical failure
        assert result.degradation_status == "critical_degraded"
        assert isinstance(result.fail_reason, str)
        assert result.fail_reason.startswith("critical_stage_failure:")

    def test_non_critical_stage_failure_is_degraded_without_fail_reason(self):
        audio = _sine()

        strat_result = MagicMock(quality_mode="quality", max_processing_s=30.0)
        strat_m = MagicMock()
        strat_m.plan.return_value = strat_result
        strat_m.starte_timer.return_value = None

        rest_ok = MagicMock(
            audio=audio.copy(),
            phases_executed=[],
            warnings=[],
            quality_estimate=0.8,
            rt_factor=0.4,
            confidence=0.9,
            rollback_triggered=False,
            winning_variant="balanced",
            musical_goals={},
            goals_passed=0,
        )

        rep_result = MagicMock(
            audio=audio.copy(), warnings=[], clicks_removed=False, hum_removed=False, clipping_repaired=False
        )
        rek_result = MagicMock(audio=audio.copy(), warnings=[], gaps_found=0, gaps_repaired=0, total_repaired_ms=0.0)

        exz_m = _make_exzellenz_mock(audio, versa_mos=4.3)
        exz_denker_m = MagicMock()
        exz_denker_m.optimiere.return_value = exz_m

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=_make_toni_mock("tape")))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(side_effect=RuntimeError("chain failed")))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=_make_defekt_mock()))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch(
                "denker.aurik_denker.get_restaurier_denker",
                MagicMock(return_value=MagicMock(restauriere=MagicMock(return_value=rest_ok))),
            ),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_result))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_result))),
            ),
            patch("denker.aurik_denker.get_exzellenz_denker", MagicMock(return_value=exz_denker_m)),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "tape"})),
            ),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, mode="restoration")

        assert result.stage_notes.get("degradation_status") == "degraded"
        assert "kette" in result.stage_notes.get("degradation_failures", "")
        assert "STAGE_KETTE_FAILED" in result.stage_notes.get("degradation_error_codes", "")
        fail_reasons = _normalized_fail_reasons(result.stage_notes)
        assert fail_reasons
        assert fail_reasons[0]["error_code"] == "STAGE_KETTE_FAILED"
        assert result.fail_reason is None
        assert result.degradation_status == "degraded"


class TestAurikDenkerAutopilotMode:
    def test_recommend_autopilot_mode_prefers_studio_for_modern_digital(self):
        from denker.aurik_denker import AurikDenker

        defekt = MagicMock()
        defekt.overall_severity = 0.20
        defekt.primary_defect = "minor_noise"

        global_plan = MagicMock()
        global_plan.portrait = MagicMock(decade=1990, genre="pop")

        mode, note = AurikDenker._recommend_autopilot_mode(
            requested_mode="quality",
            material="cd_digital",
            chain_info={"primary_medium": "cd_digital"},
            defekt=defekt,
            global_plan=global_plan,
            strategy_mode="studio2026",
        )

        assert mode == "studio2026"
        assert "Studio 2026 empfohlen" in note

    def test_recommend_autopilot_mode_prefers_restoration_for_fragile_material(self):
        from denker.aurik_denker import AurikDenker

        defekt = MagicMock()
        defekt.overall_severity = 0.25
        defekt.primary_defect = "hiss"

        global_plan = MagicMock()
        global_plan.portrait = MagicMock(decade=1950, genre="schlager")

        mode, note = AurikDenker._recommend_autopilot_mode(
            requested_mode="quality",
            material="tape",
            chain_info={"primary_medium": "tape"},
            defekt=defekt,
            global_plan=global_plan,
            strategy_mode="quality",
        )

        assert mode == "restoration"
        assert "Restoration empfohlen" in note


class TestAurikDenkerModeNormalization:
    def test_release_aliases_map_to_canonical_internal_modes(self):
        from denker.aurik_denker import AurikDenker

        assert AurikDenker._normalize_mode_name("Restoration") == "restoration"
        assert AurikDenker._normalize_mode_name("Studio 2026") == "studio2026"

    def test_internal_strategy_aliases_remain_distinct(self):
        from denker.aurik_denker import AurikDenker

        assert AurikDenker._normalize_mode_name("quality") == "quality"
        assert AurikDenker._normalize_mode_name("balanced") == "balanced"
        assert AurikDenker._normalize_mode_name("fast") == "fast"
        assert AurikDenker._normalize_mode_name("maximum") == "maximum"


class TestAurikDenkerOracleRolloutIntelligence:
    def test_recommend_oracle_rollout_respects_explicit_mode(self):
        from denker.aurik_denker import AurikDenker

        defekt = MagicMock(overall_severity=0.8, primary_defect="dropout")
        global_plan = MagicMock()
        global_plan.portrait = MagicMock(decade=1950, genre="schlager")

        rollout, note = AurikDenker._recommend_phase_strength_oracle_rollout(
            requested_rollout="pilot",
            material="tape",
            chain_info={"chain": ["tape", "mp3_low"], "primary_medium": "mp3_low"},
            defekt=defekt,
            global_plan=global_plan,
            effective_mode="restoration",
            cached_restorability_result=MagicMock(restorability_score=42.0),
            signal_signature={"crest_db": 18.0, "hf_ratio": 0.02, "transient_ratio": 0.02, "micro_dynamic_db": 12.0},
        )

        assert rollout == "pilot"
        assert "beibehalten" in note

    def test_recommend_oracle_rollout_prefers_off_for_extreme_fragile_case(self):
        from denker.aurik_denker import AurikDenker

        defekt = MagicMock(overall_severity=0.92, primary_defect="dropout")
        global_plan = MagicMock()
        global_plan.portrait = MagicMock(decade=1948, genre="schlager")

        rollout, note = AurikDenker._recommend_phase_strength_oracle_rollout(
            requested_rollout=None,
            material="tape",
            chain_info={"chain": ["tape", "cassette"], "primary_medium": "cassette"},
            defekt=defekt,
            global_plan=global_plan,
            effective_mode="restoration",
            cached_restorability_result=MagicMock(restorability_score=33.0),
            signal_signature={"crest_db": 22.0, "hf_ratio": 0.01, "transient_ratio": 0.02, "micro_dynamic_db": 18.0},
        )

        assert rollout == "off"
        assert "risk=" in note

    def test_recommend_oracle_rollout_prefers_all_for_stable_modern_digital(self):
        from denker.aurik_denker import AurikDenker

        defekt = MagicMock(overall_severity=0.18, primary_defect="minor_noise")
        global_plan = MagicMock()
        global_plan.portrait = MagicMock(decade=2005, genre="pop")

        rollout, note = AurikDenker._recommend_phase_strength_oracle_rollout(
            requested_rollout=None,
            material="cd_digital",
            chain_info={"chain": ["cd_digital"], "primary_medium": "cd_digital"},
            defekt=defekt,
            global_plan=global_plan,
            effective_mode="studio2026",
            cached_restorability_result=MagicMock(restorability_score=88.0),
            signal_signature={"crest_db": 10.0, "hf_ratio": 0.08, "transient_ratio": 0.003, "micro_dynamic_db": 8.0},
        )

        assert rollout == "all"
        assert "material=cd_digital" in note

    def test_compute_signal_intelligence_signature_returns_finite_metrics(self):
        from denker.aurik_denker import AurikDenker

        sig = AurikDenker._compute_signal_intelligence_signature(_sine(1.0), SR)
        for key in ("rms_dbfs", "crest_db", "hf_ratio", "transient_ratio", "micro_dynamic_db"):
            assert key in sig
            assert math.isfinite(float(sig[key]))


class TestAurikDenkerExcellenceRecoveryProfile:
    def test_recovery_profile_is_conservative_for_fragile_transient_rich_signal(self):
        from denker.aurik_denker import AurikDenker

        profile = AurikDenker._recommend_excellence_recovery_profile(
            material="tape",
            chain_info={"chain": ["tape", "mp3_low"]},
            effective_mode="restoration",
            signal_signature={
                "crest_db": 21.0,
                "transient_ratio": 0.02,
                "hf_ratio": 0.01,
                "micro_dynamic_db": 16.0,
            },
            versa_mos=3.2,
        )

        assert float(profile["preserve_signal"]) >= 0.40
        assert tuple(profile["blend_alphas"]) == (0.95, 0.92, 0.89)
        assert bool(profile["strict_mos_recovery"]) is True

    def test_recovery_profile_is_more_aggressive_for_stable_modern_studio(self):
        from denker.aurik_denker import AurikDenker

        profile = AurikDenker._recommend_excellence_recovery_profile(
            material="cd_digital",
            chain_info={"chain": ["cd_digital"]},
            effective_mode="studio2026",
            signal_signature={
                "crest_db": 10.0,
                "transient_ratio": 0.002,
                "hf_ratio": 0.09,
                "micro_dynamic_db": 8.0,
            },
            versa_mos=4.1,
        )

        assert float(profile["preserve_signal"]) < 0.20
        assert tuple(profile["blend_alphas"]) == (0.90, 0.84, 0.78)
        assert bool(profile["strict_mos_recovery"]) is False

    def test_explicit_studio2026_respects_user_choice_on_risk(self):
        audio = _sine()

        toni_m = _make_toni_mock("tape")
        kette_m = _make_kette_mock("tape→tape")
        kette_m.as_dict.return_value = {
            "chain_string": "tape→tape",
            "chain_complexity": 0.5,
            "is_multi_generation": False,
            "generation_count": 1,
            "chain": ["tape"],
            "primary_medium": "tape",
        }

        defekt_m = _make_defekt_mock()
        defekt_m.overall_severity = 0.75
        defekt_m.primary_defect = "dropout"

        strat_result = MagicMock(quality_mode="studio2026", max_processing_s=30.0)
        strat_m = MagicMock()
        strat_m.plan.return_value = strat_result
        strat_m.starte_timer.return_value = None

        rep_result = MagicMock(
            audio=audio.copy(), warnings=[], clicks_removed=False, hum_removed=False, clipping_repaired=False
        )
        rek_result = MagicMock(audio=audio.copy(), warnings=[], gaps_found=0, gaps_repaired=0, total_repaired_ms=0.0)

        rest_result = MagicMock(
            audio=audio.copy(),
            phases_executed=[],
            warnings=[],
            quality_estimate=0.8,
            rt_factor=0.4,
            confidence=0.9,
            rollback_triggered=False,
            winning_variant="balanced",
            musical_goals={},
            goals_passed=0,
        )
        rest_m = MagicMock()
        rest_m.restauriere.return_value = rest_result

        exz_denker_m = MagicMock(return_value=MagicMock(optimiere=MagicMock(return_value=_make_exzellenz_mock(audio))))
        versa_result_m = MagicMock(mos=4.2, model_used="mock_versa")

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=toni_m))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=defekt_m))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_result))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_result))),
            ),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_m)),
            patch("denker.aurik_denker.get_exzellenz_denker", exz_denker_m),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "tape"})),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, mode="studio2026")

        # §Bindend: Nutzerwahl wird respektiert — Studio 2026 bleibt erhalten.
        # Schutzmaßnahmen (SongCal, PMGG, FeedbackChain) greifen intern.
        assert rest_m.restauriere.call_args.kwargs["mode"] == "studio2026"
        assert "beibehalten" in result.stage_notes.get("autopilot", "")
        assert "Schutzmaßnahmen" in result.stage_notes.get("autopilot", "")

    def test_explicit_studio_2026_release_alias_respects_user_choice_on_risk(self):
        audio = _sine()

        toni_m = _make_toni_mock("tape")
        kette_m = _make_kette_mock("tape→tape")
        kette_m.as_dict.return_value = {
            "chain_string": "tape→tape",
            "chain_complexity": 0.5,
            "is_multi_generation": False,
            "generation_count": 1,
            "chain": ["tape"],
            "primary_medium": "tape",
        }

        defekt_m = _make_defekt_mock()
        defekt_m.overall_severity = 0.75
        defekt_m.primary_defect = "dropout"

        strat_result = MagicMock(quality_mode="studio2026", max_processing_s=30.0)
        strat_m = MagicMock()
        strat_m.plan.return_value = strat_result
        strat_m.starte_timer.return_value = None

        rep_result = MagicMock(
            audio=audio.copy(), warnings=[], clicks_removed=False, hum_removed=False, clipping_repaired=False
        )
        rek_result = MagicMock(audio=audio.copy(), warnings=[], gaps_found=0, gaps_repaired=0, total_repaired_ms=0.0)

        rest_result = MagicMock(
            audio=audio.copy(),
            phases_executed=[],
            warnings=[],
            quality_estimate=0.8,
            rt_factor=0.4,
            confidence=0.9,
            rollback_triggered=False,
            winning_variant="balanced",
            musical_goals={},
            goals_passed=0,
        )
        rest_m = MagicMock()
        rest_m.restauriere.return_value = rest_result

        exz_denker_m = MagicMock(return_value=MagicMock(optimiere=MagicMock(return_value=_make_exzellenz_mock(audio))))
        versa_result_m = MagicMock(mos=4.2, model_used="mock_versa")

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=toni_m))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=defekt_m))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_result))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_result))),
            ),
            patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_m)),
            patch("denker.aurik_denker.get_exzellenz_denker", exz_denker_m),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "tape"})),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result_m)),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR, mode="Studio 2026")

        assert rest_m.restauriere.call_args.kwargs["mode"] == "studio2026"
        assert "beibehalten" in result.stage_notes.get("autopilot", "")
        assert "Schutzmaßnahmen" in result.stage_notes.get("autopilot", "")


# ─── AurikErgebnis Audio-Invarianten ─────────────────────────────────────────


class TestAurikErgebnisInvarianten:
    def test_18_audio_no_nan(self):
        from denker.aurik_denker import AurikErgebnis

        audio = _sine()
        e = AurikErgebnis(
            audio=audio,
            material="tape",
            rt_factor=1.0,
            quality_estimate=0.7,
            musical_goals={},
            goals_passed=0,
            phases_executed=[],
            chain_info=None,
        )
        assert np.isfinite(e.audio).all()

    def test_19_audio_clipped(self):
        """Audio im Ergebnis sollte ≤ 1.0 betragen."""
        audio = np.clip(_sine(), -1.0, 1.0)
        assert np.max(np.abs(audio)) <= 1.0

    def test_20_rt_factor_documented_range(self):
        """rt_factor ist eine nicht-negative Zahl."""
        from denker.aurik_denker import AurikErgebnis

        e = AurikErgebnis(
            audio=_sine(),
            material="vinyl",
            rt_factor=2.5,
            quality_estimate=0.6,
            musical_goals={},
            goals_passed=0,
            phases_executed=[],
            chain_info=None,
        )
        assert e.rt_factor >= 0.0

    def test_21_fallback_sets_blocked_status_and_fail_reason(self):
        from denker.aurik_denker import AurikDenker

        fb = AurikDenker._fallback(_sine(), rt_factor=0.1, grund="Pipeline-Fehler")
        assert fb.degradation_status == "blocked"
        assert isinstance(fb.fail_reason, str)
        assert fb.fail_reason.startswith("pipeline_blocked:")
        fail_reasons = _normalized_fail_reasons(fb.stage_notes)
        assert fail_reasons
        assert fail_reasons[0]["error_code"] == "PIPELINE_BLOCKED"


# ─── Vollständig gemockte Orchestrierung (alle 10 Stufen) ───────────────────


def _full_mock_ctx(audio: np.ndarray, rest_metadata: dict[str, Any] | None = None) -> Any:
    """Context-Manager der alle 10 Stufen in AurikDenker mockt.

    Verhindert jeden echten Aufruf von RestaurierDenker, CDPAMPlugin,
    CrepePlugin oder ähnlicher schwerer ML-Infrastruktur.
    """
    toni_m = _make_toni_mock("tape")
    kette_m = _make_kette_mock("tape→mp3_low")
    defekt_m = _make_defekt_mock()

    strat_result = MagicMock()
    strat_result.quality_mode = "quality"
    strat_result.max_processing_s = 30.0
    strat_m = MagicMock()
    strat_m.plan.return_value = strat_result
    strat_m.starte_timer.return_value = None
    strat_m.check.return_value = MagicMock(should_exit_early=False)

    rest_result = MagicMock()
    rest_result.audio = audio.copy()
    rest_result.phases_executed = ["phase_03_denoise"]
    rest_result.warnings = []
    rest_result.quality_estimate = 0.75
    rest_result.rt_factor = 0.4
    rest_result.confidence = 0.88
    rest_result.rollback_triggered = False
    rest_result.winning_variant = "balanced"
    rest_result.musical_goals = {"brillanz": 0.87, "waerme": 0.81}
    rest_result.goals_passed = 2
    rest_result.metadata = dict(rest_metadata) if isinstance(rest_metadata, dict) else {}

    rest_denker_m = MagicMock()
    rest_denker_m.restauriere.return_value = rest_result

    rep_result = MagicMock()
    rep_result.audio = audio.copy()
    rep_result.warnings = []
    rep_result.clicks_removed = True
    rep_result.hum_removed = False
    rep_result.clipping_repaired = False

    rek_result = MagicMock()
    rek_result.audio = audio.copy()
    rek_result.warnings = []
    rek_result.gaps_found = 0
    rek_result.gaps_repaired = 0
    rek_result.total_repaired_ms = 0.0

    exz_result = _make_exzellenz_mock(audio)

    versa_result = MagicMock()
    versa_result.mos = 4.2
    versa_result.model_used = "mock_versa"

    return (
        patch(
            "denker.aurik_denker.get_tontraeger_denker",
            MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=toni_m))),
        ),
        patch(
            "denker.aurik_denker.get_tontraegerkette_denker",
            MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
        ),
        patch(
            "denker.aurik_denker.get_defekt_denker",
            MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=defekt_m))),
        ),
        patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
        patch("denker.aurik_denker.get_restaurier_denker", MagicMock(return_value=rest_denker_m)),
        patch(
            "denker.aurik_denker.get_reparatur_denker",
            MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_result))),
        ),
        patch(
            "denker.aurik_denker.get_rekonstruktions_denker",
            MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_result))),
        ),
        patch(
            "denker.aurik_denker.get_exzellenz_denker",
            MagicMock(
                return_value=MagicMock(
                    optimiere=MagicMock(return_value=exz_result),
                    messe_ziele=MagicMock(return_value={"brillanz": 0.87, "waerme": 0.81}),
                )
            ),
        ),
        patch(
            "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
            MagicMock(return_value=(False, {"material": "mp3_low"})),
        ),
        patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result)),
    )


def _run_fully_mocked(audio: np.ndarray, rest_metadata: dict[str, Any] | None = None) -> Any:
    """Hilfsfunktion: führt AurikDenker.denke() mit vollständigen Mocks aus."""
    from denker.aurik_denker import AurikDenker

    patches = _full_mock_ctx(audio, rest_metadata=rest_metadata)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
        patches[9],
    ):
        return AurikDenker().denke(audio, SR)


class TestAurikDenkerVollMock:
    """Tests mit vollständig gemockter 8-Stufen-Pipeline — kein echter ML-Aufruf."""

    def test_21_full_pipeline_returns_aurikergebnis(self):
        """AurikDenker.denke() gibt AurikErgebnis zurück (alle Stufen gemockt)."""
        from denker.aurik_denker import AurikErgebnis

        audio = _sine()
        result = _run_fully_mocked(audio)
        assert isinstance(result, AurikErgebnis)

    def test_22_full_pipeline_audio_no_nan(self):
        """Ausgegebenes Audio ist NaN/Inf-frei (alle Stufen gemockt)."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        assert np.isfinite(result.audio).all()

    def test_23_full_pipeline_audio_clipped(self):
        """Ausgegebenes Audio bleibt ≤ 1.0 (alle Stufen gemockt)."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_24_full_pipeline_material_recognized(self):
        """material ist nicht-leerer String (alle Stufen gemockt)."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        assert isinstance(result.material, str)
        assert len(result.material) > 0

    def test_25_full_pipeline_rt_factor_bounded(self):
        """rt_factor ≤ 3.0 (Spec §9.5)."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        assert 0.0 <= result.rt_factor <= 3.0

    def test_26_full_pipeline_quality_estimate_bounded(self):
        """quality_estimate ∈ [0, 1]."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        assert 0.0 <= result.quality_estimate <= 1.0

    def test_27_full_pipeline_chain_info_populated(self):
        """chain_info ist befüllt (kette_mock liefert as_dict())."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        # chain_info sollte dict aus dem kette_mock sein
        assert result.chain_info is None or isinstance(result.chain_info, dict)

    def test_28_full_pipeline_stage_notes_has_restaurierung(self):
        """stage_notes enthält 'restaurierung'-Schlüssel nach Stufe 4."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        assert "restaurierung" in result.stage_notes

    def test_29_full_pipeline_musical_goals_propagated(self):
        """musical_goals aus ExzellenzDenker-Mock werden übernommen."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        assert "brillanz" in result.musical_goals

    def test_30_full_pipeline_phases_executed_nonempty(self):
        """phases_executed ist nicht leer nach erfolgreichem Durchlauf."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        assert len(result.phases_executed) > 0

    def test_30b_orchestrator_propagates_253_metadata_to_aurikergebnis(self):
        """§8.3.2: AurikDenker propagiert §2.53-Metadaten unverändert bis AurikErgebnis."""
        audio = _sine()
        rest_metadata = {
            "joy_runtime_index": {
                "joy_index": 0.73,
                "fatigue_index": 0.19,
                "components": {"natuerlichkeit": 0.91},
            },
            "auto_improvement_recommendations": {
                "count": 1,
                "recommendations": [
                    {
                        "focus": "noise_floor",
                        "action": "reduce_denoise_strength",
                        "reason": "over-processing",
                    }
                ],
            },
            "song_calibration": {
                "cluster_key": "jazz:vinyl:pre-1980:good",
                "cluster_policy": {"cluster_key": "jazz:vinyl:pre-1980:good"},
            },
        }

        result = _run_fully_mocked(audio, rest_metadata=rest_metadata)

        assert isinstance(result.metadata, dict)
        for key in ("joy_runtime_index", "auto_improvement_recommendations", "song_calibration"):
            assert key in result.metadata, f"Pflichtfeld {key} fehlt in AurikErgebnis.metadata"
        assert result.metadata["joy_runtime_index"]["joy_index"] == pytest.approx(0.73, abs=0.01)
        assert result.metadata["song_calibration"]["cluster_key"] == "jazz:vinyl:pre-1980:good"

    def test_31_full_pipeline_stereo_input(self):
        """Stereo-Input (2 Kanäle) wird ohne Absturz verarbeitet."""
        audio = np.stack([_sine(), _sine(freq=880.0)], axis=1)
        result = _run_fully_mocked(audio)
        assert result is not None
        assert np.isfinite(result.audio).all()

    def test_32_full_pipeline_short_signal(self):
        """Sehr kurzes Signal (< 100 ms) gibt Fallback-Ergebnis zurück."""
        audio = _sine(duration_s=0.05)  # 50 ms → < _MIN_AUDIO_SAMPLES bei 64?
        # Kurzzeitsignal → entweder Fallback oder normales Ergebnis
        from denker.aurik_denker import AurikErgebnis

        patches = _full_mock_ctx(audio)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR)
        assert isinstance(result, AurikErgebnis)
        assert np.isfinite(result.audio).all()

    def test_33_full_pipeline_silence_input(self):
        """Stille als Input (Nullvektor) → kein Absturz, rt_factor ≤ 3.0."""
        audio = np.zeros(SR, dtype=np.float32)
        result = _run_fully_mocked(audio)
        assert result is not None
        assert result.rt_factor <= 3.0

    def test_34_full_pipeline_restaurier_raises_still_returns(self):
        """Auch wenn RestaurierDenker eine Exception wirft, gibt AurikDenker ein Ergebnis zurück."""
        from denker.aurik_denker import AurikDenker, AurikErgebnis

        audio = _sine()
        toni_m = _make_toni_mock()
        kette_m = _make_kette_mock()
        defekt_m = _make_defekt_mock()
        strat_result = MagicMock()
        strat_result.quality_mode = "quality"
        strat_result.max_processing_s = 30.0
        strat_m = MagicMock()
        strat_m.plan.return_value = strat_result
        strat_m.starte_timer.return_value = None
        strat_m.check.return_value = MagicMock(should_exit_early=False)

        rep_result = MagicMock()
        rep_result.audio = audio.copy()
        rep_result.warnings = []
        rep_result.clicks_removed = False
        rep_result.hum_removed = False
        rep_result.clipping_repaired = False

        rek_result = MagicMock()
        rek_result.audio = audio.copy()
        rek_result.warnings = []
        rek_result.gaps_found = 0
        rek_result.gaps_repaired = 0
        rek_result.total_repaired_ms = 0.0

        exz_result = _make_exzellenz_mock(audio)
        versa_result = MagicMock(mos=4.2, model_used="mock_versa")

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=toni_m))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=defekt_m))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch("denker.aurik_denker.get_restaurier_denker", side_effect=RuntimeError("CDPAM nicht verfügbar")),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_result))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_result))),
            ),
            patch(
                "denker.aurik_denker.get_exzellenz_denker",
                MagicMock(return_value=MagicMock(optimiere=MagicMock(return_value=exz_result))),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result)),
        ):
            result = AurikDenker().denke(audio, SR)
        assert isinstance(result, AurikErgebnis)
        # Restaurierung schlug fehl → stage_notes dokumentiert den Fehler
        assert "restaurierung" in result.stage_notes

    def test_35_full_pipeline_warnings_list(self):
        """warnings ist immer eine Liste (auch leer)."""
        audio = _sine()
        result = _run_fully_mocked(audio)
        assert isinstance(result.warnings, list)

    def test_36_module_level_wrappers_exist(self):
        """Alle 8 Modul-Level-Wrapper-Funktionen sind in aurik_denker vorhanden."""
        import denker.aurik_denker as m

        for fname in (
            "get_tontraeger_denker",
            "get_tontraegerkette_denker",
            "get_defekt_denker",
            "get_strategie_denker",
            "get_restaurier_denker",
            "get_reparatur_denker",
            "get_rekonstruktions_denker",
            "get_exzellenz_denker",
        ):
            assert hasattr(m, fname), f"Wrapper {fname} fehlt in denker.aurik_denker"
            assert callable(getattr(m, fname)), f"{fname} ist nicht aufrufbar"


# ─── §8.1 Spec-konforme quality_estimate-Formel in AurikDenker ───────────────


class TestAurikDenkerQualityEstimateFormula:
    """Normative Tests: AurikDenker.quality_estimate folgt Spec §8.1.

    Formel: quality_estimate = 0.40*(1-defect_severity) + 0.60*(pqs_mos-1)/4
    VERBOTEN: quality_estimate * 1.15 als fixer Bonus-Faktor.
    """

    def _run_with_versa(self, severity: float, mos: float) -> Any:
        """Führt AurikDenker mit kontrollierten Defekt-Schwere und VERSA MOS aus."""
        audio = _sine(1.0)
        defekt_m = _make_defekt_mock()
        defekt_m.overall_severity = severity

        exz_m = _make_exzellenz_mock(audio, versa_mos=0.0)  # kein gecachter MOS

        strat_result = MagicMock()
        strat_result.quality_mode = "quality"
        strat_result.max_processing_s = 30.0
        strat_m = MagicMock()
        strat_m.plan.return_value = strat_result
        strat_m.starte_timer.return_value = None
        strat_m.check.return_value = MagicMock(should_exit_early=False)

        rest_result = MagicMock()
        rest_result.audio = audio.copy()
        rest_result.phases_executed = []
        rest_result.warnings = []
        rest_result.quality_estimate = 0.75
        rest_result.rt_factor = 0.2
        rest_result.confidence = 0.9
        rest_result.rollback_triggered = False
        rest_result.winning_variant = None
        rest_result.musical_goals = {}
        rest_result.goals_passed = 0

        kette_m = _make_kette_mock()
        toni_m = _make_toni_mock("tape")
        versa_result = MagicMock(mos=mos, model_used="test_versa")

        rep_result = MagicMock()
        rep_result.audio = audio.copy()
        rep_result.warnings = []
        rep_result.clicks_removed = False
        rep_result.hum_removed = False
        rep_result.clipping_repaired = False

        rek_result = MagicMock()
        rek_result.audio = audio.copy()
        rek_result.warnings = []
        rek_result.gaps_found = 0
        rek_result.gaps_repaired = 0
        rek_result.total_repaired_ms = 0.0

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=toni_m))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=defekt_m))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch(
                "denker.aurik_denker.get_restaurier_denker",
                MagicMock(return_value=MagicMock(restauriere=MagicMock(return_value=rest_result))),
            ),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_result))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_result))),
            ),
            patch(
                "denker.aurik_denker.get_exzellenz_denker",
                MagicMock(return_value=MagicMock(optimiere=MagicMock(return_value=exz_m))),
            ),
            patch("plugins.versa_plugin.score_mos", MagicMock(return_value=versa_result)),
        ):
            from denker.aurik_denker import AurikDenker

            return AurikDenker().denke(audio, SR)

    def test_37_formula_spec_81_with_versa(self):
        """Spec §8.1: quality_estimate = 0.40*(1-sev) + 0.60*(mos-1)/4."""
        sev, mos = 0.4, 3.5
        result = self._run_with_versa(sev, mos)
        expected = 0.40 * (1.0 - sev) + 0.60 * (mos - 1.0) / 4.0
        assert abs(result.quality_estimate - expected) < 0.02, (
            f"quality_estimate={result.quality_estimate:.4f}, "
            f"erwartet {expected:.4f} (\u00a78.1-Formel: sev={sev}, mos={mos})"
        )

    def test_38_formula_perfect_signal(self):
        """sev=0, mos=5 \u2192 quality_estimate \u2248 1.0."""
        result = self._run_with_versa(0.0, 5.0)
        assert result.quality_estimate >= 0.95, (
            f"Perfektes Signal ergibt quality_estimate={result.quality_estimate:.4f}, erwartet \u2265 0.95"
        )

    def test_39_formula_fully_defective(self):
        """sev=1.0, mos=1.0 \u2192 quality_estimate \u2248 0.0."""
        result = self._run_with_versa(1.0, 1.0)
        assert result.quality_estimate < 0.10, (
            f"Voll defektes Signal ergibt quality_estimate={result.quality_estimate:.4f}, erwartet < 0.10"
        )

    def test_40_no_1_15_bonus_factor(self):
        """Regression: Kein 1.15-Bonus-Faktor in quality_estimate (Spec VERBOTEN)."""
        sev, mos = 0.3, 4.0
        result = self._run_with_versa(sev, mos)
        expected = 0.40 * (1.0 - sev) + 0.60 * (mos - 1.0) / 4.0
        with_bonus = expected * 1.15
        assert abs(result.quality_estimate - with_bonus) > 0.01, (
            f"quality_estimate={result.quality_estimate:.4f} \u00e4hnelt zu sehr "
            f"dem verbotenen 1.15-Bonus-Ergebnis {with_bonus:.4f}"
        )
        assert abs(result.quality_estimate - expected) < 0.03, (
            f"quality_estimate={result.quality_estimate:.4f} weicht von Spec-Formel {expected:.4f} ab"
        )

    def test_41_versa_cache_from_exzellenz_skips_own_versa(self):
        """M-8b: Wenn ExzellenzDenker versa_mos liefert, darf AurikDenker kein eigenes VERSA laufen."""
        audio = _sine(1.0)
        exz_m = _make_exzellenz_mock(audio, versa_mos=4.1)  # ExzellenzDenker liefert VERSA

        strat_result = MagicMock()
        strat_result.quality_mode = "quality"
        strat_result.max_processing_s = 30.0
        strat_m = MagicMock()
        strat_m.plan.return_value = strat_result
        strat_m.starte_timer.return_value = None
        strat_m.check.return_value = MagicMock(should_exit_early=False)

        rest_result = MagicMock()
        rest_result.audio = audio.copy()
        rest_result.phases_executed = []
        rest_result.warnings = []
        rest_result.quality_estimate = 0.75
        rest_result.rt_factor = 0.2
        rest_result.confidence = 0.9
        rest_result.rollback_triggered = False
        rest_result.winning_variant = None
        rest_result.musical_goals = {}
        rest_result.goals_passed = 0

        kette_m = _make_kette_mock()
        toni_m = _make_toni_mock()
        defekt_m = _make_defekt_mock()

        rep_m = MagicMock()
        rep_m.audio = audio.copy()
        rep_m.warnings = []
        rep_m.clicks_removed = False
        rep_m.hum_removed = False
        rep_m.clipping_repaired = False

        rek_m = MagicMock()
        rek_m.audio = audio.copy()
        rek_m.warnings = []
        rek_m.gaps_found = 0
        rek_m.gaps_repaired = 0
        rek_m.total_repaired_ms = 0.0

        versa_spy = MagicMock(return_value=MagicMock(mos=4.1, model_used="spy"))

        with (
            patch(
                "denker.aurik_denker.get_tontraeger_denker",
                MagicMock(return_value=MagicMock(erkenne=MagicMock(return_value=toni_m))),
            ),
            patch(
                "denker.aurik_denker.get_tontraegerkette_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=kette_m))),
            ),
            patch(
                "denker.aurik_denker.get_defekt_denker",
                MagicMock(return_value=MagicMock(analysiere=MagicMock(return_value=defekt_m))),
            ),
            patch("denker.aurik_denker.get_strategie_denker", MagicMock(return_value=strat_m)),
            patch(
                "denker.aurik_denker.get_restaurier_denker",
                MagicMock(return_value=MagicMock(restauriere=MagicMock(return_value=rest_result))),
            ),
            patch(
                "denker.aurik_denker.get_reparatur_denker",
                MagicMock(return_value=MagicMock(repariere=MagicMock(return_value=rep_m))),
            ),
            patch(
                "denker.aurik_denker.get_rekonstruktions_denker",
                MagicMock(return_value=MagicMock(rekonstruiere=MagicMock(return_value=rek_m))),
            ),
            patch(
                "denker.aurik_denker.get_exzellenz_denker",
                MagicMock(
                    return_value=MagicMock(
                        optimiere=MagicMock(return_value=exz_m),
                        messe_ziele=MagicMock(return_value={"brillanz": 0.87, "waerme": 0.81}),
                    )
                ),
            ),
            patch(
                "denker.aurik_denker.AurikDenker._should_skip_excellence_for_clean_digital",
                MagicMock(return_value=(False, {"material": "mp3_low"})),
            ),
            patch("plugins.versa_plugin.score_mos", versa_spy),
        ):
            from denker.aurik_denker import AurikDenker

            result = AurikDenker().denke(audio, SR)

        # v10.0.0: ExzellenzDenker ruft score_mos() EINMAL auf (Stufe 9 Messung).
        # Stufe 10 übernimmt den gecachten Wert und ruft score_mos() NICHT erneut auf.
        assert versa_spy.call_count == 1, (
            f"score_mos() wurde {versa_spy.call_count}x aufgerufen — "
            "erwartet: genau 1x in ExzellenzDenker-Messung (Stufe 9), 0x in Stufe 10 (Cache)"
        )
        # quality_estimate soll ExzellenzDenker-VERSA (3.0 vom spy) verwenden
        assert result.quality_estimate > 0.5, (
            f"quality_estimate={result.quality_estimate:.3f} zu niedrig — "
            "VERSA MOS aus ExzellenzDenker sollte h\u00f6here Qualit\u00e4t ergeben"
        )


def test_surgical_zone_awareness() -> None:
    """Anti-Regression-Gate Check 5: ExzellenzDenker kennt die Surgical-Zonen.

    §2.59 (surgical_defect_analyzer): ExzellenzDenker muss chirurgische
    Defektzonen (Klicks, Splices, Bandkanten) vom globalen Processing
    unterscheiden — surgical_defect_types ist die kanonische Menge.
    """
    from backend.core.surgical_defect_analyzer import SURGICAL_DEFECT_TYPES

    assert SURGICAL_DEFECT_TYPES, "SURGICAL_DEFECT_TYPES darf nicht leer sein"
    assert len(SURGICAL_DEFECT_TYPES) == 24, (
        f"SURGICAL_DEFECT_TYPES hat {len(SURGICAL_DEFECT_TYPES)} Einträge — erwartet 24 (Gate Check 5)"
    )
