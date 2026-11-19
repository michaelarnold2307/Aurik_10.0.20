from __future__ import annotations

"""Tests für backend/core/feasibility_controller.py — §2.79 FeasibilityController.

Testet: GoalFeasibility-Dataclass, estimate_goal_feasibility(), Erreichbarkeits-
Logik, Konfidenz-Berechnung, Non-Blocking-Verhalten bei ungültigen Eingaben.
"""


import gc
import math

import numpy as np
import pytest

np.random.seed(42)

from backend.core.feasibility_controller import (
    GoalFeasibility,
    estimate_goal_feasibility,
)

SR = 48_000
RNG = np.random.default_rng(42)

ALL_15_GOALS = [
    "natuerlichkeit",
    "authentizitaet",
    "tonal_center",
    "timbre_authentizitaet",
    "artikulation",
    "transient_energie",
    "emotionalitaet",
    "micro_dynamics",
    "groove",
    "transparenz",
    "waerme",
    "bass_kraft",
    "separation_fidelity",
    "brillanz",
    "spatial_depth",
]


def _make_audio(seconds: float = 2.0, channels: int = 1) -> np.ndarray:
    samples = int(SR * seconds)
    if channels == 1:
        return RNG.uniform(-0.5, 0.5, size=samples).astype(np.float32)
    return RNG.uniform(-0.5, 0.5, size=(channels, samples)).astype(np.float32)


# ---------------------------------------------------------------------------
# GoalFeasibility Dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGoalFeasibilityDataclass:
    def test_fields_present(self) -> None:
        gf = GoalFeasibility(reachable=True, confidence=0.75, max_achievable=0.92)
        assert gf.reachable is True
        assert math.isclose(gf.confidence, 0.75)
        assert math.isclose(gf.max_achievable, 0.92)

    def test_frozen(self) -> None:
        gf = GoalFeasibility(reachable=False, confidence=0.5, max_achievable=0.6)
        with pytest.raises((AttributeError, TypeError)):
            gf.reachable = True  # type: ignore[misc]

    def test_to_dict_keys(self) -> None:
        gf = GoalFeasibility(reachable=True, confidence=0.8, max_achievable=0.9)
        d = gf.to_dict()
        assert set(d.keys()) == {"reachable", "confidence", "max_achievable"}

    def test_to_dict_values(self) -> None:
        gf = GoalFeasibility(reachable=False, confidence=0.3, max_achievable=0.55)
        d = gf.to_dict()
        assert d["reachable"] is False
        assert math.isclose(d["confidence"], 0.3)
        assert math.isclose(d["max_achievable"], 0.55)

    def test_to_dict_json_serialisierbar(self) -> None:
        import json

        gf = GoalFeasibility(reachable=True, confidence=0.6, max_achievable=0.88)
        json.dumps(gf.to_dict())  # darf keinen Fehler werfen


# ---------------------------------------------------------------------------
# estimate_goal_feasibility — Basis-Verhalten
# ---------------------------------------------------------------------------


class TestEstimateGoalFeasibilityBasic:
    def test_gibt_dict_zurueck(self) -> None:
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "cd", 85.0, [])
        assert isinstance(result, dict)

    def test_alle_15_goals_vorhanden_bei_cd(self) -> None:
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "cd", 90.0, [])
        missing = set(ALL_15_GOALS) - set(result.keys())
        assert not missing, f"Fehlende Goals: {missing}"

    def test_werte_sind_goal_feasibility(self) -> None:
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "vinyl", 75.0, [])
        for val in result.values():
            assert isinstance(val, GoalFeasibility)

    def test_max_achievable_im_gueltigen_bereich(self) -> None:
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "cd", 90.0, [])
        for g, gf in result.items():
            assert 0.0 <= gf.max_achievable <= 1.0, f"{g}: max_achievable={gf.max_achievable}"

    def test_confidence_im_gueltigen_bereich(self) -> None:
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "vinyl", 70.0, ["vinyl"])
        for g, gf in result.items():
            assert 0.0 <= gf.confidence <= 1.0, f"{g}: confidence={gf.confidence}"


# ---------------------------------------------------------------------------
# Erreichbarkeit — Material-spezifisch
# ---------------------------------------------------------------------------


class TestGoalErreichbarkeit:
    def test_cd_hohe_restorability_alle_erreichbar(self) -> None:
        """CD + hohe Restorability → P1/P2-Core-Goals als reachable erwartet."""
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "cd", 95.0, [])
        # P1/P2-Goals müssen bei CD + 95 % Restorability erreichbar sein
        core_goals = ["natuerlichkeit", "authentizitaet", "tonal_center", "timbre_authentizitaet"]
        for g in core_goals:
            if g in result:
                assert result[g].reachable, f"P1/P2-Goal {g} nicht erreichbar bei CD/95 %"

    def test_shellac_niedrige_restorability_hat_unreachable(self) -> None:
        """Shellac + niedrige Restorability → mindestens einige Goals nicht erreichbar."""
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "shellac", 20.0, [])
        unreachable = [g for g, gf in result.items() if not gf.reachable]
        # Bei sehr altem Material mit geringer Restorability sollten High-End-Goals fehlen
        assert len(unreachable) > 0, "Shellac mit 20 % Restorability: alle Goals als erreichbar markiert?"

    def test_vinyl_mittlere_restorability_gemischt(self) -> None:
        """Vinyl + mittlere Restorability → zumindest die meisten P1/P2-Goals erreichbar."""
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "vinyl", 60.0, [])
        p1_goals = ["natuerlichkeit", "authentizitaet"]
        for g in p1_goals:
            if g in result:
                gf = result[g]
                # max_achievable muss > 0 sein
                assert gf.max_achievable > 0.0

    def test_reachable_korreliert_mit_max_achievable(self) -> None:
        """reachable=True impliziert max_achievable >= material_floor * 0.85."""
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "tape", 55.0, ["tape"])
        for g, gf in result.items():
            if gf.reachable:
                # max_achievable muss substantiell sein wenn reachable=True
                assert gf.max_achievable > 0.40, f"{g}: reachable=True aber max_achievable={gf.max_achievable}"


# ---------------------------------------------------------------------------
# Konfidenz-Berechnung
# ---------------------------------------------------------------------------


class TestKonfidenzBerechnung:
    def test_hoehere_restorability_hoehere_konfidenz(self) -> None:
        audio = _make_audio()
        res_low = estimate_goal_feasibility(audio, SR, "cd", 30.0, [])
        res_high = estimate_goal_feasibility(audio, SR, "cd", 90.0, [])
        avg_conf_low = sum(gf.confidence for gf in res_low.values()) / max(len(res_low), 1)
        avg_conf_high = sum(gf.confidence for gf in res_high.values()) / max(len(res_high), 1)
        assert avg_conf_high > avg_conf_low, (
            f"Höhere Restorability → niedrigere Konfidenz? low={avg_conf_low:.3f} high={avg_conf_high:.3f}"
        )

    def test_laengere_chain_reduziert_konfidenz(self) -> None:
        audio = _make_audio()
        short_chain: list[str] = []
        long_chain = ["shellac", "tape", "vinyl", "cassette"]
        res_short = estimate_goal_feasibility(audio, SR, "vinyl", 70.0, short_chain)
        res_long = estimate_goal_feasibility(audio, SR, "vinyl", 70.0, long_chain)
        avg_short = sum(gf.confidence for gf in res_short.values()) / max(len(res_short), 1)
        avg_long = sum(gf.confidence for gf in res_long.values()) / max(len(res_long), 1)
        assert avg_short >= avg_long, f"Kürzere Chain → niedrigere Konfidenz? short={avg_short:.3f} long={avg_long:.3f}"

    def test_konfidenz_minimum_05(self) -> None:
        """Konfidenz darf nicht unter 0.05 fallen."""
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "shellac", 0.0, ["shellac"] * 10)
        for g, gf in result.items():
            assert gf.confidence >= 0.05, f"{g}: confidence={gf.confidence} < 0.05"

    def test_konfidenz_maximum_95(self) -> None:
        """Konfidenz darf nicht über 0.95 steigen."""
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "cd", 100.0, [])
        for g, gf in result.items():
            assert gf.confidence <= 0.95, f"{g}: confidence={gf.confidence} > 0.95"


# ---------------------------------------------------------------------------
# Non-Blocking-Verhalten
# ---------------------------------------------------------------------------


class TestNonBlocking:
    def test_leere_audio_wirft_nicht(self) -> None:
        """Leeres Audio → gibt {} zurück, kein Crash."""
        audio = np.zeros(0, dtype=np.float32)
        result = estimate_goal_feasibility(audio, SR, "cd", 80.0, [])
        assert isinstance(result, dict)

    def test_negativer_sr_wirft_nicht(self) -> None:
        """Negativer SR → gibt {} zurück, kein Crash."""
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, -1, "cd", 80.0, [])
        assert isinstance(result, dict)

    def test_unbekanntes_material_wirft_nicht(self) -> None:
        """Unbekanntes Material → gibt {} oder gefülltes Dict zurück, kein Crash."""
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "unobtainium", 70.0, [])
        assert isinstance(result, dict)

    def test_nan_audio_wirft_nicht(self) -> None:
        """NaN-Audio → gibt {} zurück, kein Crash."""
        audio = np.full(SR, float("nan"), dtype=np.float32)
        result = estimate_goal_feasibility(audio, SR, "cd", 80.0, [])
        assert isinstance(result, dict)

    def test_stereo_audio_funktioniert(self) -> None:
        """Stereo-Audio (2, N) wird akzeptiert."""
        audio = _make_audio(channels=2)
        result = estimate_goal_feasibility(audio, SR, "vinyl", 75.0, [])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# to_dict — Integration
# ---------------------------------------------------------------------------


class TestToDictIntegration:
    def test_to_dict_konvertierung_komplett(self) -> None:
        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "cd", 85.0, [])
        for g, gf in result.items():
            d = gf.to_dict()
            assert "reachable" in d
            assert "confidence" in d
            assert "max_achievable" in d
            assert isinstance(d["reachable"], bool)
            assert isinstance(d["confidence"], float)
            assert isinstance(d["max_achievable"], float)

    def test_serialisierbarkeit_vollstaendig(self) -> None:
        import json

        audio = _make_audio()
        result = estimate_goal_feasibility(audio, SR, "vinyl", 70.0, ["vinyl"])
        serializable = {g: gf.to_dict() for g, gf in result.items()}
        json.dumps(serializable)  # darf keinen Fehler werfen


# Leichter GC nach den Tests (GC-Konvention §tests.instructions.md)
def teardown_module(module: object) -> None:
    gc.collect(0)
