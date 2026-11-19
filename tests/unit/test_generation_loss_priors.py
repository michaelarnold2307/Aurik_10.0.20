from __future__ import annotations

import pytest

"""Tests für GENERATION_LOSS-Priors in CausalDefectReasoner.

Prüft, dass die aktualisierten Prior-Werte für vinyl, mp3_low, mp3_high,
cd_digital und streaming korrekt gesetzt sind (>= neue Mindestwerte).

Normativ: §2.47 Adaptive-Intelligence – Bayesianische Priors müssen reale
Häufigkeitsverteilung von Multi-Generationen-Ketten abbilden.
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_priors(material: str) -> dict:
    """Holt die Material-Prior-Dictionary für ein gegebenes Material."""
    from backend.core.causal_defect_reasoner import MATERIAL_PRIORS

    return MATERIAL_PRIORS.get(material, {})


# ---------------------------------------------------------------------------
# Tests: GENERATION_LOSS priors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerationLossPriors:
    """generation_loss-Priors müssen auf rationalere Werte angehoben worden sein."""

    def test_01_vinyl_prior_at_least_005(self):
        p = _get_priors("vinyl")
        assert "generation_loss" in p
        assert p["generation_loss"] >= 0.05, f"vinyl generation_loss prior too low: {p['generation_loss']} < 0.05"

    def test_02_mp3_low_prior_at_least_006(self):
        p = _get_priors("mp3_low")
        assert "generation_loss" in p
        assert p["generation_loss"] >= 0.06, f"mp3_low generation_loss prior too low: {p['generation_loss']} < 0.06"

    def test_03_mp3_high_prior_at_least_004(self):
        p = _get_priors("mp3_high")
        assert "generation_loss" in p
        assert p["generation_loss"] >= 0.04, f"mp3_high generation_loss prior too low: {p['generation_loss']} < 0.04"

    def test_04_cd_digital_prior_at_least_003(self):
        p = _get_priors("cd_digital")
        assert "generation_loss" in p
        assert p["generation_loss"] >= 0.03, f"cd_digital generation_loss prior too low: {p['generation_loss']} < 0.03"

    def test_05_streaming_prior_at_least_004(self):
        p = _get_priors("streaming")
        assert "generation_loss" in p
        assert p["generation_loss"] >= 0.04, f"streaming generation_loss prior too low: {p['generation_loss']} < 0.04"

    def test_06_mp3_low_higher_than_mp3_high(self):
        """mp3_low typically endpoint of worse chains than mp3_high."""
        p_low = _get_priors("mp3_low")
        p_high = _get_priors("mp3_high")
        assert p_low["generation_loss"] >= p_high["generation_loss"]

    def test_07_mp3_low_highest_of_digital_materials(self):
        """mp3_low should have the highest generation_loss prior among digital materials."""
        materials = ["mp3_low", "mp3_high", "cd_digital", "streaming"]
        priors = {m: _get_priors(m).get("generation_loss", 0.0) for m in materials}
        mp3_low_val = priors["mp3_low"]
        for mat, val in priors.items():
            if mat != "mp3_low":
                assert mp3_low_val >= val, (
                    f"mp3_low ({mp3_low_val}) should have >= generation_loss prior than {mat} ({val})"
                )

    def test_08_priors_are_probabilities(self):
        """All priors must be valid probabilities [0.0, 1.0]."""
        for material in ["vinyl", "mp3_low", "mp3_high", "cd_digital", "streaming"]:
            p = _get_priors(material)
            val = p.get("generation_loss", -1.0)
            assert 0.0 <= val <= 1.0, f"generation_loss prior for {material}={val} out of [0,1] range"
