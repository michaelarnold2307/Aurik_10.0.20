from types import SimpleNamespace

import numpy as np
import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3


@pytest.mark.unit
def test_cached_restorability_score_wins_over_estimator():
    score, source, result = UnifiedRestorerV3._resolve_pmgg_restorability_score(
        cached_result=SimpleNamespace(restorability_score=42.0),
        analysis_audio=np.zeros(64, dtype=np.float32),
        analysis_sample_rate=48000,
        material_key="vinyl",
        fallback_score=65.0,
        estimator_fn=lambda *args, **kwargs: SimpleNamespace(restorability_score=99.0),
    )

    assert score == 42.0
    assert source == "cached"
    assert getattr(result, "restorability_score", None) == 42.0


def test_incomplete_cached_result_falls_back_to_estimator():
    score, source, result = UnifiedRestorerV3._resolve_pmgg_restorability_score(
        cached_result=SimpleNamespace(),
        analysis_audio=np.zeros(64, dtype=np.float32),
        analysis_sample_rate=48000,
        material_key="shellac",
        fallback_score=65.0,
        estimator_fn=lambda *args, **kwargs: SimpleNamespace(restorability_score=37.5),
    )

    assert score == 37.5
    assert source == "estimated"
    assert getattr(result, "restorability_score", None) == 37.5


def test_estimator_failure_uses_neutral_fallback_score():
    def _raise_estimator(*args, **kwargs):
        raise RuntimeError("estimator unavailable")

    score, source, result = UnifiedRestorerV3._resolve_pmgg_restorability_score(
        cached_result=None,
        analysis_audio=np.zeros(64, dtype=np.float32),
        analysis_sample_rate=48000,
        material_key="tape",
        fallback_score=61.0,
        estimator_fn=_raise_estimator,
    )

    assert score == 61.0
    assert source == "fallback"
    assert result is None


def test_resolver_calls_estimator_with_real_signature():
    """Regression-Guard: Der Resolver darf den Estimator nur mit der echten
    Signatur estimate_restorability(audio, sr, material=...) aufrufen.

    Der frühere Bug übergab quality_mode="restoration" — ein kwarg, den
    estimate_restorability() nicht akzeptiert. Das warf still TypeError,
    landete im Fallback und lieferte für JEDEN Song konstant 70.0 (§2.29).
    Die alten Mocks (lambda *args, **kwargs) maskierten das, weil sie jeden
    kwarg schluckten. Dieser Mock spiegelt die echte Signatur strikt wider.
    """

    def _strict_estimator(audio, sr, material="unknown"):
        # Keine **kwargs: jeder zusätzliche kwarg löst TypeError aus →
        # der Resolver würde fälschlich in den Fallback fallen.
        assert isinstance(material, str)
        return SimpleNamespace(restorability_score=58.0)

    score, source, result = UnifiedRestorerV3._resolve_pmgg_restorability_score(
        cached_result=None,
        analysis_audio=np.zeros(64, dtype=np.float32),
        analysis_sample_rate=48000,
        material_key="mp3_low",
        fallback_score=70.0,
        estimator_fn=_strict_estimator,
    )

    assert source == "estimated", "Resolver fiel in Fallback → falscher kwarg an estimate_restorability"
    assert score == 58.0
    assert getattr(result, "restorability_score", None) == 58.0


def test_resolver_uses_real_estimate_restorability_function():
    """End-to-End-Guard gegen kwarg-Drift: Der Resolver muss mit der echten,
    importierten estimate_restorability-Funktion einen geschätzten Score
    liefern (source='estimated'), nicht den Fallback.
    """
    from backend.core.restorability_estimator import estimate_restorability

    rng = np.random.default_rng(0)
    audio = (0.05 * rng.standard_normal(48000)).astype(np.float32)

    score, source, result = UnifiedRestorerV3._resolve_pmgg_restorability_score(
        cached_result=None,
        analysis_audio=audio,
        analysis_sample_rate=48000,
        material_key="mp3_low",
        fallback_score=70.0,
        estimator_fn=estimate_restorability,
    )

    assert source == "estimated", "Echte estimate_restorability wurde nicht erfolgreich aufgerufen (kwarg-Drift?)"
    assert result is not None
    assert np.isfinite(float(score))
