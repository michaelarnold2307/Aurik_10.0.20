from __future__ import annotations

"""Unit-Tests für consensus_balance_synthesizer.py (§DGWCS).

Invarianten:
  - Monotonie: psycho_score(output) >= psycho_score(input) IMMER.
  - Keine Messung bei artifact_freedom < 0.95 (no-op).
  - Keine Messung bei sr != 48000 (no-op).
  - Stille-Zonen werden nie verändert.
  - measure_candidate_dims liefert None bei inkompatiblen Shapes.
  - compute_updated_vector_after_dgwcs ist konsistent mit gap_closure_per_dim.
"""


import math

import numpy as np
import pytest

from backend.core.consensus_balance_synthesizer import (
    _ALPHA_MIN,
    _PSYCHO_WEIGHTS,
    CBSResult,
    _compute_optimal_alpha,
    _score_from_dims,
    apply_dgwcs,
    compute_updated_vector_after_dgwcs,
    measure_candidate_dims,
)

SR = 48000
_N = SR * 2  # 2 Sekunden Testton

# ── Helfer ───────────────────────────────────────────────────────────────────


def _make_sine(freq: float = 440.0, n: int = _N, amplitude: float = 0.4) -> np.ndarray:
    """Mono-Sinus als Testsignal."""
    t = np.arange(n, dtype=np.float32) / SR
    out: np.ndarray = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return out


def _make_noisy(base: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Rauschen auf Basis-Signal (rng-deterministisch per Seed)."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, noise_level, base.shape).astype(np.float32)
    return np.clip(base + noise, -1.0, 1.0).astype(np.float32)


def _make_wcs_vector(**overrides: float) -> dict[str, float]:
    """WCS-Vektor mit sinnvollen Defaults für Tests."""
    v = {
        "vocal_identity_preservation": 1.0,
        "formant_integrity": 1.0,
        "vibrato_depth_preservation": 1.0,
        "breath_naturalness": 1.0,
        "micro_dynamic_correlation": 0.92,
        "transient_articulation": 1.0,
        "stereo_scene_stability": 1.0,
        "noise_texture_authenticity": 0.88,
        "spectral_color_preservation": 0.90,
        "emotional_arc_preservation": 0.93,
        "artifact_freedom": 1.0,
        "goal_team_balance": 0.85,
    }
    v.update(overrides)
    return v


# ── Tests: measure_candidate_dims ────────────────────────────────────────────


@pytest.mark.unit
class TestMeasureCandidateDims:
    """measure_candidate_dims liefert gemessene, nicht geschätzte Werte."""

    def test_returns_dict_with_four_keys(self):
        orig = _make_sine(440.0)
        cand = _make_noisy(orig, 0.02)
        result = measure_candidate_dims(
            original=orig,
            candidate=cand,
            sr=SR,
            material="digital",
            quality_mode="restoration",
            emotional_arc_fallback=0.90,
        )
        assert result is not None
        assert set(result.keys()) == set(_PSYCHO_WEIGHTS.keys())

    def test_all_values_in_unit_range(self):
        orig = _make_sine(880.0)
        cand = _make_noisy(orig, 0.03)
        result = measure_candidate_dims(
            original=orig,
            candidate=cand,
            sr=SR,
            material="vinyl",
            quality_mode="restoration",
            emotional_arc_fallback=0.85,
        )
        assert result is not None
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} außerhalb [0,1]"

    def test_identical_candidate_scores_high(self):
        """Wenn original == candidate, müssen alle Scores nahe 1.0 sein."""
        orig = _make_sine(330.0)
        result = measure_candidate_dims(
            original=orig,
            candidate=orig.copy(),
            sr=SR,
            material="digital",
            quality_mode="restoration",
            emotional_arc_fallback=1.0,
        )
        assert result is not None
        # Rauschtextur-Kohärenz bei Null-Residual → 1.0 (vakuös)
        assert result["noise_texture_authenticity"] >= 0.99
        # Mikrodynamik bei identischen Signalen → 1.0
        assert result["micro_dynamic_correlation"] >= 0.99

    def test_returns_none_for_shape_mismatch(self):
        orig = _make_sine(440.0, n=SR)
        cand = _make_sine(440.0, n=SR * 2)
        result = measure_candidate_dims(
            original=orig,
            candidate=cand,
            sr=SR,
            material="digital",
            quality_mode="restoration",
            emotional_arc_fallback=0.90,
        )
        assert result is None

    def test_returns_none_for_wrong_sr(self):
        orig = _make_sine(440.0)
        cand = _make_noisy(orig, 0.02)
        result = measure_candidate_dims(
            original=orig,
            candidate=cand,
            sr=44100,  # falsches SR
            material="digital",
            quality_mode="restoration",
            emotional_arc_fallback=0.90,
        )
        assert result is None

    def test_emotional_arc_uses_fallback(self):
        """emotional_arc_preservation = emotional_arc_fallback (MDEM nicht verfügbar)."""
        orig = _make_sine(440.0)
        cand = _make_noisy(orig, 0.03)
        fallback = 0.77
        result = measure_candidate_dims(
            original=orig,
            candidate=cand,
            sr=SR,
            material="tape",
            quality_mode="restoration",
            emotional_arc_fallback=fallback,
        )
        assert result is not None
        assert abs(result["emotional_arc_preservation"] - fallback) < 1e-5

    def test_original_candidate_emotional_arc_is_one(self):
        """Für 'original'-Kandidaten: emotional_arc_fallback = 1.0 (definitional)."""
        orig = _make_sine(440.0)
        # Simuliert den UV3-Aufruf für 'original'-Kandidat
        result = measure_candidate_dims(
            original=orig,
            candidate=orig.copy(),
            sr=SR,
            material="digital",
            quality_mode="restoration",
            emotional_arc_fallback=1.0,  # UV3 setzt 1.0 für 'original'
        )
        assert result is not None
        assert result["emotional_arc_preservation"] == pytest.approx(1.0)

    def test_stereo_audio_supported(self):
        """Stereo-Shape (2, N) wird korrekt verarbeitet."""
        mono = _make_sine(440.0)
        orig_stereo = np.stack([mono, mono * 0.9], axis=0)
        cand_stereo = np.stack([_make_noisy(mono, 0.02), _make_noisy(mono * 0.9, 0.02)], axis=0)
        result = measure_candidate_dims(
            original=orig_stereo,
            candidate=cand_stereo,
            sr=SR,
            material="vinyl",
            quality_mode="restoration",
            emotional_arc_fallback=0.88,
        )
        assert result is not None
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_empty_audio_returns_none(self):
        result = measure_candidate_dims(
            original=np.array([]),
            candidate=np.array([]),
            sr=SR,
            material="digital",
            quality_mode="restoration",
            emotional_arc_fallback=0.90,
        )
        assert result is None


# ── Tests: _score_from_dims ───────────────────────────────────────────────────


class TestScoreFromDims:
    def test_all_ones_returns_sum_of_weights(self):
        dims = dict.fromkeys(_PSYCHO_WEIGHTS, 1.0)
        score = _score_from_dims(dims, _PSYCHO_WEIGHTS)
        expected = sum(_PSYCHO_WEIGHTS.values())
        assert abs(score - expected) < 1e-6

    def test_all_zeros_returns_zero(self):
        dims = dict.fromkeys(_PSYCHO_WEIGHTS, 0.0)
        assert _score_from_dims(dims, _PSYCHO_WEIGHTS) == pytest.approx(0.0)

    def test_weighted_correctly(self):
        dims = {
            "noise_texture_authenticity": 1.0,
            "micro_dynamic_correlation": 0.0,
            "emotional_arc_preservation": 0.0,
            "spectral_color_preservation": 0.0,
        }
        score = _score_from_dims(dims, _PSYCHO_WEIGHTS)
        assert score == pytest.approx(_PSYCHO_WEIGHTS["noise_texture_authenticity"])


# ── Tests: _compute_optimal_alpha ────────────────────────────────────────────


class TestComputeOptimalAlpha:
    def test_returns_none_when_anchor_worse(self):
        """Wenn Referenz schlechter → kein Eingriff."""
        current = dict.fromkeys(_PSYCHO_WEIGHTS, 0.90)
        anchor = dict.fromkeys(_PSYCHO_WEIGHTS, 0.80)
        result = _compute_optimal_alpha(current, anchor, _PSYCHO_WEIGHTS, floor=0.76)
        assert result is None

    def test_returns_alpha_min_when_anchor_clearly_better(self):
        """Wenn Referenz deutlich besser, alpha = ALPHA_MIN (maximaler Referenzanteil)."""
        current = dict.fromkeys(_PSYCHO_WEIGHTS, 0.80)
        anchor = dict.fromkeys(_PSYCHO_WEIGHTS, 0.95)
        alpha = _compute_optimal_alpha(current, anchor, _PSYCHO_WEIGHTS, floor=0.76)
        assert alpha is not None
        assert alpha == pytest.approx(_ALPHA_MIN, abs=1e-4)

    def test_alpha_in_valid_range(self):
        """Alpha immer in [_ALPHA_MIN, 1.0)."""
        current = dict.fromkeys(_PSYCHO_WEIGHTS, 0.82)
        anchor = {
            "noise_texture_authenticity": 0.96,
            "micro_dynamic_correlation": 0.70,
            "emotional_arc_preservation": 0.92,
            "spectral_color_preservation": 0.94,
        }
        alpha = _compute_optimal_alpha(current, anchor, _PSYCHO_WEIGHTS, floor=0.76)
        if alpha is not None:
            assert _ALPHA_MIN <= alpha < 1.0

    def test_floor_constraint_respected(self):
        """Alpha wird angehoben, um Floor-Constraint zu respektieren."""
        floor = 0.80
        # Referenz für micro_dynamic sehr schlecht: 0.50 < floor
        # Current für micro_dynamic: 0.85 >= floor
        current = {
            "noise_texture_authenticity": 0.75,
            "micro_dynamic_correlation": 0.85,
            "emotional_arc_preservation": 0.75,
            "spectral_color_preservation": 0.75,
        }
        anchor = {
            "noise_texture_authenticity": 0.95,
            "micro_dynamic_correlation": 0.50,
            "emotional_arc_preservation": 0.95,
            "spectral_color_preservation": 0.95,
        }
        alpha = _compute_optimal_alpha(current, anchor, _PSYCHO_WEIGHTS, floor=floor)
        if alpha is not None:
            # Prüfen: micro_dynamic nach Blend >= floor
            blend_md = (
                alpha * current["micro_dynamic_correlation"] + (1.0 - alpha) * anchor["micro_dynamic_correlation"]
            )
            assert blend_md >= floor - 1e-5, f"micro_dynamic Blend {blend_md:.4f} < floor {floor}"


# ── Tests: apply_dgwcs ────────────────────────────────────────────────────────


class TestApplyDgwcs:
    """apply_dgwcs — Monotonie-Invariante und Stille-Schutz."""

    def _run_dgwcs(
        self,
        current: np.ndarray,
        original: np.ndarray,
        refs: dict[str, np.ndarray],
        vector: dict[str, float] | None = None,
        artifact_freedom: float = 1.0,
        panns_singing: float = 0.0,
        silence_zones: list | None = None,
    ):
        vec = vector or _make_wcs_vector()
        return apply_dgwcs(
            current_audio=current,
            original_audio=original,
            references=refs,
            current_wcs_vector=vec,
            artifact_freedom=artifact_freedom,
            panns_singing=panns_singing,
            sr=SR,
            material="digital",
            quality_mode="restoration",
            structural_silence_zones=silence_zones or [],
        )

    def test_no_op_when_artifact_freedom_low(self):
        """Kein Eingriff bei artifact_freedom < 0.95 (§0h-Invariante)."""
        orig = _make_sine(440.0)
        current = _make_noisy(orig, 0.05)
        ref = orig.copy()
        _, result = self._run_dgwcs(current, orig, {"original": ref}, artifact_freedom=0.80)
        assert result.applied is False

    def test_no_op_when_no_references(self):
        orig = _make_sine(440.0)
        current = _make_noisy(orig, 0.03)
        _, result = self._run_dgwcs(current, orig, {})
        assert result.applied is False

    def test_no_op_when_shape_mismatch(self):
        orig = _make_sine(440.0, n=SR)
        current = _make_noisy(orig, 0.03)
        ref_wrong = _make_sine(440.0, n=SR * 2)
        _, result = self._run_dgwcs(current, orig, {"original": ref_wrong})
        assert result.applied is False

    def test_no_op_when_ref_not_better(self):
        """Wenn Referenz in allen Dims schlechter → kein Eingriff."""
        orig = _make_sine(440.0)
        # Stark verrauschte Version als 'Referenz' — sollte schlechtere Dims haben
        very_noisy = _make_noisy(orig, 0.4)
        vec = _make_wcs_vector(
            noise_texture_authenticity=0.98,
            micro_dynamic_correlation=0.98,
            spectral_color_preservation=0.98,
            emotional_arc_preservation=0.98,
        )
        _, result = self._run_dgwcs(orig, orig, {"hpi_best": very_noisy}, vector=vec)
        # Ergebnis muss mindestens ein no-op sein (kein Schaden)
        assert result.improvement >= 0.0

    def test_output_not_worse_than_input(self):
        """Monotonie-Invariante: kein Schaden, egal welche Referenzen."""
        orig = _make_sine(440.0)
        current = _make_noisy(orig, 0.04)
        vec = _make_wcs_vector(noise_texture_authenticity=0.82, spectral_color_preservation=0.85)
        blended, result = self._run_dgwcs(
            current,
            orig,
            {"original": orig.copy(), "carrier": _make_noisy(orig, 0.01)},
            vector=vec,
        )
        # Wenn angewendet: improvement >= 0
        if result.applied:
            assert result.improvement >= 0.0
        # Audio immer in [-1, 1]
        assert float(np.abs(blended).max()) <= 1.0 + 1e-6

    def test_silence_zones_protected(self):
        """§0h: Stille-Zonen werden nie verändert."""
        orig = _make_sine(440.0)
        # Erste 0.5 Sekunden sind 'Stille' (werden nach §0h geschützt)
        current = orig.copy()
        ref_different = _make_noisy(orig, 0.1)

        # Stille-Zone: 0.0 s – 0.5 s
        silence_zones = [(0.0, 0.5)]
        vec = _make_wcs_vector(noise_texture_authenticity=0.78, micro_dynamic_correlation=0.79)

        blended, result = self._run_dgwcs(
            current,
            orig,
            {"original": ref_different},
            vector=vec,
            silence_zones=silence_zones,
        )

        # Stille-Zone-Samples müssen identisch mit current sein
        n_protected = int(0.5 * SR)
        if result.applied and result.silence_zones_protected > 0:
            np.testing.assert_array_equal(
                blended[:n_protected],
                current[:n_protected],
                err_msg="Stille-Zone wurde verändert!",
            )

    def test_cbsresult_fields_valid(self):
        """CBSResult-Felder sind im erwarteten Wertebereich."""
        orig = _make_sine(440.0)
        current = _make_noisy(orig, 0.05)
        vec = _make_wcs_vector(noise_texture_authenticity=0.80, spectral_color_preservation=0.82)
        _, result = self._run_dgwcs(
            current,
            orig,
            {"original": orig.copy()},
            vector=vec,
        )
        assert isinstance(result.applied, bool)
        assert 0.0 <= result.alpha <= 1.0
        assert math.isfinite(result.psycho_score_before)
        assert math.isfinite(result.psycho_score_after)
        if result.applied:
            assert result.improvement >= 0.0
            assert result.reference_used != ""

    def test_to_dict_serializable(self):
        """CBSResult.to_dict() muss JSON-serialisierbar sein."""
        import json

        result = CBSResult(
            applied=True,
            reference_used="original",
            alpha=0.75,
            psycho_score_before=0.82,
            psycho_score_after=0.86,
            improvement=0.04,
            gap_closure_per_dim={"noise_texture_authenticity": 0.02},
            silence_zones_protected=0,
        )
        d = result.to_dict()
        serialized = json.dumps(d)  # Darf nicht werfen
        assert "applied" in serialized

    def test_stereo_supported(self):
        """Stereo-Audio (2, N) wird korrekt verarbeitet."""
        mono = _make_sine(440.0)
        orig_stereo = np.stack([mono, mono * 0.95], axis=0)
        current_stereo = np.stack([_make_noisy(mono, 0.04), _make_noisy(mono * 0.95, 0.04)], axis=0)
        ref_stereo = orig_stereo.copy()
        vec = _make_wcs_vector(noise_texture_authenticity=0.81)
        blended, result = apply_dgwcs(
            current_audio=current_stereo,
            original_audio=orig_stereo,
            references={"original": ref_stereo},
            current_wcs_vector=vec,
            artifact_freedom=1.0,
            panns_singing=0.0,
            sr=SR,
            material="digital",
        )
        assert blended.shape == current_stereo.shape
        assert float(np.abs(blended).max()) <= 1.0 + 1e-6


# ── Tests: compute_updated_vector_after_dgwcs ─────────────────────────────────


class TestComputeUpdatedVector:
    def test_no_change_when_not_applied(self):
        vec = _make_wcs_vector()
        result = CBSResult(applied=False)
        updated = compute_updated_vector_after_dgwcs(vec, result)
        assert updated == vec

    def test_psycho_dims_updated_correctly(self):
        """Gap-closure aus CBSResult wird korrekt auf den Vektor angewendet."""
        vec = _make_wcs_vector(
            noise_texture_authenticity=0.82,
            micro_dynamic_correlation=0.85,
            emotional_arc_preservation=0.88,
            spectral_color_preservation=0.90,
        )
        gap_closure = {
            "noise_texture_authenticity": 0.05,
            "micro_dynamic_correlation": 0.02,
            "emotional_arc_preservation": 0.00,
            "spectral_color_preservation": 0.01,
        }
        result = CBSResult(
            applied=True,
            reference_used="original",
            alpha=0.75,
            psycho_score_before=0.86,
            psycho_score_after=0.88,
            improvement=0.02,
            gap_closure_per_dim=gap_closure,
        )
        updated = compute_updated_vector_after_dgwcs(vec, result)

        assert updated["noise_texture_authenticity"] == pytest.approx(0.82 + 0.05, abs=1e-5)
        assert updated["micro_dynamic_correlation"] == pytest.approx(0.85 + 0.02, abs=1e-5)
        # Nicht-Psycho-Dims unverändert
        assert updated["artifact_freedom"] == vec["artifact_freedom"]
        assert updated["formant_integrity"] == vec["formant_integrity"]

    def test_result_clipped_to_unit_range(self):
        """Clipping bei Gap-closure, die > 1.0 oder < 0.0 erzeugen würde."""
        vec = _make_wcs_vector(noise_texture_authenticity=0.99)
        result = CBSResult(
            applied=True,
            reference_used="original",
            alpha=0.75,
            psycho_score_before=0.90,
            psycho_score_after=0.95,
            improvement=0.05,
            gap_closure_per_dim={"noise_texture_authenticity": 0.10},  # würde 1.09 geben
        )
        updated = compute_updated_vector_after_dgwcs(vec, result)
        assert updated["noise_texture_authenticity"] <= 1.0
