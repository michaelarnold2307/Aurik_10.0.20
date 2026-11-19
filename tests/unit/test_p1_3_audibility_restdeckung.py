"""P1-3-Restdeckung — Audibility-Guards der verbleibenden Schwellwert-Guards.

Vertrag (§G10 (copilot-instructions.md), Hörordnung Ebene 2 = Maskierungs-
schwelle statt Mess-Null, §SOTA-PSY-A1):
    Ein maskierter Phasen-Delta ist unhörbar → Schwellwert-Guards lösen dafür
    keine Rücknahmen/Blend/Rollbacks aus.

Abgedeckte Guards (Restdeckung 2026-11-03):
- hnr_guard (§ATI-Form): effektive ΔHNR-Schwelle = max(3.0 dB, JND).
- mikrodynamik_guard (§SCK-Form): Korrelations-Schwelle relaxiert linear
  um max. 0.20 bei voller 6-dB-JND.
- vibrato_guard (§SCK-Form): Reduktionstoleranz = 10 % + max. 5 pp bei
  voller 6-dB-JND.
- hallucination_guard (§SCK-Form): nur hörbare Spectral-Novelty zählt.
- noise_texture_guard (§SCK-Form): Textur-Abweichung skaliert mit dem
  exponierten Delta-Anteil; vollständig maskiert → Distanz 0.

Regressionstests: maskierte vs. unmaskierte Fälle (JND über Monkeypatch
deterministisch gesetzt — die JND-Schätzung selbst ist in
test_p1_3_masking_jnd_guards.py getestet).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

SR = 48000


def _jnd(jnd_db: float, above_db: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(jnd_db=jnd_db, delta_above_db=above_db, threshold_db=-60.0)


# ── HNR-Guard: effektive Schwelle = max(3.0 dB, JND) (§ATI-Form) ───────────


class TestHNRMaskingJND:
    def test_unmasked_over_cleaning_triggers(self) -> None:
        from backend.core.dsp.hnr_guard import check_hnr_delta

        audio = np.zeros(8192, dtype=np.float32)
        with (
            patch("backend.core.dsp.hnr_guard.compute_hnr", side_effect=[10.0, 15.0]),
            patch(
                "backend.core.dsp.hnr_guard.estimate_delta_masking_jnd_db",
                return_value=_jnd(0.0, above_db=8.0),
            ),
        ):
            result = check_hnr_delta(audio, audio, SR)
        assert result["effective_threshold_db"] == pytest.approx(3.0)
        assert result["delta_hnr"] == pytest.approx(5.0)
        assert result["over_cleaned"] is True
        assert result["blend_ratio"] > 0.0

    def test_masked_over_cleaning_is_tolerated(self) -> None:
        from backend.core.dsp.hnr_guard import check_hnr_delta

        audio = np.zeros(8192, dtype=np.float32)
        with (
            patch("backend.core.dsp.hnr_guard.compute_hnr", side_effect=[10.0, 15.0]),
            patch(
                "backend.core.dsp.hnr_guard.estimate_delta_masking_jnd_db",
                return_value=_jnd(6.0),
            ),
        ):
            result = check_hnr_delta(audio, audio, SR)
        # ΔHNR 5.0 dB unter effektiver Schwelle max(3.0, 6.0) = 6.0 dB
        assert result["effective_threshold_db"] == pytest.approx(6.0)
        assert result["masking_jnd_db"] == pytest.approx(6.0)
        assert result["over_cleaned"] is False
        assert result["blend_ratio"] == 0.0

    def test_unmasked_threshold_never_below_fixed(self) -> None:
        from backend.core.dsp.hnr_guard import check_hnr_delta

        audio = np.zeros(8192, dtype=np.float32)
        with (
            patch("backend.core.dsp.hnr_guard.compute_hnr", side_effect=[10.0, 12.0]),
            patch(
                "backend.core.dsp.hnr_guard.estimate_delta_masking_jnd_db",
                return_value=_jnd(1.0, above_db=2.0),
            ),
        ):
            result = check_hnr_delta(audio, audio, SR)
        # max(fest, JND) = max(3.0, 1.0) = 3.0 — feste Schwelle bleibt Floor
        assert result["effective_threshold_db"] == pytest.approx(3.0)
        assert result["over_cleaned"] is False


# ── Mikrodynamik-Guard: §SCK-Form (max. −0.20 bei voller 6-dB-JND) ────────


class TestMikrodynamikMaskingJND:
    def test_without_delta_context_unchanged(self) -> None:
        from backend.core.dsp.mikrodynamik_guard import recommend_mikrodynamik_wet

        # Baseline ohne pre/post: 0.80 < floor 0.87 → nennenswerter Dry-Anteil
        wet = recommend_mikrodynamik_wet(0.80, material="unknown")
        assert wet < 1.0

    def test_masked_delta_relaxes_correlation_threshold(self) -> None:
        from backend.core.dsp.mikrodynamik_guard import recommend_mikrodynamik_wet

        pre = np.zeros(8192, dtype=np.float32)
        post = np.zeros(8192, dtype=np.float32)
        with patch(
            "backend.core.dsp.mikrodynamik_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(6.0),
        ):
            wet = recommend_mikrodynamik_wet(0.80, material="unknown", pre=pre, post=post, sr=SR)
        # Ziel 0.93 − 0.20 = 0.73 ≤ 0.80 → kein Blend nötig
        assert wet == pytest.approx(1.0)

    def test_unmasked_delta_keeps_strict_threshold(self) -> None:
        from backend.core.dsp.mikrodynamik_guard import recommend_mikrodynamik_wet

        pre = np.zeros(8192, dtype=np.float32)
        post = np.zeros(8192, dtype=np.float32)
        with patch(
            "backend.core.dsp.mikrodynamik_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(0.0, above_db=8.0),
        ):
            wet = recommend_mikrodynamik_wet(0.80, material="unknown", pre=pre, post=post, sr=SR)
        assert wet < 1.0

    def test_relaxation_is_bounded(self) -> None:
        from backend.core.dsp.mikrodynamik_guard import recommend_mikrodynamik_wet

        pre = np.zeros(8192, dtype=np.float32)
        post = np.zeros(8192, dtype=np.float32)
        with patch(
            "backend.core.dsp.mikrodynamik_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(6.0),
        ):
            wet = recommend_mikrodynamik_wet(0.70, material="unknown", pre=pre, post=post, sr=SR)
        # Relaxation endet bei 0.20 → Ziel 0.73; 0.70 bleibt darunter → Blend
        assert wet < 1.0


# ── Vibrato-Guard: §SCK-Form (10 % + max. 5 pp bei voller 6-dB-JND) ───────


class TestVibratoMaskingJND:
    def _audio(self) -> np.ndarray:
        rng = np.random.default_rng(21)
        t = np.arange(SR) / SR
        return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(SR)).astype(np.float32)

    def test_unmasked_reduction_fails(self) -> None:
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        audio = self._audio()
        with (
            patch(
                "backend.core.dsp.vibrato_guard._measure_vibrato_depth",
                side_effect=[8.0, 7.0],
            ),
            patch(
                "backend.core.dsp.vibrato_guard.estimate_delta_masking_jnd_db",
                return_value=_jnd(0.0, above_db=8.0),
            ),
        ):
            result = check_vibrato_depth_preservation(audio, audio.copy(), SR)
        # 12.5 % Reduktion > 10 % (keine JND-Relaxation)
        assert result.effective_tolerance_pct == pytest.approx(10.0)
        assert result.ok is False

    def test_masked_reduction_passes(self) -> None:
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        audio = self._audio()
        with (
            patch(
                "backend.core.dsp.vibrato_guard._measure_vibrato_depth",
                side_effect=[8.0, 7.0],
            ),
            patch(
                "backend.core.dsp.vibrato_guard.estimate_delta_masking_jnd_db",
                return_value=_jnd(6.0),
            ),
        ):
            result = check_vibrato_depth_preservation(audio, audio.copy(), SR)
        # 12.5 % ≤ 10 % + 5 pp = 15 % — maskierter Delta ist unhörbar
        assert result.masking_jnd_db == pytest.approx(6.0)
        assert result.effective_tolerance_pct == pytest.approx(15.0)
        assert result.ok is True

    def test_relaxation_cap_holds(self) -> None:
        from backend.core.dsp.vibrato_guard import check_vibrato_depth_preservation

        audio = self._audio()
        with (
            patch(
                "backend.core.dsp.vibrato_guard._measure_vibrato_depth",
                side_effect=[8.0, 6.5],
            ),
            patch(
                "backend.core.dsp.vibrato_guard.estimate_delta_masking_jnd_db",
                return_value=_jnd(6.0),
            ),
        ):
            result = check_vibrato_depth_preservation(audio, audio.copy(), SR)
        # 18.75 % > 15 % (Cap) → auch bei voller JND nicht ok
        assert result.ok is False


# ── Hallucination-Guard: nur hörbare Novelty zählt ────────────────────────


class TestHallucinationMaskingJND:
    def _audio(self) -> np.ndarray:
        rng = np.random.default_rng(33)
        return (0.2 * rng.standard_normal(SR)).astype(np.float32)

    def test_unmasked_novelty_rolls_back(self) -> None:
        from backend.core.dsp.hallucination_guard import check_hallucination

        pre, post = self._audio(), self._audio()
        with (
            patch(
                "backend.core.hallucination_guard.compute_spectral_novelty",
                return_value=(0.30, {}),
            ),
            patch(
                "backend.core.dsp.hallucination_guard.estimate_delta_masking_jnd_db",
                return_value=_jnd(0.0, above_db=8.0),
            ),
            patch(
                "backend.core.dsp.hallucination_guard._get_adaptive_rollback_threshold",
                return_value=0.15,
            ),
        ):
            result = check_hallucination(pre, post, sr=SR, mode="restoration", bw_extension_context=False)
        assert result.requires_rollback is True
        assert result.score_penalty == pytest.approx(0.3)
        assert result.metadata["spectral_novelty_audible"] == pytest.approx(0.30)

    def test_masked_novelty_does_not_roll_back(self) -> None:
        from backend.core.dsp.hallucination_guard import check_hallucination

        pre, post = self._audio(), self._audio()
        with (
            patch(
                "backend.core.hallucination_guard.compute_spectral_novelty",
                return_value=(0.30, {}),
            ),
            patch(
                "backend.core.dsp.hallucination_guard.estimate_delta_masking_jnd_db",
                return_value=_jnd(6.0),
            ),
            patch(
                "backend.core.dsp.hallucination_guard._get_adaptive_rollback_threshold",
                return_value=0.15,
            ),
        ):
            result = check_hallucination(pre, post, sr=SR, mode="restoration", bw_extension_context=False)
        # Vollständig maskiert → hörbare Novelty 0.0 → kein Rollback, keine Penalty
        assert result.requires_rollback is False
        assert result.score_penalty == 0.0
        assert result.metadata["spectral_novelty_audible"] == 0.0
        assert result.metadata["novelty_masking_jnd_db"] == pytest.approx(6.0)

    def test_partial_exposure_scales_novelty(self) -> None:
        from backend.core.dsp.hallucination_guard import check_hallucination

        pre, post = self._audio(), self._audio()
        with (
            patch(
                "backend.core.hallucination_guard.compute_spectral_novelty",
                return_value=(0.30, {}),
            ),
            patch(
                "backend.core.dsp.hallucination_guard.estimate_delta_masking_jnd_db",
                return_value=_jnd(3.0, above_db=3.0),
            ),
            patch(
                "backend.core.dsp.hallucination_guard._get_adaptive_rollback_threshold",
                return_value=0.15,
            ),
        ):
            result = check_hallucination(pre, post, sr=SR, mode="restoration", bw_extension_context=False)
        # Exponierter Anteil 3/(3+3) = 0.5 → hörbare Novelty 0.15
        assert result.metadata["spectral_novelty_audible"] == pytest.approx(0.15)
        assert result.requires_rollback is False


# ── Noise-Texture-Guard: Textur-Abweichung nur soweit hörbar ──────────────


def _whitened_residual() -> np.ndarray:
    """Residual mit stark negativem Slope (musikähnlich, < −5 dB/oct)."""
    rng = np.random.default_rng(41)
    n = 65536
    spec = np.fft.rfft(rng.standard_normal(n).astype(np.float64))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec *= 1.0 / (1.0 + freqs / 300.0) ** 2.0
    x = np.fft.irfft(spec, n=n).astype(np.float32)
    x /= np.max(np.abs(x)) + 1e-9
    return x


class TestNoiseTextureMaskingJND:
    def test_unmasked_whitening_is_flagged(self) -> None:
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = _whitened_residual()
        dist_plain = compute_noise_texture_distance(residual, "unknown", sr=SR)
        assert dist_plain > 0.25
        with patch(
            "backend.core.dsp.noise_texture_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(0.0, above_db=8.0),
        ):
            dist_exposed = compute_noise_texture_distance(residual, "unknown", sr=SR, pre=residual, post=residual * 0.0)
        # Kein maskierter Anteil → Distanz unverändert
        assert dist_exposed == pytest.approx(dist_plain)

    def test_masked_residual_reports_zero(self) -> None:
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = _whitened_residual()
        with patch(
            "backend.core.dsp.noise_texture_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(6.0),
        ):
            dist = compute_noise_texture_distance(residual, "unknown", sr=SR, pre=residual, post=residual * 0.0)
        assert dist == 0.0

    def test_partial_exposure_scales_distance(self) -> None:
        from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance

        residual = _whitened_residual()
        dist_plain = compute_noise_texture_distance(residual, "unknown", sr=SR)
        with patch(
            "backend.core.dsp.noise_texture_guard.estimate_delta_masking_jnd_db",
            return_value=_jnd(3.0, above_db=3.0),
        ):
            dist_half = compute_noise_texture_distance(residual, "unknown", sr=SR, pre=residual, post=residual * 0.0)
        assert dist_half == pytest.approx(0.5 * dist_plain, rel=1e-3)
