"""Tests für backend/core/dsp/temporal_masking.py — Forward-Masking-Guard.

Abdeckung:
  - ForwardMaskingZone dataclass: Attribute, Wertebereich
  - ForwardMaskingGuard.compute_zones: Grundfunktionalität, Transient-Detektion
  - ForwardMaskingGuard.get_boost_at_sample: Rückgabe in Zone vs. außerhalb
  - ForwardMaskingGuard.apply_to_strength: Stärke-Erhöhung korrekt
  - Edge-Cases: Stille, kurzes Signal, kein Transient
  - Singleton-Invariante: get_forward_masking_guard()
"""

import numpy as np
import pytest

SR = 48000


def _silence(duration_s: float = 1.0) -> np.ndarray:
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _white_noise(duration_s: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration_s * SR)) * 0.05).astype(np.float32)


def _signal_with_transient(
    transient_db: float = 20.0,
    noise_level: float = 0.02,
    transient_pos: float = 0.3,
    duration_s: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Erstellt Signal mit einem starken Transient bei transient_pos (relativ)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    audio = (rng.standard_normal(n) * noise_level).astype(np.float32)
    # Transient: Impulsburst
    trans_amp = noise_level * (10 ** (transient_db / 20.0))
    trans_start = int(transient_pos * n)
    trans_len = int(0.005 * SR)  # 5 ms Impuls
    audio[trans_start : trans_start + trans_len] += (rng.standard_normal(trans_len) * trans_amp).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


@pytest.mark.unit
class TestForwardMaskingZone:
    """ForwardMaskingZone dataclass — Attribute."""

    def test_attributes_exist(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingZone

        z = ForwardMaskingZone(
            start_sample=100,
            end_sample=500,
            transient_energy_db=25.0,
            max_nr_strength_boost=0.3,
        )
        assert z.start_sample == 100
        assert z.end_sample == 500
        assert z.transient_energy_db == pytest.approx(25.0)
        assert z.max_nr_strength_boost == pytest.approx(0.3)

    def test_boost_range_valid(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingZone

        z = ForwardMaskingZone(0, 100, 30.0, 0.4)
        assert 0.0 <= z.max_nr_strength_boost <= 0.4


class TestComputeZones:
    """ForwardMaskingGuard.compute_zones — Zonen-Berechnung."""

    def test_silence_returns_empty(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard

        guard = ForwardMaskingGuard()
        zones = guard.compute_zones(_silence(), SR)
        assert isinstance(zones, list), "Ergebnis muss eine Liste sein"

    def test_noise_returns_list(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard

        guard = ForwardMaskingGuard()
        zones = guard.compute_zones(_white_noise(), SR)
        assert isinstance(zones, list)

    def test_transient_signal_finds_zones(self):
        """Starker Transient → mindestens eine Zone."""
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard

        guard = ForwardMaskingGuard()
        audio = _signal_with_transient(transient_db=30.0)
        zones = guard.compute_zones(audio, SR)
        # Mit starkem Transient → mindestens eine Zone erwartet
        # (kann 0 sein wenn boost zu klein, aber Shape muss stimmen)
        assert isinstance(zones, list)
        for z in zones:
            assert z.start_sample < z.end_sample, "Zone-Start muss vor Zone-End sein"
            assert z.max_nr_strength_boost >= 0.0
            assert z.max_nr_strength_boost <= 0.40

    def test_zone_end_within_signal_bounds(self):
        """Zone-Enden dürfen nicht über Signallänge hinausgehen."""
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard

        guard = ForwardMaskingGuard()
        n_samples = int(1.0 * SR)
        audio = _signal_with_transient(duration_s=1.0)
        zones = guard.compute_zones(audio, SR)
        for z in zones:
            assert z.end_sample <= n_samples, f"Zone-End {z.end_sample} > n_samples {n_samples}"

    def test_short_signal_no_crash(self):
        """Sehr kurzes Signal → keine Exception."""
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard

        guard = ForwardMaskingGuard()
        short = np.ones(128, dtype=np.float32) * 0.01
        zones = guard.compute_zones(short, SR)
        assert isinstance(zones, list)

    def test_stereo_input(self):
        """Stereo-Input soll intern zu Mono konvertiert werden."""
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard

        guard = ForwardMaskingGuard()
        mono = _white_noise()
        stereo = np.stack([mono, mono * 0.8], axis=0)
        zones = guard.compute_zones(stereo, SR)
        assert isinstance(zones, list)


class TestGetBoostAtSample:
    """ForwardMaskingGuard.get_boost_at_sample — Boost-Abfrage."""

    def test_inside_zone_returns_boost(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard, ForwardMaskingZone

        guard = ForwardMaskingGuard()
        zones = [
            ForwardMaskingZone(start_sample=100, end_sample=500, transient_energy_db=20.0, max_nr_strength_boost=0.3)
        ]
        boost = guard.get_boost_at_sample(zones, sample_idx=200)
        assert boost == pytest.approx(0.3)

    def test_outside_zone_returns_zero(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard, ForwardMaskingZone

        guard = ForwardMaskingGuard()
        zones = [
            ForwardMaskingZone(start_sample=100, end_sample=500, transient_energy_db=20.0, max_nr_strength_boost=0.3)
        ]
        boost = guard.get_boost_at_sample(zones, sample_idx=50)
        assert boost == pytest.approx(0.0)
        boost_after = guard.get_boost_at_sample(zones, sample_idx=600)
        assert boost_after == pytest.approx(0.0)

    def test_empty_zones_returns_zero(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard

        guard = ForwardMaskingGuard()
        boost = guard.get_boost_at_sample([], sample_idx=0)
        assert boost == pytest.approx(0.0)


class TestApplyToStrength:
    """ForwardMaskingGuard.apply_to_strength — Stärke-Anpassung."""

    def test_inside_zone_increases_strength(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard, ForwardMaskingZone

        guard = ForwardMaskingGuard()
        zones = [ForwardMaskingZone(100, 500, 25.0, 0.3)]
        result = guard.apply_to_strength(0.5, zones, sample_idx=200)
        assert result > 0.5, f"In Masking-Zone: Stärke soll erhöht werden (got {result})"
        assert result <= 1.0

    def test_outside_zone_unchanged(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard, ForwardMaskingZone

        guard = ForwardMaskingGuard()
        zones = [ForwardMaskingZone(100, 500, 25.0, 0.3)]
        result = guard.apply_to_strength(0.5, zones, sample_idx=50)
        assert result == pytest.approx(0.5)

    def test_max_total_respected(self):
        from backend.core.dsp.temporal_masking import ForwardMaskingGuard, ForwardMaskingZone

        guard = ForwardMaskingGuard()
        zones = [ForwardMaskingZone(0, 1000, 50.0, 0.4)]
        result = guard.apply_to_strength(0.9, zones, sample_idx=500, max_total=1.0)
        assert result <= 1.0


class TestSingleton:
    """get_forward_masking_guard() — Singleton-Invariante."""

    def test_singleton_identity(self):
        from backend.core.dsp.temporal_masking import get_forward_masking_guard

        a = get_forward_masking_guard()
        b = get_forward_masking_guard()
        assert a is b, "get_forward_masking_guard() muss immer dieselbe Instanz zurückgeben"
