import pytest

"""Tests für PlaybackDeviceProfile (§PDV-1).

Spec: §PDV-1 Translation-EQ für Consumer-Geräte
"""

import numpy as np


@pytest.mark.unit
class TestPlaybackDeviceProfileModule:
    def test_list_device_ids_not_empty(self):
        from backend.core.playback_device_profile import list_device_ids

        ids = list_device_ids()
        assert len(ids) >= 5, "Mindestens 5 Geräteprofile erwartet"

    def test_get_cached_profile_known_device(self):
        from backend.core.playback_device_profile import get_cached_profile

        profile = get_cached_profile("consumer_headphone_avg")
        assert profile is not None
        assert profile.device_id == "consumer_headphone_avg"

    def test_get_cached_profile_alias(self):
        from backend.core.playback_device_profile import get_cached_profile

        # "headphone" ist ein Alias
        profile = get_cached_profile("headphone")
        assert profile is not None
        assert profile.device_id in ("consumer_headphone_avg",)

    def test_get_cached_profile_unknown_fallback(self):
        from backend.core.playback_device_profile import get_cached_profile

        profile = get_cached_profile("unbekanntes_geraet_xyz")
        # Fallback auf consumer_headphone_avg
        assert profile is not None
        assert profile.device_id == "consumer_headphone_avg"

    def test_inverse_curve_is_list(self):
        from backend.core.playback_device_profile import get_cached_profile

        profile = get_cached_profile("laptop_speaker")
        curve = profile.get_inverse_curve()
        # get_inverse_curve() gibt Liste von (freq, delta_db) Tupeln zurück
        assert isinstance(curve, list)
        assert len(curve) >= 3

    def test_apply_translation_eq_preserves_length(self):
        from backend.core.playback_device_profile import (
            apply_translation_eq,
            get_cached_profile,
        )

        sr = 48000
        audio = np.random.randn(2, sr * 3).astype(np.float32) * 0.3
        profile = get_cached_profile("consumer_headphone_avg")
        result = apply_translation_eq(audio, sr, profile, strength=0.5)
        assert result.shape == audio.shape

    def test_apply_translation_eq_zero_strength_passthrough(self):
        from backend.core.playback_device_profile import (
            apply_translation_eq,
            get_cached_profile,
        )

        sr = 48000
        audio = np.random.randn(48000).astype(np.float32) * 0.3
        profile = get_cached_profile("consumer_headphone_avg")
        result = apply_translation_eq(audio, sr, profile, strength=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_apply_translation_eq_peak_guard(self):
        """§0h: Pegelexplosion verhindert."""
        from backend.core.playback_device_profile import (
            apply_translation_eq,
            get_cached_profile,
        )

        sr = 48000
        # Sehr lautes Signal
        audio = np.ones(sr * 2, dtype=np.float32) * 0.95
        profile = get_cached_profile("laptop_speaker")
        result = apply_translation_eq(audio, sr, profile, strength=1.0)
        assert float(np.max(np.abs(result))) <= 1.01  # Kein Over-Clipping

    def test_singleton_returns_same_instance(self):
        from backend.core.playback_device_profile import get_cached_profile

        p1 = get_cached_profile("studio_headphone")
        p2 = get_cached_profile("studio_headphone")
        assert p1 is p2

    def test_airpods_profile_exists(self):
        from backend.core.playback_device_profile import get_cached_profile

        profile = get_cached_profile("airpods")
        assert profile is not None

    def test_max_correction_db_attribute_exists(self):
        """Profile hat max_correction_db Attribut."""
        from backend.core.playback_device_profile import get_cached_profile

        profile = get_cached_profile("smartphone_speaker")
        assert hasattr(profile, "max_correction_db")
        assert profile.max_correction_db > 0.0
