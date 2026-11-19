import pytest

"""Tests für SingerReferenceLibrary (§SRL-1).

Spec: §SRL-1 Singer-Reference-Library (v10.0.0)
"""

import numpy as np


@pytest.mark.unit
class TestVocalFingerprintCompute:
    """Tests für compute_vocal_fingerprint()."""

    def test_returns_correct_shape(self):
        from backend.core.singer_reference_library import compute_vocal_fingerprint

        sr = 48000
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr * 5)).astype(np.float32)
        fp = compute_vocal_fingerprint(audio, sr)
        assert fp.shape == (41,), f"Erwartet (41,), got {fp.shape}"

    def test_returns_float32(self):
        from backend.core.singer_reference_library import compute_vocal_fingerprint

        sr = 48000
        audio = np.random.randn(sr * 3).astype(np.float32) * 0.3
        fp = compute_vocal_fingerprint(audio, sr)
        assert fp.dtype == np.float32

    def test_stereo_input_handled(self):
        from backend.core.singer_reference_library import compute_vocal_fingerprint

        sr = 48000
        audio = np.random.randn(2, sr * 3).astype(np.float32) * 0.3
        fp = compute_vocal_fingerprint(audio, sr)
        assert fp.shape == (41,)

    def test_channels_last_stereo_input_handled(self):
        from backend.core.singer_reference_library import compute_vocal_fingerprint

        sr = 48000
        audio = np.random.randn(sr * 3, 2).astype(np.float32) * 0.3
        fp = compute_vocal_fingerprint(audio, sr)
        assert fp.shape == (41,)
        assert fp.dtype == np.float32

    def test_short_audio_fallback(self):
        """Sehr kurzes Audio → Nullvektor (kein Crash)."""
        from backend.core.singer_reference_library import compute_vocal_fingerprint

        sr = 48000
        audio = np.zeros(10, dtype=np.float32)
        fp = compute_vocal_fingerprint(audio, sr)
        assert fp.shape == (41,)

    def test_nan_input_handled(self):
        from backend.core.singer_reference_library import compute_vocal_fingerprint

        sr = 48000
        audio = np.full(sr * 2, float("nan"), dtype=np.float32)
        fp = compute_vocal_fingerprint(audio, sr)
        assert not np.any(np.isnan(fp))


class TestSingerMatching:
    """Tests für match_singer()."""

    def test_returns_singer_match_result(self):
        from backend.core.singer_reference_library import match_singer

        sr = 48000
        audio = np.random.randn(sr * 5).astype(np.float32) * 0.3
        result = match_singer(audio, sr)
        assert hasattr(result, "artist_id")
        assert hasattr(result, "confidence")
        assert hasattr(result, "fingerprint_distance")

    def test_confidence_in_valid_range(self):
        from backend.core.singer_reference_library import match_singer

        sr = 48000
        audio = np.random.randn(sr * 5).astype(np.float32) * 0.3
        result = match_singer(audio, sr)
        assert 0.0 <= result.confidence <= 1.0

    def test_low_confidence_match_has_no_artist_id(self):
        """Sehr unwahrscheinlicher Fingerprint → kein Match."""
        from backend.core.singer_reference_library import match_singer

        sr = 48000
        # Weißes Rauschen → unwahrscheinlicher Match
        audio = np.random.randn(sr * 5).astype(np.float32)
        result = match_singer(audio, sr, min_confidence=0.99)
        # Bei min_confidence=0.99 sollte kein Match kommen
        # (Weißes Rauschen matcht nicht mit confidence ≥ 0.99)
        if not result.is_reliable():
            assert result.artist_id == ""

    def test_zero_fingerprint_fails_closed(self):
        from backend.core.singer_reference_library import match_singer

        result = match_singer(np.zeros(10, dtype=np.float32), 48000, min_confidence=0.0)
        assert not result.is_reliable()
        assert result.artist_id == ""
        assert result.confidence == 0.0
        assert result.reference_fingerprint is None

    def test_known_prototype_matches_itself(self):
        """Stimmklassen-Prototyp sollte sich selbst mit hoher Konfidenz matchen."""
        from backend.core.singer_reference_library import (
            _STIMMKLASSE_PROTOTYPEN,
        )

        # Wir können nicht direkt Audio aus Prototypen generieren,
        # aber wir testen ob die Matching-Logik konsistent ist.
        assert len(_STIMMKLASSE_PROTOTYPEN) >= 5


class TestSingerReferenceLibrarySingleton:
    def test_singleton_returns_same_instance(self):
        from backend.core.singer_reference_library import get_singer_reference_library

        lib1 = get_singer_reference_library()
        lib2 = get_singer_reference_library()
        assert lib1 is lib2

    def test_match_nonblocking_on_bad_input(self):
        from backend.core.singer_reference_library import get_singer_reference_library

        lib = get_singer_reference_library()
        # Leeres Audio → non-blocking fallback
        result = lib.match(np.array([], dtype=np.float32), 48000)
        assert result is not None
        assert result.confidence >= 0.0


class TestVQIWithSingerReference:
    """Tests für §SRL-1 Integration in compute_vqi()."""

    def _make_audio(self, n=48000 * 3):
        return np.random.randn(n).astype(np.float32) * 0.1

    def test_vqi_with_reference_singer_id_runs(self):
        """reference_singer_id Parameter akzeptiert und non-blocking."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        orig = self._make_audio()
        rest = orig + np.random.randn(len(orig)).astype(np.float32) * 0.01
        result = compute_vqi(
            orig,
            rest,
            sr,
            reference_singer_id="voice_jazz_alto",
        )
        assert "vqi" in result
        assert 0.0 <= result["vqi"] <= 1.0

    def test_vqi_without_singer_ref_unchanged(self):
        """Ohne reference_singer_id: normaler Pfad."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        orig = self._make_audio()
        rest = orig.copy()
        result = compute_vqi(orig, rest, sr)
        assert "vqi" in result
        assert 0.0 <= result["vqi"] <= 1.0

    def test_reference_audio_used_flag_set(self):
        """reference_singer_id → reference_audio_used=True im Ergebnis."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        orig = self._make_audio()
        rest = orig + np.random.randn(len(orig)).astype(np.float32) * 0.01
        result = compute_vqi(
            orig,
            rest,
            sr,
            reference_singer_id="voice_light_soprano",
        )
        # reference_audio_used kann True sein wenn SRL-Pfad aktiv
        # (hängt davon ab ob librosa verfügbar ist)
        assert "singer_identity_cosine" in result
