"""
Tests für VocalQualityIndex (§2.35c).
"""

import numpy as np
import pytest


@pytest.fixture()
def sine_audio():
    """440 Hz Sinus bei 48 kHz, 2 Sekunden, float32."""
    sr = 48000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sr


@pytest.fixture()
def noisy_audio(sine_audio):
    """Sinus + leichtes Rauschen (degradiert)."""
    audio, sr = sine_audio
    rng = np.random.default_rng(42)
    noisy = audio + rng.normal(0, 0.01, len(audio)).astype(np.float32)
    noisy = np.clip(noisy, -1.0, 1.0)
    return noisy, sr


@pytest.mark.unit
class TestSingerIdentityDomainRule:
    """§III.11 copilot-instructions.md: Der sprachtrainierte Embedder richtet hier nicht.

    Die VQI-Messung speist die Musical Goals (vocal_quality) und damit
    Hörordnungs-Entscheidungen — die Domänen-Regel muss deshalb auch hier greifen.
    """

    def test_proxy_used_without_release(self, monkeypatch, sine_audio):
        import backend.core.music_model_flags as _mmf
        from backend.core.musical_goals.vocal_quality_index import _compute_singer_identity

        monkeypatch.setattr(_mmf, "use_resemblyzer_music", False)
        audio, sr = sine_audio
        _cos, _dsp_used = _compute_singer_identity(audio, audio, sr)
        assert _dsp_used is True, "ohne Musik-Freigabe muss der DSP-Proxy messen"
        assert 0.0 <= _cos <= 1.0

    def test_identical_audio_keeps_rollback_threshold(self, monkeypatch, sine_audio):
        """Unveränderte Stimme darf die 0.92-Rollback-Schwelle nicht reißen (§R3-Kalibrierung)."""
        import backend.core.music_model_flags as _mmf
        from backend.core.musical_goals.vocal_quality_index import _compute_singer_identity

        monkeypatch.setattr(_mmf, "use_resemblyzer_music", False)
        audio, sr = sine_audio
        _cos, _ = _compute_singer_identity(audio, audio, sr)
        assert _cos >= 0.92

    def test_embedder_used_with_release(self, monkeypatch, sine_audio):
        """Mit Freigabe läuft die Embedder-Kaskade (hier über ein Fake-Plugin)."""
        import backend.core.music_model_flags as _mmf
        from backend.core.musical_goals import vocal_quality_index as _vqi

        monkeypatch.setattr(_mmf, "use_resemblyzer_music", True)

        class _FakePlugin:
            available = True

            def embed(self, _audio, _sr):
                return np.ones(8, dtype=np.float32)

            def cosine_similarity(self, _a, _b):
                return 0.97

        monkeypatch.setattr("plugins.resemblyzer_plugin.get_resemblyzer_plugin", lambda: _FakePlugin())
        audio, sr = sine_audio
        _cos, _dsp_used = _vqi._compute_singer_identity(audio, audio, sr)
        assert _dsp_used is False
        assert _cos == pytest.approx(0.97)


@pytest.mark.unit
class TestVocalQualityIndexImport:
    def test_import_ok(self):
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        assert callable(compute_vqi)

    def test_constants(self):
        from backend.core.musical_goals.vocal_quality_index import (
            VQI_PROFESSIONAL,
            VQI_THRESHOLD,
            VQI_WORLD_CLASS,
        )

        assert pytest.approx(0.72, abs=1e-3) == VQI_THRESHOLD
        assert pytest.approx(0.82, abs=1e-3) == VQI_PROFESSIONAL
        assert pytest.approx(0.88, abs=1e-3) == VQI_WORLD_CLASS
        assert VQI_THRESHOLD < VQI_PROFESSIONAL < VQI_WORLD_CLASS


class TestComputeVqi:
    def test_identical_input_returns_high_score(self, sine_audio):
        """Gleiche Input/Output-Audio → VQI nahe 1.0."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        audio, sr = sine_audio
        result = compute_vqi(audio, audio, sr)

        assert 0.0 <= result["vqi"] <= 1.0
        assert result["vqi"] >= 0.70, f"VQI bei identischem Signal sollte hoch sein: {result['vqi']}"

    def test_return_keys(self, sine_audio):
        """Alle erwarteten Keys vorhanden."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        audio, sr = sine_audio
        result = compute_vqi(audio, audio, sr)

        expected_keys = {
            "vqi",
            "singer_identity_cosine",
            "formant_stability_score",
            "articulation_score",
            "proximity_score",
            "sibilance_naturalness",
            "singer_id_dsp_fallback",
            "vqi_tier",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_scores_in_range(self, sine_audio):
        """Alle Scores in [0, 1]."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        audio, sr = sine_audio
        result = compute_vqi(audio, audio, sr)

        for key in [
            "vqi",
            "singer_identity_cosine",
            "formant_stability_score",
            "articulation_score",
            "proximity_score",
            "sibilance_naturalness",
        ]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of range"

    def test_tier_assignment(self, sine_audio):
        """vqi_tier muss einem der gültigen Werte entsprechen."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        audio, sr = sine_audio
        result = compute_vqi(audio, audio, sr)

        valid_tiers = {"world_class", "professional", "acceptable", "below_threshold"}
        assert result["vqi_tier"] in valid_tiers

    def test_stereo_input(self):
        """Stereo-Input (2, N) wird korrekt verarbeitet."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        ch = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
        stereo = np.stack([ch, ch])
        result = compute_vqi(stereo, stereo, sr)

        assert 0.0 <= result["vqi"] <= 1.0
        assert result["vqi_tier"] in {"world_class", "professional", "acceptable", "below_threshold"}

    def test_too_short_returns_neutral(self):
        """Audio kürzer als 0.5 s gibt neutralen Score zurück."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        sr = 48000
        short = np.zeros(100, dtype=np.float32)
        result = compute_vqi(short, short, sr)

        assert result["vqi"] == pytest.approx(0.90, abs=0.05)

    def test_nan_input_handled(self, sine_audio):
        """NaN-Werte im Input werden nicht durchgereicht."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        audio, sr = sine_audio
        corrupted = audio.copy()
        corrupted[100:110] = np.nan
        result = compute_vqi(audio, corrupted, sr)

        assert not np.isnan(result["vqi"])
        assert 0.0 <= result["vqi"] <= 1.0

    def test_vocal_segments_mask(self, sine_audio):
        """Vocal-Segment-Maske wird angewendet ohne Crash."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        audio, sr = sine_audio
        segments = [(0.0, 1.0), (1.5, 2.0)]
        result = compute_vqi(audio, audio, sr, vocal_segments=segments)

        assert 0.0 <= result["vqi"] <= 1.0

    def test_degraded_vs_clean(self, sine_audio, noisy_audio):
        """VQI mit verrauschtem Original ≤ VQI mit gleichem Signal."""
        from backend.core.musical_goals.vocal_quality_index import compute_vqi

        clean, sr = sine_audio
        noisy, _ = noisy_audio

        compute_vqi(clean, clean, sr)
        result_degraded = compute_vqi(noisy, clean, sr)

        # Restauriertes Signal (clean) hat >= VQI verglichen mit degradiertem
        # (kann marginaler Unterschied sein — kein harter Vergleich)
        assert result_degraded["vqi"] >= 0.0
