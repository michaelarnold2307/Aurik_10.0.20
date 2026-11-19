from __future__ import annotations

"""Tests für §3.0 Source-Aware-Restorer: Per-Stem-Phasenfilter, Remix, Fallback.

Testet:
- StemConfig-Filterung, Remix-Gains
- ONNX-Fallback bei fehlendem Modell
"""


import numpy as np
import pytest


@pytest.mark.unit
class TestStemConfig:
    """§3.0a: SourceAwareFahrplan — Per-Stem-Konfiguration."""

    def test_vocals_skips_aggressive_phases(self) -> None:
        from backend.core.source_aware_fahrplan import get_stem_config

        cfg = get_stem_config("vocals")
        assert cfg.name == "vocals"
        # Vocals: Denoise/Exciter/Dereverb MÜSSEN geskippt werden
        assert cfg.phase_strengths.get("phase_03_denoise", 1.0) == 0.0
        assert cfg.phase_strengths.get("phase_21_exciter", 1.0) == 0.0
        assert cfg.phase_strengths.get("phase_20_reverb_reduction", 1.0) == 0.0
        # De-Esser und EQ dürfen laufen
        assert cfg.phase_strengths.get("phase_19_de_esser", 0.0) > 0.0
        assert cfg.phase_strengths.get("phase_04_eq_correction", 0.0) > 0.0
        # Gain > 1.0 (Vocals hervorheben)
        assert cfg.remix_gain > 1.0
        # Default-Phasen werden geskippt
        assert cfg.skip_all_default is True

    def test_drums_skips_stereo_eq(self) -> None:
        from backend.core.source_aware_fahrplan import get_stem_config

        cfg = get_stem_config("drums")
        # Drums: kein Stereo, kein EQ, kein Exciter
        assert cfg.phase_strengths.get("phase_22_stereo_enhancement", 1.0) == 0.0
        assert cfg.phase_strengths.get("phase_04_eq_correction", 1.0) == 0.0
        assert cfg.phase_strengths.get("phase_21_exciter", 1.0) == 0.0
        # Transient-Preservation maximal
        assert cfg.phase_strengths.get("phase_08_transient_preservation", 0.0) > 0.5

    def test_bass_allows_rumble_harmonics(self) -> None:
        from backend.core.source_aware_fahrplan import get_stem_config

        cfg = get_stem_config("bass")
        # Bass: Rumble-Filter aktiv, Harmonic-Enhancement via Spectral-Repair
        assert cfg.phase_strengths.get("phase_05_rumble_filter", 0.0) > 0.0
        assert cfg.phase_strengths.get("phase_23_spectral_repair", 0.0) > 0.0
        # Kein Exciter/Presence
        assert cfg.phase_strengths.get("phase_21_exciter", 1.0) == 0.0
        assert cfg.phase_strengths.get("phase_38_presence_boost", 1.0) == 0.0

    def test_other_allows_all_phases(self) -> None:
        from backend.core.source_aware_fahrplan import get_stem_config

        cfg = get_stem_config("other")
        assert cfg.skip_all_default is False
        assert cfg.phase_strengths.get("_default", 0.0) == 1.0

    def test_filter_phases_for_stem_vocals_reduces(self) -> None:
        from backend.core.source_aware_fahrplan import filter_phases_for_stem

        full_plan = [
            "phase_03_denoise",
            "phase_04_eq_correction",
            "phase_19_de_esser",
            "phase_21_exciter",
            "phase_08_transient_preservation",
            "phase_22_stereo_enhancement",
        ]
        filtered = filter_phases_for_stem(full_plan, "vocals")
        # Nur erlaubte Phasen
        assert "phase_19_de_esser" in filtered
        assert "phase_04_eq_correction" in filtered
        # Geskippte Phasen
        assert "phase_03_denoise" not in filtered
        assert "phase_21_exciter" not in filtered
        assert "phase_22_stereo_enhancement" not in filtered

    def test_filter_phases_for_stem_drums_reduces(self) -> None:
        from backend.core.source_aware_fahrplan import filter_phases_for_stem

        full_plan = [
            "phase_03_denoise",
            "phase_04_eq_correction",
            "phase_08_transient_preservation",
            "phase_21_exciter",
        ]
        filtered = filter_phases_for_stem(full_plan, "drums")
        assert "phase_08_transient_preservation" in filtered
        assert "phase_04_eq_correction" not in filtered
        assert "phase_21_exciter" not in filtered

    def test_filter_phases_empty_plan(self) -> None:
        from backend.core.source_aware_fahrplan import filter_phases_for_stem

        result = filter_phases_for_stem([], "vocals")
        assert result == {}

    def test_remix_gains_sum_near_unity(self) -> None:
        from backend.core.source_aware_fahrplan import STEM_REMIX_GAINS

        total = sum(STEM_REMIX_GAINS.values())
        # Summe der Gains sollte nah an 1.0 × 4 sein (leichte Anpassungen)
        assert 3.9 < total < 4.2


class TestPNM:
    """§0: Primum non nocere — Source-Separation gefährdet Original nicht."""

    def test_fallback_fullmix_returns_original_shape(self) -> None:
        """Fallback auf Vollmix erhält Audio-Shape."""
        from backend.core.source_aware_restorer import _restore_fullmix

        audio = np.zeros(48000, dtype=np.float32)

        def fake_restore(a, **kw):
            return a * 1.1

        result = _restore_fullmix(audio, 48000, fake_restore, {})
        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_remix_preserves_dimensions(self) -> None:
        """_remix_stems erhält Stereo/Mono-Dimensionalität."""
        from backend.core.source_aware_restorer import _remix_stems

        # Mono
        stems = {"drums": np.ones(1000, dtype=np.float32) * 0.1}
        result = _remix_stems(stems, stems, (1000,))
        assert result.shape == (1000,)
        assert result.dtype == np.float32

        # Stereo
        stems_stereo = {"drums": np.ones((2, 1000), dtype=np.float32) * 0.1}
        result_stereo = _remix_stems(stems_stereo, stems_stereo, (2, 1000))
        assert result_stereo.shape == (2, 1000)

    def test_remix_clips_to_valid_range(self) -> None:
        """_remix_stems clippt auf [-1.0, 1.0]."""
        from backend.core.source_aware_restorer import _remix_stems

        stems = {
            "drums": np.ones(100, dtype=np.float32) * 0.8,
            "bass": np.ones(100, dtype=np.float32) * 0.8,
        }
        result = _remix_stems(stems, stems, (100,))
        assert result.max() <= 1.0
        assert result.min() >= -1.0
