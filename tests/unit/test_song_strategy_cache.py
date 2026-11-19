"""Tests für SongStrategyCache (§SSC-1).

Spec: §SSC-1 Song-Strategy-Cache (v10.0.0)
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.unit
class TestComputeAudioFingerprint:
    def test_deterministic_same_audio(self):
        from backend.core.song_strategy_cache import compute_audio_fingerprint

        sr = 48000
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr), dtype="float64").astype(np.float32)
        h1 = compute_audio_fingerprint(audio, sr)
        h2 = compute_audio_fingerprint(audio, sr)
        assert h1 == h2

    def test_different_audio_different_hash(self):
        from backend.core.song_strategy_cache import compute_audio_fingerprint

        sr = 48000
        a1 = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32)
        a2 = np.sin(np.linspace(0, 2 * np.pi * 880, sr)).astype(np.float32)
        assert compute_audio_fingerprint(a1, sr) != compute_audio_fingerprint(a2, sr)

    def test_stereo_input(self):
        from backend.core.song_strategy_cache import compute_audio_fingerprint

        sr = 48000
        audio = np.random.randn(2, sr * 3).astype(np.float32) * 0.3
        h = compute_audio_fingerprint(audio, sr)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_short_audio_pads(self):
        from backend.core.song_strategy_cache import compute_audio_fingerprint

        sr = 48000
        audio = np.random.randn(100).astype(np.float32) * 0.3
        h = compute_audio_fingerprint(audio, sr)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_level_invariant(self):
        """RMS-Normalisierung → gleicher Hash bei unterschiedlichem Level."""
        from backend.core.song_strategy_cache import compute_audio_fingerprint

        sr = 48000
        base = np.sin(np.linspace(0, 2 * np.pi * 440, sr * 3)).astype(np.float32)
        h1 = compute_audio_fingerprint(base * 0.1, sr)
        h2 = compute_audio_fingerprint(base * 0.8, sr)
        assert h1 == h2

    def test_sample_rate_participates_in_hash(self):
        from backend.core.song_strategy_cache import compute_audio_fingerprint

        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 48000)).astype(np.float32)
        assert compute_audio_fingerprint(audio, 44100) != compute_audio_fingerprint(audio, 48000)


class TestSongStrategyCacheReadWrite:
    def _make_temp_cache(self):
        tmpdir = tempfile.mkdtemp()
        from backend.core.song_strategy_cache import SongStrategyCache

        return SongStrategyCache(cache_file=Path(tmpdir) / "test_cache.json")

    def _make_entry(self, song_id="abc123", mode="restoration", hpi=0.88):
        from backend.core.song_strategy_cache import PhaseStrategyEntry

        return PhaseStrategyEntry(
            song_id=song_id,
            mode=mode,
            last_used=time.time(),
            use_count=0,
            phase_strength_overrides={"phase_03": 0.65, "phase_29": 0.80},
            hpi_achieved=hpi,
            vqi_achieved=0.85,
            oqs_achieved=78.0,
            era="1960s",
            genre="jazz",
            material="vinyl",
            confidence=0.9,
        )

    def test_store_and_retrieve(self):
        cache = self._make_temp_cache()
        entry = self._make_entry()
        cache.store(entry)
        result = cache.get("abc123", "restoration")
        assert result is not None
        assert result.song_id == "abc123"
        assert result.hpi_achieved == pytest.approx(0.88, abs=1e-5)

    def test_get_nonexistent_returns_none(self):
        cache = self._make_temp_cache()
        assert cache.get("nonexistent", "restoration") is None

    def test_low_confidence_not_stored(self):
        cache = self._make_temp_cache()
        entry = self._make_entry()
        entry.confidence = 0.3  # Zu niedrig für Store
        cache.store(entry)
        assert cache.get("abc123", "restoration") is None

    def test_use_count_increments_on_get(self):
        cache = self._make_temp_cache()
        entry = self._make_entry()
        cache.store(entry)
        cache.get("abc123", "restoration")
        result = cache.get("abc123", "restoration")
        assert result is not None
        assert result.use_count >= 1

    def test_better_hpi_overwrites_existing(self):
        cache = self._make_temp_cache()
        cache.store(self._make_entry(hpi=0.80))
        cache.store(self._make_entry(hpi=0.90))  # Besser
        result = cache.get("abc123", "restoration")
        assert result is not None
        assert result.hpi_achieved == pytest.approx(0.90, abs=1e-5)

    def test_worse_hpi_does_not_overwrite(self):
        cache = self._make_temp_cache()
        cache.store(self._make_entry(hpi=0.90))
        cache.store(self._make_entry(hpi=0.70))  # Schlechter
        result = cache.get("abc123", "restoration")
        assert result is not None
        assert result.hpi_achieved == pytest.approx(0.90, abs=1e-5)

    def test_different_modes_stored_separately(self):
        cache = self._make_temp_cache()
        cache.store(self._make_entry(mode="restoration", hpi=0.85))
        cache.store(self._make_entry(mode="studio_2026", hpi=0.92))
        r1 = cache.get("abc123", "restoration")
        r2 = cache.get("abc123", "studio_2026")
        assert r1 is not None
        assert r2 is not None
        assert r1.hpi_achieved != r2.hpi_achieved

    def test_lru_eviction_at_max_entries(self):
        import tempfile

        from backend.core.song_strategy_cache import SongStrategyCache

        tmpdir = tempfile.mkdtemp()
        cache = SongStrategyCache(cache_file=Path(tmpdir) / "lru_test.json")
        # 510 Einträge → 10 älteste müssen entfernt werden
        from backend.core.song_strategy_cache import PhaseStrategyEntry

        for i in range(510):
            e = PhaseStrategyEntry(
                song_id=f"song_{i:04d}",
                mode="restoration",
                last_used=float(i),
                use_count=0,
                phase_strength_overrides={},
                hpi_achieved=0.85,
                vqi_achieved=0.82,
                oqs_achieved=75.0,
                confidence=0.9,
            )
            cache.store(e)
        assert cache.size() <= 500


class TestBuildStrategyEntry:
    def test_high_hpi_gives_high_confidence(self):
        from backend.core.song_strategy_cache import build_strategy_entry_from_result

        entry = build_strategy_entry_from_result(
            song_id="test",
            mode="restoration",
            phase_scores={"phase_03": 0.1},
            hpi=0.90,
            vqi=0.85,
            oqs=82.0,
        )
        assert entry.confidence >= 0.85

    def test_low_hpi_gives_lower_confidence(self):
        from backend.core.song_strategy_cache import build_strategy_entry_from_result

        entry = build_strategy_entry_from_result(
            song_id="test",
            mode="restoration",
            phase_scores={},
            hpi=0.50,
            vqi=0.60,
            oqs=50.0,
        )
        assert entry.confidence < 0.7
