"""Unit-Tests für den StreamingAudioPlayer (gapless, callback-basiert).

Tests validieren:
  - Korrekte Audio-Normalisierung und Shape-Handling
  - Crossfade-Overlay-Berechnung (Quellwechsel, Seek, Stop)
  - Sample-genaues Position-Tracking
  - Thread-Safety der Kern-API
  - Resample-Cache (LRU-Eviction)
  - Shutdown / Ressourcen-Freigabe
"""

import threading
import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: Mock sounddevice so tests run without audio hardware
# ---------------------------------------------------------------------------
class _FakeOutputStream:
    """Minimal OutputStream mock that calls the callback synchronously."""

    def __init__(self, **kwargs):
        self.callback = kwargs.get("callback")
        self.samplerate = kwargs.get("samplerate", 48000)
        self.channels = kwargs.get("channels", 2)
        self._active = False
        self.closed = False

    def start(self):
        self._active = True

    def stop(self):
        self._active = False

    def close(self):
        self._active = False
        self.closed = True

    @property
    def active(self):
        return self._active

    def pump(self, frames: int = 1024) -> np.ndarray:
        """Drive the callback once and return what it wrote."""
        buf = np.zeros((frames, self.channels), dtype=np.float32)
        if self.callback:
            self.callback(buf, frames, None, None)
        return buf


class _FakeSounddevice:
    """Module-level mock for ``sounddevice``."""

    OutputStream = _FakeOutputStream

    @staticmethod
    def query_devices(kind="output"):
        return {"default_samplerate": 48000.0, "max_output_channels": 2}


@pytest.fixture(autouse=True)
def _patch_sd(monkeypatch):
    """Patch sounddevice globally for all tests in this module."""
    import Aurik10.ui.audio_player as ap

    fake_sd = _FakeSounddevice()
    monkeypatch.setattr(ap, "sd", fake_sd)
    monkeypatch.setattr(ap, "_SD_AVAILABLE", True)
    # Reset singleton
    monkeypatch.setattr(ap, "_instance", None)
    yield


def _make_player():
    from Aurik10.ui.audio_player import StreamingAudioPlayer

    return StreamingAudioPlayer()


def _pump(player, frames=1024):
    """Pump audio through the player's callback (bypass real audio hardware)."""
    stream = player._stream
    assert stream is not None, "Player has no active stream"
    return stream.pump(frames)


# ---------------------------------------------------------------------------
# Shape / normalisation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrepareAudio:
    """Tests for _prepare() shape normalisation and resampling."""

    def test_mono_to_stereo(self):
        p = _make_player()
        p._output_sr = 48000
        mono = np.random.randn(48000).astype(np.float32)
        result = p._prepare(mono, 48000, 48000)
        assert result is not None
        assert result.ndim == 2
        assert result.shape[1] == 2
        np.testing.assert_array_equal(result[:, 0], result[:, 1])

    def test_transposed_channels(self):
        p = _make_player()
        p._output_sr = 48000
        # Shape (2, 48000) — channels first
        audio = np.random.randn(2, 48000).astype(np.float32) * 0.3
        result = p._prepare(audio, 48000, 48000)
        assert result is not None
        assert result.shape == (48000, 2)

    def test_clipping(self):
        p = _make_player()
        p._output_sr = 48000
        loud = np.full((100, 2), 5.0, dtype=np.float32)
        result = p._prepare(loud, 48000, 48000)
        assert result is not None
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_nan_handling(self):
        p = _make_player()
        p._output_sr = 48000
        audio = np.array([0.5, np.nan, np.inf, -np.inf, 0.3], dtype=np.float32)
        result = p._prepare(audio, 48000, 48000)
        assert result is not None
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_resample_cache(self):
        p = _make_player()
        p._output_sr = 48000
        audio = np.random.randn(44100).astype(np.float32) * 0.5
        r1 = p._prepare(audio, 44100, 48000)
        r2 = p._prepare(audio, 44100, 48000)
        # Same object from cache (identity)
        assert r1 is r2

    def test_cache_eviction(self):
        p = _make_player()
        p._output_sr = 48000
        p._MAX_CACHE = 2
        a1 = np.random.randn(1000).astype(np.float32)
        a2 = np.random.randn(1000).astype(np.float32)
        a3 = np.random.randn(1000).astype(np.float32)
        p._prepare(a1, 48000, 48000)
        p._prepare(a2, 48000, 48000)
        p._prepare(a3, 48000, 48000)
        # Only 2 entries remain
        assert len(p._resample_cache) == 2


# ---------------------------------------------------------------------------
# Playback lifecycle
# ---------------------------------------------------------------------------


class TestPlayback:
    """Tests for play/stop/seek lifecycle."""

    def test_play_starts_stream(self):
        p = _make_player()
        audio = np.random.randn(48000, 2).astype(np.float32) * 0.5
        ok = p.play(audio, 48000)
        assert ok is True
        assert p.is_playing is True
        assert p._stream is not None

    def test_play_produces_audio(self):
        p = _make_player()
        audio = np.ones((4800, 2), dtype=np.float32) * 0.5
        p.play(audio, 48000)
        out = _pump(p, frames=1024)
        assert out.shape == (1024, 2)
        # Should be non-zero (playing audio)
        assert np.max(np.abs(out)) > 0.1

    def test_position_advances(self):
        p = _make_player()
        audio = np.random.randn(48000, 2).astype(np.float32) * 0.3
        p.play(audio, 48000)
        _pump(p, frames=4800)
        assert p.position_frac > 0.0
        assert p.elapsed_seconds > 0.0

    def test_stop_fade_out(self):
        p = _make_player()
        audio = np.ones((48000, 2), dtype=np.float32) * 0.5
        p.play(audio, 48000)
        _pump(p, frames=1024)
        p.stop()
        # Pump through the fade-out
        out = _pump(p, frames=512)
        # After fade-out completes, player should not be playing
        # (fade is 384 samples, so 512 frames covers it)
        assert p.is_playing is False

    def test_natural_end(self):
        p = _make_player()
        audio = np.ones((500, 2), dtype=np.float32) * 0.3
        p.play(audio, 48000)
        # Pump more frames than audio length → natural end
        _pump(p, frames=1024)
        assert p.is_playing is False

    def test_seek_no_gap(self):
        p = _make_player()
        audio = np.random.randn(48000, 2).astype(np.float32) * 0.5
        p.play(audio, 48000)
        _pump(p, frames=1024)
        # Seek to 50% — should stay playing without gap
        p.seek(0.5)
        assert p.is_playing is True
        expected_pos = int(0.5 * 48000)
        assert abs(p._pos - expected_pos) < 10

    def test_start_frac(self):
        p = _make_player()
        audio = np.random.randn(48000, 2).astype(np.float32) * 0.3
        p.play(audio, 48000, start_frac=0.5)
        assert p._pos == 24000

    def test_duration_seconds(self):
        p = _make_player()
        audio = np.random.randn(48000, 2).astype(np.float32) * 0.3
        p.play(audio, 48000)
        assert abs(p.duration_seconds - 1.0) < 0.01

    def test_position_frac_idle(self):
        p = _make_player()
        assert p.position_frac == -1.0


# ---------------------------------------------------------------------------
# Gapless source switching (crossfade)
# ---------------------------------------------------------------------------


class TestGaplessSwitch:
    """Tests for gapless A/B source switching."""

    def test_switch_while_playing(self):
        p = _make_player()
        src_a = np.full((48000, 2), 0.3, dtype=np.float32)
        src_b = np.full((48000, 2), -0.3, dtype=np.float32)

        p.play(src_a, 48000)
        _pump(p, frames=1024)

        # Switch to source B — should create crossfade overlay
        p.play(src_b, 48000)
        assert p._cf_overlay is not None
        assert p.is_playing is True

        # Pump through crossfade
        out = _pump(p, frames=512)
        assert out.shape == (512, 2)
        # After crossfade, output should converge to src_b values
        # (The first samples are a mix due to crossfade)
        assert np.max(np.abs(out)) > 0.0

    def test_switch_creates_continuity(self):
        """At t=0 of crossfade, output should be close to old source value."""
        p = _make_player()
        val_a = 0.5
        val_b = -0.5
        src_a = np.full((48000, 2), val_a, dtype=np.float32)
        src_b = np.full((48000, 2), val_b, dtype=np.float32)

        p.play(src_a, 48000)
        _pump(p, frames=1024)

        p.play(src_b, 48000)
        # First sample of output should be close to val_a (continuity)
        out = _pump(p, frames=1)
        # overlay = (old - new) * 1.0 = (0.5 - (-0.5)) * 1.0 = 1.0
        # output = new + overlay = -0.5 + 1.0 = 0.5 (= old value)
        assert abs(out[0, 0] - val_a) < 0.05

    def test_no_crossfade_when_idle(self):
        p = _make_player()
        audio = np.random.randn(48000, 2).astype(np.float32) * 0.3
        p.play(audio, 48000)
        assert p._cf_overlay is None  # No previous source → no crossfade

    def test_crossfade_clipping(self):
        """Crossfade overlay must not produce samples outside [-1, 1]."""
        p = _make_player()
        src_a = np.full((48000, 2), 0.95, dtype=np.float32)
        src_b = np.full((48000, 2), -0.95, dtype=np.float32)

        p.play(src_a, 48000)
        _pump(p, frames=1024)
        p.play(src_b, 48000)

        # Pump through crossfade
        out = _pump(p, frames=512)
        assert np.max(np.abs(out)) <= 1.0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Basic thread-safety smoke tests."""

    def test_concurrent_play_stop(self):
        p = _make_player()
        audio = np.random.randn(96000, 2).astype(np.float32) * 0.3
        errors = []

        def _play_loop():
            try:
                for _ in range(20):
                    p.play(audio, 48000, start_frac=np.random.random())
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def _stop_loop():
            try:
                for _ in range(20):
                    p.stop()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=_play_loop)
        t2 = threading.Thread(target=_stop_loop)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert len(errors) == 0, f"Thread-safety error: {errors}"

    def test_concurrent_seek(self):
        p = _make_player()
        audio = np.random.randn(96000, 2).astype(np.float32) * 0.3
        p.play(audio, 48000)
        errors = []

        def _seek_loop():
            try:
                for _ in range(50):
                    p.seek(np.random.random())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_seek_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_cleans_up(self):
        p = _make_player()
        audio = np.random.randn(48000, 2).astype(np.float32) * 0.3
        p.play(audio, 48000)
        p.shutdown()
        assert p._stream is None
        assert p._buf is None
        assert p.is_playing is False
        assert len(p._resample_cache) == 0

    def test_double_shutdown_safe(self):
        p = _make_player()
        p.shutdown()
        p.shutdown()  # Should not raise


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_streaming_player_returns_same(self):
        from Aurik10.ui.audio_player import get_streaming_player

        p1 = get_streaming_player()
        p2 = get_streaming_player()
        assert p1 is p2

    def test_available_property(self):
        from Aurik10.ui.audio_player import get_streaming_player

        p = get_streaming_player()
        assert p.available is True


# ---------------------------------------------------------------------------
# On-finished callback
# ---------------------------------------------------------------------------


class TestFinishedCallback:
    def test_on_finished_called_at_natural_end(self):
        p = _make_player()
        finished = threading.Event()
        p.play(
            np.ones((200, 2), dtype=np.float32) * 0.3,
            48000,
            on_finished=finished.set,
        )
        # Pump past end of audio
        _pump(p, frames=512)
        assert finished.is_set()

    def test_on_finished_called_on_stop(self):
        p = _make_player()
        finished = threading.Event()
        p.play(
            np.ones((48000, 2), dtype=np.float32) * 0.3,
            48000,
            on_finished=finished.set,
        )
        _pump(p, frames=100)
        p.stop()
        # Pump through fade-out
        _pump(p, frames=512)
        assert finished.is_set()

    def test_on_finished_at_pos_ge_n(self):
        """Regression: _fire_finished must fire even on the pos >= n early path."""
        p = _make_player()
        finished = threading.Event()
        p.play(
            np.ones((500, 2), dtype=np.float32) * 0.3,
            48000,
            start_frac=1.0,  # Start at end → pos == n
            on_finished=finished.set,
        )
        # First pump triggers pos >= n path
        _pump(p, frames=256)
        assert finished.is_set(), "_fire_finished was not fired on pos >= n path"
        assert p.is_playing is False
