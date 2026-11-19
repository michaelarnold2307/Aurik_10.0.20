"""
Regression-Test Phase 05 (Rumble Filter) — Loudness-Drift-Guard.

Stellt sicher, dass die Rumble-Entfernung den musikalischen Pegel
NICHT katastrophal schädigt.

Bug-Hintergrund (v10.0.0):
    Der ursprüngliche Loudness-Guard hatte einen defekten Headroom-Guard:
        _headroom_05 = min(1.0, 0.95 / _peak99_05)
    → _headroom immer ≤ 1.0 → Makeup-Gain wurde NIE angewendet.
    Außerdem: Globale RMS statt Gated-RMS (§2.45a-I Verletzung),
    uniformer Gain statt Envelope-Aware (§2.45a-II Verletzung).

Testszenarien:
    1. Vinyl-Rumpeln (33 Hz + 66 Hz) mit Musik (80 Hz Kick + 500 Hz Melodie)
       → RMS-Drop darf max 2.5 dB betragen (Guard muss kompensieren)
    2. Tape-Rumpeln (25 Hz + 50 Hz) mit Modal-Musik
       → gleiche Invariante
    3. Stille-dominiertes Signal → Guard darf nicht pumpen (Envelope-Aware)
    4. Verifikation: Makeup-Gain wird tatsächlich > 1.0 angewendet bei Bedarf
"""

import numpy as np
import pytest

SR = 48_000


def _make_vinyl_rumble_signal(duration_s: float = 2.0) -> np.ndarray:
    """Realistisches Stereo-Signal: Vinyl-Rumpeln + Musik bei 0.5× Amplitude."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)

    # Musical content: kick drum 80 Hz pulsing + melody 440 Hz + noise floor
    kick = 0.25 * np.sin(2 * np.pi * 80 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
    melody = 0.15 * np.sin(2 * np.pi * 440 * t)
    music = kick + melody + 0.01 * rng.standard_normal(len(t))

    # Strong vinyl rumble: 33 Hz turntable motor + 66 Hz harmonic
    rumble = 0.35 * np.sin(2 * np.pi * 33 * t) + 0.2 * np.sin(2 * np.pi * 66 * t)

    audio = music + rumble
    # Stereo with slight difference
    stereo = np.column_stack([audio, audio * 0.95 + 0.005 * rng.standard_normal(len(t))])
    stereo = np.clip(stereo, -1.0, 1.0)
    return stereo.astype(np.float32)


def _make_tape_rumble_signal(duration_s: float = 2.0) -> np.ndarray:
    """Tape-Material mit moderatem LF-Rumpeln."""
    rng = np.random.default_rng(123)
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)

    music = 0.20 * np.sin(2 * np.pi * 220 * t) + 0.10 * np.sin(2 * np.pi * 110 * t)
    music += 0.01 * rng.standard_normal(len(t))

    rumble = 0.3 * np.sin(2 * np.pi * 25 * t) + 0.15 * np.sin(2 * np.pi * 50 * t)

    audio = music + rumble
    stereo = np.column_stack([audio, audio * 0.98])
    stereo = np.clip(stereo, -1.0, 1.0)
    return stereo.astype(np.float32)


def _make_fadeout_rumble_signal(duration_s: float = 2.0) -> np.ndarray:
    """Musik mit leisem Fade-out-Tail und starkem Rumpeln für Pumping-Regressionen."""
    rng = np.random.default_rng(7)
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)

    music = 0.18 * np.sin(2 * np.pi * 220 * t) + 0.08 * np.sin(2 * np.pi * 440 * t)
    envelope = np.ones_like(t)
    fade_start = int(1.4 * SR)
    envelope[fade_start:] = np.linspace(1.0, 0.02, len(t) - fade_start)
    music *= envelope

    rumble = 0.22 * np.sin(2 * np.pi * 26 * t) + 0.10 * np.sin(2 * np.pi * 52 * t)
    tail_noise = 0.004 * rng.standard_normal(len(t))

    audio = music + rumble + tail_noise
    stereo = np.column_stack([audio, audio * 0.985 + 0.001 * rng.standard_normal(len(t))])
    return np.clip(stereo, -1.0, 1.0).astype(np.float32)


def _gated_rms_db(audio: np.ndarray, gate_dbfs: float = -50.0) -> float:
    """Gated RMS in dBFS — identisch zur Phase-05-Implementierung."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        flat = ((arr[:, 0] + arr[:, 1]) * 0.5).ravel()
    else:
        flat = arr.ravel()
    frame_len = 2048
    n_frames = max(1, len(flat) // frame_len)
    frames = flat[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    frame_dbfs = 20.0 * np.log10(frame_rms + 1e-12)
    gate_mask = frame_dbfs > gate_dbfs
    if np.sum(gate_mask) < max(1, int(0.05 * n_frames)):
        all_rms = float(np.sqrt(np.mean(flat**2) + 1e-12))
        return float(20.0 * np.log10(all_rms + 1e-12))
    gated_rms = float(np.sqrt(np.mean(frame_rms[gate_mask] ** 2) + 1e-12))
    return float(20.0 * np.log10(gated_rms + 1e-12))


def _zone_peak_db(audio: np.ndarray, start: int, end: int) -> float:
    """99.9%-Peak in dBFS for mono/stereo segments."""
    seg = np.asarray(audio[start:end], dtype=np.float32)
    if seg.ndim == 2:
        seg = seg.ravel()
    peak = float(np.percentile(np.abs(seg), 99.9))
    return float(20.0 * np.log10(peak + 1e-12))


@pytest.mark.unit
class TestPhase05LoudnessGuard:
    """Phase 05 Rumble Filter darf den musikalischen Pegel nicht zerstören."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from backend.core.phases.phase_05_rumble_filter import RumbleFilterPhase

        self.phase = RumbleFilterPhase(sample_rate=SR)

    def test_vinyl_rumble_rms_drop_within_limit(self):
        """Vinyl-Rumpeln entfernen darf max 2.5 dB Gated-RMS-Drop verursachen."""
        audio = _make_vinyl_rumble_signal(duration_s=2.0)
        rms_before = _gated_rms_db(audio)

        result = self.phase.process(audio.copy(), material_type="vinyl")
        assert result.success, f"Phase 05 fehlgeschlagen: {result.warnings}"

        rms_after = _gated_rms_db(result.audio)
        rms_drop = rms_after - rms_before  # Negative = Pegelverlust

        assert rms_drop > -2.5, (
            f"KATASTROPHALER Pegelverlust in Phase 05: {rms_drop:.2f} dB (Limit: -2.5 dB). Loudness-Guard defekt!"
        )

    def test_tape_rumble_rms_drop_within_limit(self):
        """Tape-Rumpeln entfernen darf max 2.5 dB Gated-RMS-Drop verursachen."""
        audio = _make_tape_rumble_signal(duration_s=2.0)
        rms_before = _gated_rms_db(audio)

        result = self.phase.process(audio.copy(), material_type="tape")
        assert result.success

        rms_after = _gated_rms_db(result.audio)
        rms_drop = rms_after - rms_before

        assert rms_drop > -2.5, (
            f"KATASTROPHALER Pegelverlust in Phase 05 (tape): {rms_drop:.2f} dB "
            f"(Limit: -2.5 dB). Loudness-Guard defekt!"
        )

    def test_makeup_gain_actually_applied(self):
        """Der Makeup-Gain muss > 1.0 sein wenn RMS-Drop die Schwelle reißt."""
        audio = _make_vinyl_rumble_signal(duration_s=2.0)

        result = self.phase.process(audio.copy(), material_type="vinyl")
        assert result.success

        # Wenn Rumble entfernt wurde, muss der Guard aktiv geworden sein
        if result.modifications.get("rumble_filtered"):
            makeup_db = result.metadata.get("loudness_makeup_db", 0.0)
            rms_drop = result.metadata.get("rms_drop_db", 0.0)
            # Bei starkem Rumble (Energie > 30%) MUSS der Guard kompensiert haben
            if rms_drop < -1.5:
                assert makeup_db > 0.0, (
                    f"Loudness-Guard hat NICHT kompensiert obwohl RMS-Drop={rms_drop:.2f} dB. "
                    f"Makeup war {makeup_db:.2f} dB — Guard ist defekt!"
                )

    def test_silence_frames_not_amplified(self):
        """Stille-Frames dürfen vom Guard nicht verstärkt werden (§2.45a-II)."""
        rng = np.random.default_rng(99)
        t = np.linspace(0, 2.0, SR * 2, endpoint=False)

        # 1s Musik + 1s Stille
        music = 0.3 * np.sin(2 * np.pi * 440 * t[:SR])
        rumble_music = 0.4 * np.sin(2 * np.pi * 33 * t[:SR])
        music_part = music + rumble_music

        silence_part = 0.0001 * rng.standard_normal(SR)  # Rauschboden

        audio_mono = np.concatenate([music_part, silence_part])
        audio = np.column_stack([audio_mono, audio_mono * 0.98]).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        # RMS des Stille-Bereichs vorher
        silence_rms_before = float(np.sqrt(np.mean(audio[SR:, :] ** 2) + 1e-12))

        result = self.phase.process(audio.copy(), material_type="vinyl")
        assert result.success

        # RMS des Stille-Bereichs nachher — darf NICHT signifikant steigen
        silence_rms_after = float(np.sqrt(np.mean(result.audio[SR:, :] ** 2) + 1e-12))
        amplification_ratio = silence_rms_after / max(silence_rms_before, 1e-12)

        assert amplification_ratio < 3.0, (
            f"Stille-Bereich wurde um Faktor {amplification_ratio:.1f}x verstärkt! "
            f"Envelope-Aware Gain fehlt (§2.45a-II Verletzung)."
        )

    def test_cd_digital_minimal_intervention(self):
        """CD-Material ohne Rumble: Phase darf Pegel nicht verändern."""
        rng = np.random.default_rng(77)
        t = np.linspace(0, 1.0, SR, endpoint=False)

        # Typisches CD-Signal: sauber, nur Musik, kein Rumble
        audio_mono = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.01 * rng.standard_normal(SR)
        audio = np.column_stack([audio_mono, audio_mono * 0.99]).astype(np.float32)

        rms_before = _gated_rms_db(audio)
        result = self.phase.process(audio.copy(), material_type="cd_digital")

        rms_after = _gated_rms_db(result.audio)
        rms_change = abs(rms_after - rms_before)

        assert rms_change < 0.5, (
            f"CD-Material ohne Rumble wurde um {rms_change:.2f} dB verändert. Expected: < 0.5 dB Veränderung."
        )

    def test_fadeout_tail_does_not_create_spurious_transients(self):
        """Leiser Fade-out-Tail darf den HPF-BYPASS nicht rhythmisch an/aus schalten."""
        audio = _make_fadeout_rumble_signal(duration_s=2.0)

        transient_mask = self.phase._detect_transients_professional(audio, sensitivity=0.8)

        tail = transient_mask[int(1.7 * SR) :]
        tail_ratio = float(np.mean(tail)) if len(tail) else 0.0

        assert tail_ratio < 0.15, (
            f"Quiet fade-out tail flagged as transient too often: {tail_ratio:.2%}. "
            "Phase 05 would alternate HPF bypass/filter and pump the outro."
        )

    def test_intro_outro_peak_no_explosion_after_rumble_filter(self):
        """Regression: Intro/Outro dürfen durch Phase 05 nicht als Peak explodieren."""
        audio = _make_fadeout_rumble_signal(duration_s=4.0)
        n_intro = int(0.45 * SR)
        n_outro_start = len(audio) - int(0.45 * SR)

        in_intro_peak = _zone_peak_db(audio, 0, n_intro)
        in_outro_peak = _zone_peak_db(audio, n_outro_start, len(audio))

        result = self.phase.process(audio.copy(), material_type="vinyl")
        assert result.success

        out_intro_peak = _zone_peak_db(result.audio, 0, n_intro)
        out_outro_peak = _zone_peak_db(result.audio, n_outro_start, len(result.audio))

        intro_boost = out_intro_peak - in_intro_peak
        outro_boost = out_outro_peak - in_outro_peak

        assert intro_boost <= 3.0, f"Intro peak explosion after phase_05: +{intro_boost:.2f} dB (limit +3.0 dB)"
        assert outro_boost <= 3.0, f"Outro peak explosion after phase_05: +{outro_boost:.2f} dB (limit +3.0 dB)"
