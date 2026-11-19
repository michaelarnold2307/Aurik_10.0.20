"""
Detects and reduces plosives (P, B, T, K sounds) that cause low-frequency transients.

🎵 AURIK MUSIK-RESTAURATION PHILOSOPHIE:
================================================================================
WICHTIG: Dieses Modul ist NUR für Speech/Podcast/Voice-Over gedacht!

Für AURIK's Kernkompetenz (Internationale Musik mit Gesang):
- Plosives sind Teil der authentischen Performance
- Gesang (VOCALS): NICHT behandeln (natürliche Expression)
- Nur Speech (SPEECH): Behandeln (technischer Defekt bei Nahbesprechung)

Philosophie: Authentizität & Natürlichkeit > Technische Perfektion
================================================================================

Week 10 Integration: P1 Speech Defect Treatment (NOT for music vocals!)
"""

import logging

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class AutomaticPlosiveRemover:
    """
    Erkennt and removes plosive artifacts in SPEECH recordings only.

    🎭 MUSIK-RESTAURATION: NOT for singing vocals (plosives are authentic performance)

    Plosives are characterized by:
    - Low-frequency transients (20-200 Hz)
    - Short duration (10-50ms)
    - High amplitude in bass range
    - Caused by P, B, T, K consonants in close-mic situations

    Treatment (Speech only):
    - Dynamic high-pass filter during plosive events
    - Gentle transient gating
    - Preserves natural speech character

    Usage Context:
    - ✅ Podcasts, Voice-Overs, Interviews (InstrumentType.SPEECH)
    - ❌ Music with Vocals (InstrumentType.VOCALS) - Plosives are authentic!
    """

    def __init__(
        self,
        sr: int = 44100,
        threshold_db: float = -30.0,
        reduction_db: float = 15.0,
        attack_ms: float = 1.0,
        release_ms: float = 10.0,
        highpass_cutoff: float = 80.0,
    ):
        """
        Initialisiert plosive remover.

        Args:
            sr: Sample rate
            threshold_db: Detection threshold (dB below peak)
            reduction_db: Amount of reduction to apply (dB)
            attack_ms: Attack time for processing
            release_ms: Release time for processing
            highpass_cutoff: High-pass filter cutoff during plosives (Hz)
        """
        self.sr = sr
        self.threshold_db = threshold_db
        self.reduction_db = reduction_db
        self.attack_samples = int(attack_ms * sr / 1000)
        self.release_samples = int(release_ms * sr / 1000)
        self.highpass_cutoff = highpass_cutoff

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Verarbeitet audio to remove plosives.

        Args:
            audio: Input audio (mono)

        Returns:
            Processed audio with reduced plosives
        """
        if audio.ndim > 1:
            raise ValueError("Plosive remover expects mono audio")

        assert self.sr == 48000, f"Sample rate must be 48000 Hz, got {self.sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Detect plosive events
        plosive_regions = self._detect_plosives(audio)

        if len(plosive_regions) == 0:
            return audio

        # Process each plosive region
        audio_out = audio.copy()

        for start, end in plosive_regions:
            # Apply dynamic high-pass filter + attenuation
            audio_out = self._process_plosive_region(audio_out, start, end)

        # NaN/Inf-Guard and clipping
        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        return np.asarray(audio_out)  # type: ignore[no-any-return]

    def _detect_plosives(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """
        Erkennt plosive events in audio.

        Returns:
            List of (start_sample, end_sample) tuples
        """
        # Low-pass filter to isolate bass frequencies (20-200 Hz)
        sos = signal.butter(4, 200, "lp", fs=self.sr, output="sos")
        audio_lf = signal.sosfilt(sos, audio)

        # Envelope detection (RMS in short windows)
        window_size = int(0.005 * self.sr)  # 5ms windows
        hop_size = window_size // 2

        _envelope_list: list[float] = []
        for i in range(0, len(audio_lf) - window_size, hop_size):
            window = audio_lf[i : i + window_size]
            rms = float(np.sqrt(np.mean(window**2)))
            _envelope_list.append(rms)

        envelope: np.ndarray = np.array(_envelope_list)

        # Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)

        # Threshold: peak detection
        threshold = np.max(envelope_db) + self.threshold_db

        # Find plosive events (above threshold)
        plosive_frames = envelope_db > threshold

        # Duration filtering (10-50ms)
        min_duration_frames = int(0.010 * self.sr / hop_size)  # 10ms
        max_duration_frames = int(0.050 * self.sr / hop_size)  # 50ms

        # Find contiguous regions
        plosive_regions = []
        in_plosive = False
        start_frame = 0

        for i, is_plosive in enumerate(plosive_frames):
            if is_plosive and not in_plosive:
                start_frame = i
                in_plosive = True
            elif not is_plosive and in_plosive:
                duration_frames = i - start_frame
                if min_duration_frames <= duration_frames <= max_duration_frames:
                    # Convert frame indices to sample indices
                    start_sample = max(0, start_frame * hop_size - self.attack_samples)
                    end_sample = min(len(audio), i * hop_size + self.release_samples)
                    plosive_regions.append((start_sample, end_sample))
                in_plosive = False

        return plosive_regions

    def _process_plosive_region(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        Verarbeitet a single plosive region.

        Strategy:
        1. Apply high-pass filter to remove low-frequency transient
        2. Apply gentle attenuation envelope
        3. Smooth transition (crossfade)

        Args:
            audio: Full audio signal
            start: Start sample of plosive
            end: End sample of plosive

        Returns:
            Processed audio
        """
        # Extract plosive segment
        segment = audio[start:end].copy()

        # Apply high-pass filter (remove low-frequency transient)
        sos = signal.butter(4, self.highpass_cutoff, "hp", fs=self.sr, output="sos")
        segment_filtered = signal.sosfilt(sos, segment)

        # Apply attenuation envelope (gentle reduction)
        reduction_linear = 10 ** (-self.reduction_db / 20)

        # Create smooth envelope (attack/release)
        envelope = np.ones(len(segment))

        # Attack phase
        attack_len = min(self.attack_samples, len(segment) // 3)
        envelope[:attack_len] = np.linspace(1.0, reduction_linear, attack_len)

        # Sustain phase
        sustain_start = attack_len
        sustain_end = max(sustain_start, len(segment) - self.release_samples)
        envelope[sustain_start:sustain_end] = reduction_linear

        # Release phase
        release_len = len(segment) - sustain_end
        if release_len > 0:
            envelope[sustain_end:] = np.linspace(reduction_linear, 1.0, release_len)

        # Apply envelope to filtered segment
        segment_processed = segment_filtered * envelope

        # Crossfade back into original (smooth transition)
        crossfade_len = min(self.attack_samples, len(segment) // 4)

        if crossfade_len > 0:
            # Fade in processed at start
            fade_in = np.linspace(0, 1, crossfade_len)
            segment_processed[:crossfade_len] = (
                segment[:crossfade_len] * (1 - fade_in) + segment_processed[:crossfade_len] * fade_in
            )

            # Fade out processed at end
            fade_out = np.linspace(1, 0, crossfade_len)
            segment_processed[-crossfade_len:] = segment_processed[-crossfade_len:] * fade_out + segment[
                -crossfade_len:
            ] * (1 - fade_out)

        # Write back to audio
        audio_out = audio.copy()
        audio_out[start:end] = segment_processed

        return audio_out

    def detect_only(self, audio: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
        """
        Erkennt plosives without processing.

        Args:
            audio: Input audio (mono)

        Returns:
            Tuple of (plosive_count, plosive_regions)
        """
        plosive_regions = self._detect_plosives(audio)
        return len(plosive_regions), plosive_regions


# Example usage
if __name__ == "__main__":
    import soundfile as sf

    from backend.file_import import load_audio_file

    # Load test audio (speech/podcast with plosives)
    _res = load_audio_file("test_speech_plosives.wav")
    assert _res is not None, "load_audio_file returned None"
    _audio, _sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])

    # Convert to mono if stereo
    if _audio.ndim > 1:
        _audio = np.mean(_audio, axis=0)

    # Initialize remover
    _remover = AutomaticPlosiveRemover(sr=_sr, reduction_db=15.0)

    # Detect plosives
    _count, _regions = _remover.detect_only(_audio)
    logger.info("%d Plosiv-Ereignisse erkannt", _count)

    for _i, (_start, _end) in enumerate(_regions):
        _duration_ms = (_end - _start) / _sr * 1000
        _time_s = _start / _sr
        logger.info("  Plosive %d: %.2fs, duration: %.1fms", _i + 1, _time_s, _duration_ms)

    # Process audio
    _audio_processed = _remover.process(_audio)

    # Save output
    sf.write("test_speech_processed.wav", _audio_processed, _sr)
    logger.info("Verarbeitetes Audio gespeichert")
