import numpy as np


class TargetSoundMatcher:
    """
    SOTA Target Sound Matching (Studio-Algorithmus):
    - Passt das Spektrum an ein Referenzsignal an (z.B. modernes Studio-Master)
    """

    def __init__(self, reference_audio: np.ndarray | None = None):
        self.reference_audio = reference_audio

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.reference_audio is None:
            return audio
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float64))
        ref = np.nan_to_num(np.asarray(self.reference_audio, dtype=np.float64))
        # Spektralanalyse
        S = np.abs(np.fft.rfft(audio))
        S_ref = np.abs(np.fft.rfft(ref))
        # Matching-Kurve with clamping to prevent extreme boosts
        match_curve = S_ref / (S + 1e-8)
        match_curve = np.clip(match_curve, 0.1, 10.0)  # max ±20 dB
        # Anwenden im Frequenzbereich
        audio_fft = np.fft.rfft(audio)
        matched_fft = audio_fft * match_curve
        matched = np.fft.irfft(matched_fft)
        matched = np.nan_to_num(matched, nan=0.0, posinf=0.0, neginf=0.0)
        matched = np.clip(matched[: len(audio)], -1.0, 1.0)
        return np.asarray(matched)  # type: ignore[no-any-return]
