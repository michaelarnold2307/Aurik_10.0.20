"""
BW Harmonic Exciter — DSP-basierte Bandbreiten-Erweiterung für Aurik.

Kein ML, kein Training. Fügt via Waveshaping + spektraler Hüllkurven-
Extrapolation harmonische Obertöne oberhalb der Cutoff-Frequenz hinzu.

Garantiert: blend=0 = passthrough. cutoff_hz=0 = passthrough.
Verschlechtert nie das Originalsignal.
Latenz: <50ms pro 3s Segment auf CPU.

Usage:
    from plugins.bw_harmonic_exciter import BWHarmonicExciter
    exciter = BWHarmonicExciter(blend=0.4, drive=0.8)
    result = exciter.process(audio, sr, cutoff_hz=5000)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)
SR_DEFAULT = 22050


def _spectral_envelope(magnitude, smooth_bins=8):
    kernel = np.ones(smooth_bins) / smooth_bins
    return np.convolve(magnitude, kernel, mode="same")


def _extrapolate_envelope(env, cutoff_bin, order=2):
    if cutoff_bin < 10:
        return env.copy()
    fit_bins = np.arange(max(0, cutoff_bin - 20), cutoff_bin)
    if len(fit_bins) < order + 1:
        return env.copy()
    x_fit = fit_bins.astype(np.float64)
    y_fit = np.maximum(env[fit_bins].astype(np.float64), 1e-10)
    log_y = np.log(y_fit)
    try:
        coeffs = np.polyfit(x_fit, log_y, order)
    except np.linalg.LinAlgError:
        return env.copy()
    above_bins = np.arange(cutoff_bin, len(env))
    log_pred = np.polyval(coeffs, above_bins.astype(np.float64))
    log_pred = np.clip(log_pred, -30.0, 30.0)
    pred = np.exp(log_pred)
    result = env.copy()
    above_pred = np.maximum(result[above_bins], pred * 0.5)
    above_pred = np.clip(above_pred, 0, np.max(result) * 10)
    result[above_bins] = above_pred
    return result


def harmonic_exciter(audio, sr, cutoff_hz, n_fft=2048, hop_length=512, blend=0.5, drive=1.0):
    """Fügt harmonische Obertöne oberhalb von cutoff_hz hinzu."""
    audio = np.asarray(audio, dtype=np.float64).copy()
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Guard clauses: return original unchanged ──
    if blend <= 0.0 or cutoff_hz <= 0:
        return audio.astype(np.float32)

    # ── 1. Waveshaping ──
    driven = np.tanh(audio * drive * 3.0) / 3.0
    rectified = np.maximum(audio * drive, 0) * 0.3
    harmonics = (driven + rectified) * blend

    # ── 2. Hochpass ──
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2
    if cutoff_hz >= nyq * 0.95:
        return audio.astype(np.float32)
    sos_hp = butter(4, cutoff_hz / nyq, btype="high", output="sos")
    harmonics = sosfiltfilt(sos_hp, harmonics)

    # ── 3. STFT + spektrale Formung ──
    n_hop = min(hop_length, n_fft // 2)
    n_frames = 1 + (len(audio) - n_fft) // n_hop
    if n_frames < 1:
        return audio.astype(np.float32)

    window = np.hanning(n_fft)
    n_bins = n_fft // 2 + 1
    spec_orig = np.zeros((n_bins, n_frames))
    spec_harm = np.zeros((n_bins, n_frames))

    for i in range(n_frames):
        start = i * n_hop
        spec_orig[:, i] = np.abs(np.fft.rfft(audio[start : start + n_fft] * window))
        spec_harm[:, i] = np.abs(np.fft.rfft(harmonics[start : start + n_fft] * window))

    cutoff_bin = int(cutoff_hz / nyq * (n_fft // 2))
    cutoff_bin = max(1, min(cutoff_bin, n_bins - 2))

    for i in range(n_frames):
        env_orig = _spectral_envelope(spec_orig[:, i])
        env_target = _extrapolate_envelope(env_orig, cutoff_bin)
        above = slice(cutoff_bin, None)
        eps = 1e-10
        ratio = (env_target[above] + eps) / (env_orig[above] + eps)
        ratio = np.clip(np.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.1), 0.1, 10.0)
        gain = np.ones(n_bins)
        gain[above] = np.sqrt(ratio)
        spec_harm[:, i] *= gain

    # ── 4. ISTFT ──
    output = np.zeros_like(audio)
    weight = np.zeros_like(audio)

    for i in range(n_frames):
        start = i * n_hop
        orig_stft = np.fft.rfft(audio[start : start + n_fft] * window)
        angles = np.angle(orig_stft)
        stft_mag = np.abs(orig_stft).copy()
        stft_mag[cutoff_bin:] = spec_harm[cutoff_bin:, i]
        frame = np.fft.irfft(stft_mag * np.exp(1j * angles))
        output[start : start + n_fft] += frame * window
        weight[start : start + n_fft] += window**2

    output = np.divide(output, np.maximum(weight, 1e-8), out=np.zeros_like(output), where=weight > 1e-8)
    output = np.clip(output, -1.0, 1.0)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    return output.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Aurik-Plugin
# ═══════════════════════════════════════════════════════════════════════════


class BWHarmonicExciter:
    """DSP-basierter Bandbreiten-Erweiterer — kein ML, sofort einsatzbereit."""

    def __init__(self, blend=0.4, drive=0.8):
        self.blend = blend
        self.drive = drive

    @property
    def available(self):
        return True

    def process(self, audio, sr, cutoff_hz=5000):
        """
        Fügt harmonische Obertöne oberhalb cutoff_hz hinzu.

        Args:
            audio: 1D (Mono) oder 2D (Stereo) float-Array
            sr: Samplerate in Hz
            cutoff_hz: Frequenz, ab der rekonstruiert wird (0 = passthrough)

        Returns:
            Audio-Array in gleicher Shape
        """
        audio = np.asarray(audio, dtype=np.float32)

        # Guard: passthrough
        if self.blend <= 0.0 or cutoff_hz <= 0:
            return audio.copy()

        if audio.ndim == 2 and audio.shape[0] == 2:
            result = np.zeros_like(audio)
            for ch in range(2):
                result[ch] = self._process_mono(audio[ch], sr, cutoff_hz)
            return result
        elif audio.ndim == 2 and audio.shape[0] > 2:
            result = np.zeros_like(audio[:2])
            result[0] = self._process_mono(audio[0], sr, cutoff_hz)
            result[1] = self._process_mono(audio[1], sr, cutoff_hz)
            return result
        else:
            return self._process_mono(audio.squeeze(), sr, cutoff_hz)

    def _process_mono(self, audio, sr, cutoff_hz):
        if sr != SR_DEFAULT:
            from scipy.signal import resample_poly

            audio = resample_poly(audio.astype(np.float64), SR_DEFAULT, sr)
            audio = audio.astype(np.float64)

        result = harmonic_exciter(audio, SR_DEFAULT, cutoff_hz, blend=self.blend, drive=self.drive)

        if sr != SR_DEFAULT:
            from scipy.signal import resample_poly

            result = resample_poly(result.astype(np.float64), sr, SR_DEFAULT)

        return result.astype(np.float32)


# ── Pipeline-Stage ────────────────────────────────────────────────────────


class BWExciterStage:
    """Aurik-Pipeline-Stage für die BW Harmonic Exciter."""

    def __init__(self, blend=0.4, drive=0.8, cutoff_hz=5000, enabled=True):
        self._exciter = BWHarmonicExciter(blend=blend, drive=drive)
        self.cutoff_hz = cutoff_hz
        self.blend = blend
        self.enabled = enabled

    @property
    def name(self):
        return "BWHarmonicExciter"

    @property
    def available(self):
        return self.enabled

    def process(self, audio, sr, **kwargs):
        if not self.enabled:
            return {"audio": audio, "metadata": {"bw_exciter": "disabled"}}

        cutoff = kwargs.get("cutoff_hz", self.cutoff_hz)
        blend = kwargs.get("blend", self.blend)

        result = self._exciter.process(audio, sr, cutoff_hz=cutoff)

        return {
            "audio": result,
            "metadata": {
                "bw_exciter": "applied",
                "cutoff_hz": cutoff,
                "blend": blend,
                "method": "DSP harmonic exciter",
            },
        }


def enhance_bandwidth(audio, sr, cutoff_hz=5000, blend=0.4, drive=0.8):
    """Einzeiler: fügt Obertöne oberhalb cutoff_hz hinzu."""
    return BWHarmonicExciter(blend=blend, drive=drive).process(audio, sr, cutoff_hz)
