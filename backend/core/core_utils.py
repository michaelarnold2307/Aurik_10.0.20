import numpy as np
from numpy.fft import irfft, rfft


def compute_rms(audio: np.ndarray) -> float:
    """Berechnet den RMS-Wert eines Audiosignals."""
    return float(np.sqrt(np.mean(audio**2)))


def compute_loudness(audio: np.ndarray) -> float:
    """Berechnet eine einfache Lautheitsschätzung (LUFS-Approximation)."""
    rms = compute_rms(audio)
    return float(20 * np.log10(rms + 1e-8))


def safe_peak_amplitude(audio: np.ndarray) -> float:
    """Gibt robust peak amplitude using 99.9th percentile zurück.

    A single impulse artifact (crackle, click) must not block gain
    normalization of the entire signal (§DSP Peak-Guard).
    """
    if audio.size == 0:
        return 0.0
    return float(np.percentile(np.abs(audio), 99.9))


def fft_autocorr(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """FFT-based autocorrelation — O(N log N) instead of O(N²).

    Returns the one-sided (non-negative lags) autocorrelation.
    If *max_lag* is given, only lags 0..max_lag are returned.
    """
    n = len(x)
    # Next power-of-two length for efficient FFT (zero-padded to avoid circular effects)
    fft_len = 1
    while fft_len < 2 * n:
        fft_len <<= 1
    X = rfft(x, n=fft_len)
    ac_full = irfft(X * np.conj(X), n=fft_len)[:n]
    if max_lag is not None:
        ac_full = ac_full[: max_lag + 1]
    return np.asarray(ac_full, dtype=np.float64)  # type: ignore[no-any-return]


def fft_crosscorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """FFT-based cross-correlation — O(N log N) instead of O(N²).

    Returns the full cross-correlation array (length 2N-1), with the
    zero-lag element at index len(a)-1 (same layout as
    ``np.correlate(a, b, mode='full')``).
    """
    n = max(len(a), len(b))
    fft_len = 1
    while fft_len < 2 * n:
        fft_len <<= 1
    A = rfft(a, n=fft_len)
    B = rfft(b, n=fft_len)
    cc = irfft(A * np.conj(B), n=fft_len)
    # Rearrange to match np.correlate(a, b, mode='full') layout
    out = np.empty(len(a) + len(b) - 1, dtype=np.float64)
    shift = len(b) - 1
    out[shift:] = cc[: len(a)]
    out[:shift] = cc[fft_len - shift : fft_len]
    return out  # type: ignore[no-any-return]


def audio_stats(audio: np.ndarray) -> dict:
    """Gibt zentrale Statistiken (Peak, RMS, Loudness) zurück."""
    return {
        "peak": safe_peak_amplitude(audio),
        "rms": compute_rms(audio),
        "loudness": compute_loudness(audio),
    }


def log_message(msg: str, logfile: str = "aurik6.log"):
    """Schreibt eine Lognachricht in eine Datei."""
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
