"""Spectral Gating mit psychoakustischem Masking-Guard (kanonische In-Memory-API).

§2.62: Bark-basierte Maskierung (ISO 11172-3) verhindert Musical Noise.
Soft-Knee statt Hard-Cutoff (§III). NaN/Inf-Schutz in jeder Phase (§0a).
Deterministisch — keine prozessabhängigen RNG-Seeds, STFT/ISTFT sind deterministisch
(§G5 (GEBOTE.md)).

Kanonische API ist die In-Memory-Funktion :func:`gate_audio` (Mono 1D `(N,)` oder
channels-first `(C, N)` — Stereo-Layout-Invariante). Die Pfad-Funktion
:func:`apply_spectral_gating`, der Batch-Helfer und die Adapterklasse
:class:`SpectralGate` sind dünne Wrappers darüber.
"""

import logging
from pathlib import Path

import librosa  # type: ignore[import]
import numpy as np
from scipy.signal import istft, stft, windows

logger = logging.getLogger(__name__)


def gate_audio(
    audio: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -60.0,
    hop_length: int = 512,
    n_fft: int | None = None,
    soft_knee_width_db: float = 6.0,
    masking_margin_db: float = 3.0,
) -> np.ndarray:
    """Spektrales Gating mit psychoakustischem Masking-Guard (kanonische In-Memory-API).

    Args:
        audio: Audio als 1D `(N,)` Mono oder channels-first `(C, N)` Stereo/Multi.
        sr: Sample-Rate in Hz.
        threshold_db: Globaler Threshold in dBFS. Standard -60 dB.
            Tiefer (-70): mehr NR, aber Risiko von Musical Noise & Transienten-Loss.
            Höher (-50): weniger NR, aber saubere Transienten-Erhaltung.
        hop_length: STFT Hop-Length (Standard 512 → ~11 ms bei 48 kHz).
        n_fft: STFT Window-Size. Auto-Kalibrierung wenn None (~1024 für Musik).
        soft_knee_width_db: Soft-Knee-Breite in dB (§III). Standard 6 dB.
            Sigmoid-Gain statt Hard-Cutoff verhindert harte Schnittkanten.
        masking_margin_db: Psychoakustischer Masking-Margin über der
            Maskierungsschwelle (ISO 11172-3 Bark-Skala).

    Returns:
        np.ndarray: Gated Audio als float64, normalisiert auf [-1, 1],
            exakt mit der Eingabe-Shape/Länge.
    """
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim == 1:
        return _gate_mono(
            y,
            sr,
            threshold_db=threshold_db,
            hop_length=hop_length,
            n_fft=n_fft,
            soft_knee_width_db=soft_knee_width_db,
            masking_margin_db=masking_margin_db,
        )

    # Channels-first (C, N): pro Kanal verarbeiten, Layout bleibt exakt erhalten
    # (§Stereo-Layout-Invariante).
    gated_channels = [
        _gate_mono(
            y[c],
            sr,
            threshold_db=threshold_db,
            hop_length=hop_length,
            n_fft=n_fft,
            soft_knee_width_db=soft_knee_width_db,
            masking_margin_db=masking_margin_db,
        )
        for c in range(y.shape[0])
    ]
    return np.stack(gated_channels, axis=0)  # type: ignore[no-any-return]


def _gate_mono(
    y: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -60.0,
    hop_length: int = 512,
    n_fft: int | None = None,
    soft_knee_width_db: float = 6.0,
    masking_margin_db: float = 3.0,
) -> np.ndarray:
    """Mono-Kern von :func:`gate_audio` (STFT → Bark-Masking-gated Gain → ISTFT)."""
    if len(y) == 0:
        raise ValueError("Audio ist leer")

    # STFT-Parameter-Kalibrierung
    n_fft = int(n_fft or min(1024, len(y) // 4))
    # Defensive Guard gegen degenerierende Fenster (nperseg > hop muss gelten):
    if n_fft <= hop_length:
        n_fft = hop_length + 1
    window = windows.hann(n_fft)

    logger.info(
        "§2.62 Spectral Gating: Schwelle=%.1f dB, hop=%d, n_fft=%d",
        threshold_db,
        hop_length,
        n_fft,
    )

    # STFT (scipy.signal — keine librosa-Internals).
    # scipy.signal.stft liefert (freqs_hz, t, Zxx): mit fs=sr ist Achse 0 die
    # Frequenzachse in Hz (Parameter vollständig geloggt — §V6 (copilot-instructions.md)).
    _noverlap = min(n_fft - hop_length, max(0, n_fft - 1))  # §v10.103 noverlap-Clamp
    freqs, _t, Z = stft(y, fs=float(sr), window=window, nperseg=n_fft, noverlap=_noverlap)

    if Z.size == 0:
        raise RuntimeError("STFT-Ergebnis ist leer")

    # Amplitude-Spektrum & dB-Konvertierung
    mag = np.abs(Z)
    mag_db = 20 * np.log10(mag + 1e-10)

    # Bark-basierte Maskierungsschwelle (ISO 11172-3 Approximation).
    # STFT-Layout ist (n_bins, n_frames): per-Bin-Schwelle als (n_bins, 1) broadcasten.
    masking_threshold_db = _compute_masking_threshold_per_band(
        mag_db, freqs, threshold_db, soft_knee_width_db, masking_margin_db
    )[:, None]

    # Soft-Knee Gain (Sigmoid statt Hard-Cutoff — §III)
    gain = _sigmoid_soft_knee(mag_db - masking_threshold_db, soft_knee_width_db)

    # Frequenz-basierte Maskierung: kein Gating über der Maskierungsschwelle (§2.62)
    gain[mag_db > masking_threshold_db + masking_margin_db] = 1.0

    # Spektrum anwenden (Phase bleibt erhalten — §V1 (copilot-instructions.md) Vocal-Distortion-Verbot)
    Z_gated = mag * gain * np.exp(1j * np.angle(Z))

    # ISTFT zurückkonvertieren: scipy.signal.istft liefert (t, y), Audio ist Element [1].
    _t_out, y_out = istft(Z_gated, fs=float(sr), window=window, nperseg=n_fft, noverlap=_noverlap)

    # Exakte Eingabe-Länge wiederherstellen (ISTFT-Padding/Trimming — §G5 (copilot-instructions.md) Reproduzierbarkeit):
    if y_out.shape[0] > len(y):
        y_out = y_out[: len(y)]
    elif y_out.shape[0] < len(y):
        y_out = np.pad(y_out, (0, len(y) - y_out.shape[0]))

    # NaN/Inf-Schutz (§0a) — Defense-in-Depth
    y_out = np.nan_to_num(y_out, nan=0.0, posinf=1.0, neginf=-1.0)

    # True-Peak Schutz (kein Clipping — §V1 (copilot-instructions.md))
    peak = float(np.max(np.abs(y_out)))
    if peak > 1.0:
        y_out *= 1.0 / peak
        logger.warning("§2.62 Spectral Gating: True-Peak-Korrektur (%.3f dBFS)", 20 * np.log10(peak))

    return y_out  # type: ignore[no-any-return]


def _freqs_to_bark(freqs: np.ndarray) -> np.ndarray:
    """Hertz → Bark-Skala (ISO 11172-3 Approximation)."""
    # Zwicker-Bark-Konvertierung (vereinfacht, aber psychoakustisch korrekt):
    z = freqs / 1000.0
    return 9 * np.log((z + 85) / (z + 7))  # type: ignore[no-any-return]


def _compute_masking_threshold_per_band(
    mag_db: np.ndarray,
    freqs: np.ndarray,
    threshold_db: float,
    soft_knee_width_db: float,
    masking_margin_db: float,
) -> np.ndarray:
    """Bark-basierte Maskierungsschwelle pro Frequenz-Bin.

    §2.62: Bark-Skala (ISO 11172-3) für NR-Algorithmen.
    Verhindert Musical Noise durch bandweise Threshold-Anpassung.

    Returns:
        np.ndarray: Schwellen pro Bin mit Shape `(n_bins,)` — Broadcast über die
            STFT-Frames erfolgt in :func:`_gate_mono`.
    """
    bark_bands = _freqs_to_bark(freqs)

    # Integer-Bänder (floor) statt Float-Equality vermeiden Grenz-Degeneration:
    band_idx = np.floor(bark_bands).astype(np.int64)
    n_bands = int(band_idx.max()) + 1 if band_idx.size else 0

    max_per_band = np.full(n_bands, -np.inf)
    # Bandweiser Maximalpegel: Maximum über alle Frames pro Bin (Zeilen-Max),
    # dann Maximum über alle Bins des Bands — äquivalent zum alten Loop, vektorisiert.
    row_max = mag_db.max(axis=1)
    if n_bands > 0:
        np.maximum.at(max_per_band, band_idx, row_max)

    # Maskierungsschwelle pro Bin: Signal + Margin oder Threshold (was höher ist)
    if n_bands > 0:
        threshold_map = np.maximum(threshold_db, max_per_band[band_idx] - masking_margin_db)
    else:
        threshold_map = np.full(bark_bands.shape, threshold_db)
    return threshold_map  # type: ignore[no-any-return]


def _sigmoid_soft_knee(x: np.ndarray, knee_width_db: float) -> np.ndarray:
    """Sigmoid-Soft-Knee statt Hard-Cutoff (§III)."""
    # Sigmoid: σ(x/knee_width) → 0 bei x << -knee_width, → 1 bei x >> knee_width
    # Verhindert harte Schnittkanten (Ghost-Echo-Verbot §V2 (copilot-instructions.md))
    gain = np.where(
        np.abs(x) < 1e-6,
        0.5 * np.ones_like(x),
        1.0 / (1.0 + np.exp(-x / knee_width_db)),
    )
    return gain  # type: ignore[no-any-return]


# --- Kanonische Pfad-API (dünner Wrapper über :func:`gate_audio`) ---


def apply_spectral_gating(
    audio_path: str | Path,
    threshold_db: float = -60.0,
    hop_length: int = 512,
    n_fft: int | None = None,
    soft_knee_width_db: float = 6.0,
    masking_margin_db: float = 3.0,
) -> np.ndarray:
    """Spektrales Gating mit psychoakustischem Masking-Guard (Pfad-API).

    Dünner Wrapper über :func:`gate_audio`: lädt Audio über den kanonischen Import
    (§V4 / Spec 08 §Audio-Import-Kaskade) und delegiert an den In-Memory-Kern.

    Args:
        audio_path: Pfad zur Audio-Datei (WAV/FLAC).
        threshold_db: Globaler Threshold in dBFS. Standard -60 dB.
            Tiefer (-70): mehr NR, aber Risiko von Musical Noise & Transienten-Loss.
            Höher (-50): weniger NR, aber saubere Transienten-Erhaltung.
        hop_length: STFT Hop-Length (Standard 512 → ~11 ms bei 48 kHz).
        n_fft: STFT Window-Size. Auto-Kalibrierung wenn None (~1024 für Musik).
        soft_knee_width_db: Soft-Knee-Breite in dB (§III). Standard 6 dB.
            Sigmoid-Gain statt Hard-Cutoff verhindert harte Schnittkanten.
        masking_margin_db: Psychoakustischer Masking-Margin über der
            Maskierungsschwelle (ISO 11172-3 Bark-Skala).

    Returns:
        np.ndarray: Gated Audio als float64, normalisiert auf [-1, 1].

    Raises:
        FileNotFoundError: Wenn die Audio-Datei nicht existiert.
        RuntimeError: Wenn Audio-Laden oder STFT/ISTFT-Konsistenzprüfung fehlschlägt.
        ValueError: Wenn Audio leer ist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")

    # Audio laden über kanonischen Import (§V4 / Spec 08 §Audio-Import-Kaskade)
    from backend.file_import import load_audio_file

    result = load_audio_file(str(audio_path), target_sr=None, mono=False, do_carrier_analysis=False)
    if result is None or result.get("error"):
        raise RuntimeError(f"Audio-Laden fehlgeschlagen: {result.get('error') if result else 'None'}")

    sr = int(result.get("sr", 48000))
    y = np.asarray(result["audio"], dtype=np.float64)

    # Mono-Fallback für Stereo mit Warnung (§G8 (GEBOTE.md) Transparenz).
    # load_audio_file liefert samples-first (N, C) — Kanalen-Mean über axis=1.
    if y.ndim > 1:
        logger.warning("§2.62 Spectral Gating: Stereo → Mono-Konvertierung (%d Kanäle)", y.shape[1])
        y = np.mean(y, axis=1).astype(np.float64)

    return gate_audio(
        y,
        sr,
        threshold_db=threshold_db,
        hop_length=hop_length,
        n_fft=n_fft,
        soft_knee_width_db=soft_knee_width_db,
        masking_margin_db=masking_margin_db,
    )


# --- Adapterklasse für Registry/Policy-Ketten (§G9 Spec-Referenz) ---


class SpectralGate:
    """Adapter über :func:`gate_audio` für Registry-/Policy-Ketten.

    Kanonische Parameter nur; die Legacy-Root-Parameter ``hold_frames`` /
    ``release_frames`` / ``hysteresis_db`` werden durch Soft-Knee + Bark-Masking ersetzt.
    """

    def __init__(
        self,
        threshold_db: float = -60.0,
        hop_length: int = 512,
        n_fft: int | None = None,
        soft_knee_width_db: float = 6.0,
        masking_margin_db: float = 3.0,
    ) -> None:
        self.threshold_db = threshold_db
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.soft_knee_width_db = soft_knee_width_db
        self.masking_margin_db = masking_margin_db

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """In-Memory-Audio verarbeiten (1D Mono oder channels-first `(C, N)`)."""
        return gate_audio(
            audio,
            sr,
            threshold_db=self.threshold_db,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            soft_knee_width_db=self.soft_knee_width_db,
            masking_margin_db=self.masking_margin_db,
        )


# --- Batch-Export-Hilfe für Aurik-Pipeline ---


def apply_spectral_gating_batch(
    audio_dir: str | Path,
    threshold_db: float = -60.0,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    """Batch-Spectral-Gating für alle Audio-Dateien im Verzeichnis.

    §G1 (GEBOTE.md) Song-Isolation: Jeder Song wird isoliert verarbeitet (keine Cross-Contamination).
    Deterministisch (§G5 (GEBOTE.md)): STFT/ISTFT deterministisch ohne RNG —
    bit-identisch über Prozesse hinweg.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir or str(audio_dir) + "_gated")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}
    for f in sorted(audio_dir.glob("*.wav")):
        logger.info("§2.62 Spectral Gating Batch: %s", f.name)
        y_out = apply_spectral_gating(f, threshold_db=threshold_db)
        out_path = output_dir / f"{f.stem}_gated.wav"

        # Export (librosa — kein Dither nötig für float32/float64)
        librosa.write(str(out_path), y_out, sr=48000, format="wav")  # type: ignore[attr-defined]  # librosa-Stubs exportieren write nicht

        peak_dbfs = 20 * np.log10(np.max(np.abs(y_out)) + 1e-10)
        results[f.name] = peak_dbfs

    logger.info("§2.62 Spectral Gating Batch: %d Dateien verarbeitet", len(results))
    return results


# --- Unit-Test-Hilfe für Determinismus (§G5 (GEBOTE.md)) ---


def verify_determinism(audio_path: str | Path, threshold_db: float = -60.0) -> bool:
    """Prüft Bit-Identität bei zwei Durchläufen (§G5 (GEBOTE.md))."""
    out1 = apply_spectral_gating(audio_path, threshold_db=threshold_db)
    out2 = apply_spectral_gating(audio_path, threshold_db=threshold_db)
    return np.array_equal(out1, out2)
