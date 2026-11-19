"""
Perceptual Audio Embedder — Aurik 10.0.0
======================================
Erzeugt einen L2-normierten 256-dimensionalen Embedding-Vektor der die
musikalisch-perzeptuelle Charakteristik einer Aufnahme in einem normierten
Merkmalsraum kodiert. Das Embedding erlaubt Ähnlichkeitsvergleiche via
Kosinus-Ähnlichkeit und dient als objektive Grundlage für kausal-basiertes
Routing und perceptuelle Qualitätsbewertung.

Architektur (kein neuronales Netz — ausschließlich signaltheoretisch):
  Kanal A (96 Dim):  Multi-Auflösungs-Spektrogramm-Statistiken
                     3 STFT-Auflösungen × 16 Frequenzbänder × 2 Momente
  Kanal B (48 Dim):  Psychoakustische Lautheitskarte (Zwicker-Approximation)
                     24 Bark-Bänder × 2 Momente
  Kanal C (36 Dim):  Chromatik (12 Halbtonklassen × 3 Zeitfenster)
  Kanal D (32 Dim):  Temporale Modulation (AM/FM-Detektion, 8 Träger × 4 Stats)
  Kanal E (44 Dim):  Tonal/Perkussive Separation (HPSS-Energie + Transientenrate)

Gesamt: 96 + 48 + 36 + 32 + 44 = 256 Dim → L2-normiert

Referenzen:
  - Zwicker & Fastl, Psychoacoustics (1999)
  - Müller, Fundamentals of Music Processing (2015)
  - Ellis, Chroma Feature Analysis (2007)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

_EMBED_DIM = 256
_TARGET_SR = 48000  # Pflicht-Invariante: stets 48 000 Hz (Section 6.5)

# Multi-Auflösungs-STFT
_STFT_SIZES = [256, 1024, 4096]  # FFT-Fenstergrößen
_STFT_HOPS = [64, 256, 1024]  # Hop-Sizes

# Frequenzband-Partition (log-gleichmäßig, 16 Bänder)
_FREQ_BANDS = 16

# Bark-Bänder (24 nach Zwicker)
_BARK_EDGES_HZ = [
    0,
    100,
    200,
    300,
    400,
    510,
    630,
    770,
    920,
    1080,
    1270,
    1480,
    1720,
    2000,
    2320,
    2700,
    3150,
    3700,
    4400,
    5300,
    6400,
    7700,
    9500,
    12000,
    15500,
]

# AM-Modulationsträger (Hz)
_MOD_FREQS_HZ = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]


# ---------------------------------------------------------------------------
# Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class AudioEmbedding:
    """256-dimensionaler perzeptueller Embedding-Vektor."""

    vector: np.ndarray  # shape (256,), L2-normiert
    channel_a_spectral: np.ndarray  # 96-dim Spektrogramm-Statistiken
    channel_b_loudness: np.ndarray  # 48-dim Bark-Lautheit
    channel_c_chroma: np.ndarray  # 36-dim Chromatik
    channel_d_modulation: np.ndarray  # 32-dim AM/FM
    channel_e_hpss: np.ndarray  # 44-dim Tonal/Perkussiv
    sample_rate: int
    duration_s: float

    def cosine_similarity(self, other: AudioEmbedding) -> float:
        """Kosinus-Ähnlichkeit [-1, 1]. Gleicher Sound → ~1.0."""
        a = self.vector
        b = other.vector
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def perceptual_distance(self, other: AudioEmbedding) -> float:
        """Euklidischer Abstand im normierten Raum [0, √2]."""
        return float(np.linalg.norm(self.vector - other.vector))

    def to_dict(self) -> dict:
        return {
            "dim": int(self.vector.shape[0]),
            "norm": float(np.linalg.norm(self.vector)),
            "channel_a_mean": float(np.mean(self.channel_a_spectral)),
            "channel_b_mean": float(np.mean(self.channel_b_loudness)),
            "channel_c_mean": float(np.mean(self.channel_c_chroma)),
            "channel_d_mean": float(np.mean(self.channel_d_modulation)),
            "channel_e_mean": float(np.mean(self.channel_e_hpss)),
            "sample_rate": self.sample_rate,
            "duration_s": round(self.duration_s, 3),
        }


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _to_mono_resampled(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Mono-Downmix + Resampling auf _TARGET_SR."""
    if audio.ndim == 2:
        mono = np.mean(audio, axis=0) if audio.shape[0] < audio.shape[1] else np.mean(audio, axis=1)
    else:
        mono = audio.astype(np.float64)

    mono = mono.astype(np.float64)

    # Amplitude normieren auf [-1, 1]
    peak: float = float(np.max(np.abs(mono)))
    if peak > 1e-9:
        mono = mono / peak

    if src_sr != _TARGET_SR:
        ratio = _TARGET_SR / src_sr
        n_out = int(len(mono) * ratio)
        mono = scipy.signal.resample(mono, n_out)

    return mono  # type: ignore[no-any-return]


def _stft_mag(sig: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Betragssspektrogramm via scipy.signal.ShortTimeFFT."""
    win = scipy.signal.windows.hann(n_fft)
    # Padding
    pad = n_fft // 2
    sig_p = np.pad(sig, (pad, pad))
    nframes = 1 + (len(sig_p) - n_fft) // hop
    if nframes < 1:
        return np.zeros((n_fft // 2 + 1, 1))  # type: ignore[no-any-return]
    frames = np.lib.stride_tricks.sliding_window_view(sig_p, n_fft)[::hop][:nframes]
    S = np.abs(np.fft.rfft(frames * win, n=n_fft, axis=1)).T  # (bins, frames)
    return S.astype(np.float32)  # type: ignore[no-any-return]


def _log_band_energy(S: np.ndarray, n_fft: int, n_bands: int) -> np.ndarray:
    """Aggregiert Betragsspektrum in log-gleichmäßige Frequenzbänder."""
    bins = S.shape[0]
    # log-gleichmäßige Grenzen
    edges = np.geomspace(1, bins, n_bands + 1).astype(int)
    edges = np.clip(edges, 0, bins - 1)
    band_energy = np.zeros((n_bands, S.shape[1]), dtype=np.float32)
    for b in range(n_bands):
        lo, hi = edges[b], max(edges[b + 1], edges[b] + 1)
        band_energy[b] = np.mean(S[lo:hi], axis=0)
    return band_energy  # type: ignore[no-any-return]


def _bark_hz_to_band(f_hz: float) -> int:
    """Frequenz in Hz → Bark-Band-Index (0-23)."""
    for i, edge in enumerate(_BARK_EDGES_HZ[1:], 1):
        if f_hz < edge:
            return i - 1
    return len(_BARK_EDGES_HZ) - 2


# ---------------------------------------------------------------------------
# Kanal A: Multi-Auflösungs-Spektrogramm-Statistiken (96 Dim)
# ---------------------------------------------------------------------------


def _channel_a(mono: np.ndarray) -> np.ndarray:
    """3 STFT-Auflösungen × 16 Frequenzbänder × 2 Statistiken (μ, σ) = 96 Dim."""
    feats = []
    for n_fft, hop in zip(_STFT_SIZES, _STFT_HOPS):
        S = _stft_mag(mono, n_fft, hop)
        S_db = np.log1p(S * 1e3)  # log-Komprimierung
        bands = _log_band_energy(S_db, n_fft, _FREQ_BANDS)  # (16, frames)
        mu = np.mean(bands, axis=1)  # (16,)
        sigma = np.std(bands, axis=1) + 1e-9
        feats.append(mu)
        feats.append(sigma)
    return np.concatenate(feats).astype(np.float32)  # type: ignore[no-any-return]  # 96 Dim


# ---------------------------------------------------------------------------
# Kanal B: Psychoakustische Bark-Lautheit (48 Dim)
# ---------------------------------------------------------------------------


def _channel_b(mono: np.ndarray) -> np.ndarray:
    """24 Bark-Bänder × 2 (μ, σ) = 48 Dim. Zwicker-Spezifische-Lautheit-Näherung."""
    n_fft = 1024
    hop = 256
    S = _stft_mag(mono, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / _TARGET_SR)  # (bins,)

    # Bin → Bark-Band zuordnen
    n_bark = len(_BARK_EDGES_HZ) - 1  # 24
    bark_energy = np.zeros((n_bark, S.shape[1]), dtype=np.float32)
    bin_to_bark = np.array([_bark_hz_to_band(f) for f in freqs])
    for b in range(n_bark):
        mask = bin_to_bark == b
        if np.any(mask):
            bark_energy[b] = np.mean(S[mask], axis=0)

    # Spezifische Lautheit: N' ∝ E^(0.23) (Näherung nach Zwicker Gl. 5.1)
    specific_loudness = np.power(bark_energy + 1e-10, 0.23)

    mu = np.mean(specific_loudness, axis=1)  # (24,)
    sigma = np.std(specific_loudness, axis=1) + 1e-9  # (24,)
    return np.concatenate([mu, sigma]).astype(np.float32)  # type: ignore[no-any-return]  # 48 Dim


# ---------------------------------------------------------------------------
# Kanal C: Chroma (CQT-Näherung, 36 Dim)
# ---------------------------------------------------------------------------


def _channel_c(mono: np.ndarray) -> np.ndarray:
    """12 Halbtonklassen × 3 Zeitfenster = 36 Dim. CQT-Näherung via Filterbank."""
    # Referenzfrequenzen für C1–B7 (84 Halbtöne)
    A4 = 440.0
    midi_nums = np.arange(24, 108)  # C1 = MIDI 24 … B7 = MIDI 107
    f_midi = A4 * 2.0 ** ((midi_nums - 69) / 12.0)

    # Kurze STFT
    n_fft, hop = 4096, 512
    S = _stft_mag(mono, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / _TARGET_SR)
    n_frames = S.shape[1]

    # Chroma-Energie: Summe über alle Oktaven pro Halbton
    chroma = np.zeros((12, n_frames), dtype=np.float32)
    freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    for i, f0 in enumerate(f_midi):
        pitch_class = int(midi_nums[i]) % 12
        # Gaussianisches Frequenzfenster um f0 (±Semitonebreite/2)
        sigma_f = f0 * (2 ** (1 / 24.0) - 1.0)  # halber Halbton
        window = np.exp(-0.5 * ((freqs - f0) / (sigma_f + 1)) ** 2)
        chroma[pitch_class] += np.dot(window, S)

    # Normalisierung pro Frame
    norm = np.sum(chroma, axis=0, keepdims=True) + 1e-9
    chroma = chroma / norm

    # 3 Zeitfenster: früh, mittel, spät
    feats = []
    thirds = np.array_split(np.arange(n_frames), 3)
    for seg in thirds:
        if len(seg) == 0:
            feats.append(np.zeros(12, dtype=np.float32))
        else:
            feats.append(np.mean(chroma[:, seg], axis=1).astype(np.float32))
    return np.concatenate(feats)  # type: ignore[no-any-return]  # 36 Dim


# ---------------------------------------------------------------------------
# Kanal D: Temporale AM/FM-Modulation (32 Dim)
# ---------------------------------------------------------------------------


def _channel_d(mono: np.ndarray) -> np.ndarray:
    """8 AM-Träger × 4 Statistiken (μ, σ, skew, kurt) = 32 Dim."""
    feats = []
    for f_mod in _MOD_FREQS_HZ:
        # Breitbandige Hüllkurven via Hilbert-Transform
        analytic: np.ndarray = scipy.signal.hilbert(np.asarray(mono, dtype=np.float64))  # type: ignore[call-overload]
        envelope = np.abs(analytic)

        # Bandpass um f_mod in der Hüllkurve
        nyq = _TARGET_SR / 2.0
        f_lo = max(f_mod * 0.7, 0.5)
        f_hi = min(f_mod * 1.4, nyq - 1)
        if f_hi > f_lo and f_hi < nyq:
            try:
                b, a = scipy.signal.butter(2, [f_lo / nyq, f_hi / nyq], btype="bandpass", output="ba")  # type: ignore[misc]
                mod_sig = scipy.signal.filtfilt(b, a, envelope)
            except Exception:
                mod_sig = envelope
        else:
            mod_sig = envelope

        mu = float(np.mean(mod_sig))
        sigma = float(np.std(mod_sig)) + 1e-9
        # Skewness
        skew = float(np.mean(((mod_sig - mu) / sigma) ** 3))
        skew = np.clip(skew, -5.0, 5.0)
        # Excess Kurtosis
        kurt = float(np.mean(((mod_sig - mu) / sigma) ** 4)) - 3.0
        kurt = np.clip(kurt, -5.0, 10.0)
        feats.extend([mu, sigma, skew, kurt])

    return np.array(feats, dtype=np.float32)  # type: ignore[no-any-return]  # 32 Dim


# ---------------------------------------------------------------------------
# Kanal E: Tonal/Perkussive Separation HPSS (44 Dim)
# ---------------------------------------------------------------------------


def _channel_e(mono: np.ndarray) -> np.ndarray:
    """Harmonic-Percussive Source Separation via Median-Filterung.
    Output: 44-dim — Energieverhältnis, Transienten-Rate, Spektrale Centroide,
    Tonalitäts-Index, Spektraler Kontrast (10 Bänder × 4 = 40 + 4 globale).
    """
    n_fft, hop = 1024, 256
    S = _stft_mag(mono, n_fft, hop)
    S2 = S**2

    # Median-Filterung (horizontal = harmonisch, vertikal = perkussiv)
    h_len = 17  # Harmonischer Median: zeitlich länger
    p_len = 9  # Perkussiver  Median: frequentiell kürzer

    # Harmonisch: Median entlang Zeitachse (axis=1)
    H = np.zeros_like(S2)
    P = np.zeros_like(S2)
    m = h_len // 2
    for i in range(S2.shape[0]):
        row = np.pad(S2[i], (m, m), mode="edge")
        H[i] = np.sort(np.lib.stride_tricks.sliding_window_view(row, h_len), axis=1)[:, h_len // 2]

    # Perkussiv: Median entlang Frequenzachse (axis=0)
    m2 = p_len // 2
    for j in range(S2.shape[1]):
        col = np.pad(S2[:, j], (m2, m2), mode="edge")
        P[:, j] = np.sort(np.lib.stride_tricks.sliding_window_view(col, p_len), axis=1)[:, p_len // 2]

    # Masken
    total = H + P + 1e-12
    mask_H = H / total
    mask_P = P / total
    SH = np.sqrt(H) * mask_H  # Harmonische Komponente
    SP = np.sqrt(P) * mask_P  # Perkussive Komponente

    # Globale Merkmale (4 Dim)
    e_h = float(np.mean(H))
    e_p = float(np.mean(P))
    ratio = e_h / (e_p + 1e-12)  # Harmonizitätsverhältnis
    trans_rate = float(np.mean(np.diff(np.log1p(SP), axis=1) ** 2))  # Transientenrate

    # Spektrale Centroide pro Komponente (2 Dim)
    freqs = np.linspace(0, _TARGET_SR / 2, S.shape[0])
    c_harm = float(np.sum(freqs[:, None] * SH) / (np.sum(SH) + 1e-9))
    c_perc = float(np.sum(freqs[:, None] * SP) / (np.sum(SP) + 1e-9))

    # Spektraler Kontrast: 10 Sub-Bänder (40 Dim)
    subband_edges = np.geomspace(1, S.shape[0], 11).astype(int)
    subband_edges = np.clip(subband_edges, 0, S.shape[0] - 1)
    contrast_feats = []
    for k in range(10):
        lo = subband_edges[k]
        hi = max(subband_edges[k + 1], lo + 1)
        seg = S[lo:hi]
        if seg.size == 0:
            contrast_feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        peak = float(np.mean(np.max(seg, axis=0)))
        valley = float(np.mean(np.min(seg, axis=0)))
        mu_s = float(np.mean(seg))
        sigma_s = float(np.std(seg)) + 1e-9
        contrast_feats.extend([peak, valley, mu_s, sigma_s])

    feats = [
        e_h,
        e_p,
        float(np.clip(ratio, 0, 50)),
        trans_rate,
        c_harm / (_TARGET_SR / 2 + 1),
        c_perc / (_TARGET_SR / 2 + 1),
        float(np.mean(mask_H)),
        float(np.std(mask_H)) + 1e-9,
        float(np.mean(mask_P)),
        float(np.std(mask_P)) + 1e-9,
        float(np.mean(mask_H) / (np.mean(mask_P) + 1e-9)),
        float(np.mean(SH > SP)),  # Tonalitätsindex
    ]
    feats.extend(contrast_feats[:32])  # 10 Bänder × 4 = 40, wir beschneiden auf 32
    return np.array(feats[:44], dtype=np.float32)  # type: ignore[no-any-return]  # 44 Dim


# ---------------------------------------------------------------------------
# Haupt-API
# ---------------------------------------------------------------------------


class PerceptualEmbedder:
    """
    Wandelt rohes Audio in einen 256-dim L2-normierten Embedding-Vektor um.

    Verwendung::

        embedder = PerceptualEmbedder()
        emb = embedder.embed(audio, sample_rate=44100)
        sim = emb.cosine_similarity(emb_ref)   # [0, 1] → gleicher Klang
    """

    def embed(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment_s: float | None = None,
    ) -> AudioEmbedding:
        """
        Berechnet das perzeptuelle Embedding.

        Args:
            audio:      np.ndarray mono (N,) oder stereo (2,N) / (N,2)
            sample_rate: Abtastrate in Hz
            segment_s:  Wenn angegeben, wird nur das mittlere Segment
                        dieser Länge analysiert (Geschwindigkeitsoptimierung)

        Returns:
            AudioEmbedding mit 256-dim L2-normiertem Vektor
        """
        assert sample_rate == 48000, f"PerceptualEmbedder.embed() erwartet SR=48000, erhalten: {sample_rate}"
        # §3.x NaN/Inf-Invariante: Bereinigung vor jeder weiteren Verarbeitung
        if not np.all(np.isfinite(audio)):
            logger.debug("PerceptualEmbedder.embed(): NaN/Inf in Eingabe bereinigt")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        mono = _to_mono_resampled(audio, sample_rate)

        # Segment-Extraktion (Mitte des Signals, max. segment_s Sekunden)
        if segment_s is not None:
            n_seg = int(segment_s * _TARGET_SR)
            if n_seg < len(mono):
                mid = len(mono) // 2
                mono = mono[max(0, mid - n_seg // 2) : mid + n_seg // 2]

        if len(mono) < 512:
            logger.warning("Audio zu kurz (%d Samples) – Null-Embedding", len(mono))
            vec = np.zeros(_EMBED_DIM, dtype=np.float32)
            return AudioEmbedding(
                vector=vec,
                channel_a_spectral=vec[:96],
                channel_b_loudness=vec[96:144],
                channel_c_chroma=vec[144:180],
                channel_d_modulation=vec[180:212],
                channel_e_hpss=vec[212:],
                sample_rate=sample_rate,
                duration_s=len(mono) / _TARGET_SR,
            )

        ch_a = _channel_a(mono)  # 96
        ch_b = _channel_b(mono)  # 48
        ch_c = _channel_c(mono)  # 36
        ch_d = _channel_d(mono)  # 32
        ch_e = _channel_e(mono)  # 44

        combined = np.concatenate([ch_a, ch_b, ch_c, ch_d, ch_e])  # 256
        # NaN/Inf-Sicherheitsnetz
        combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)

        # L2-Normierung
        norm = float(np.linalg.norm(combined))
        if norm > 1e-9:
            combined = combined / norm

        return AudioEmbedding(
            vector=combined.astype(np.float32),
            channel_a_spectral=combined[:96],
            channel_b_loudness=combined[96:144],
            channel_c_chroma=combined[144:180],
            channel_d_modulation=combined[180:212],
            channel_e_hpss=combined[212:],
            sample_rate=sample_rate,
            duration_s=len(mono) / _TARGET_SR,
        )


# Singleton
_embedder: PerceptualEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> PerceptualEmbedder:
    """Gibt den globalen Singleton-Embedder zurück."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = PerceptualEmbedder()
    return _embedder


def embed_audio(audio: np.ndarray, sample_rate: int, segment_s: float | None = 10.0) -> AudioEmbedding:
    """Convenience-Funktion: Embedding für ein Audio-Array."""
    return get_embedder().embed(audio, sample_rate, segment_s=segment_s)
