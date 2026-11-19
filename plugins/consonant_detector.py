"""
plugins/consonant_detector.py — ConsonantDetector (§2.8 Step 5b)
=================================================================

Eigenständiger Singleton für Frikativ-/Konsonanten-Segmenterkennung
gemäß Aurik-Spec §2.8 Step 5b:

    Erkennungskriterium: ZCR > 0.3 UND Energie in 4–16 kHz dominant.

Der Detector ist von ConsonantEnhancement (Step 5c) entkoppelt, damit
ContentAwareProcessor (§2.36) und PerceptualAttentionModel (§2.22) ihn
unabhängig aufrufen können, ohne die komplette Enhancement-Kette zu
instantiieren.

Singleton-Pattern (§3.2), NaN/Inf-Schutz (§3.1), vollständige
Type-Annotations (§3.7).

Author: Aurik Development Team
Spec:   §2.8 Step 5b, §3.2, §3.7
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── Erkennungsschwellen (§2.8 Step 5b) ──────────────────────────────────── #
ZCR_THRESHOLD: float = 0.30  # Zero-Crossing-Rate-Mindestanteil
HF_ENERGY_THRESHOLD: float = 0.25  # Mindestanteil HF-Energie an Gesamt
HF_LOW_HZ: float = 4_000.0  # Untere Grenze Hochfrequenzband
HF_HIGH_HZ: float = 16_000.0  # Obere Grenze Hochfrequenzband

# Frame-Analyse-Parameter
_FRAME_SIZE: int = 1024  # Samples pro Frame (~21 ms @ 48 kHz)
_HOP_SIZE: int = 512  # Hop-Größe (50 % Überlappung)


# ── Ergebnis-Dataclass ───────────────────────────────────────────────────── #


@dataclass
class ConsonantDetectionResult:
    """Ergebnis der Frikativ-/Konsonanten-Erkennung (§2.8 Step 5b)."""

    mask: np.ndarray = field(repr=False)
    """Sample-genaue bool-Maske [n_samples]; True = Frikativ/Konsonant."""

    n_fricative_frames: int = 0
    """Anzahl erkannter Frikativ-Frames."""

    fricative_ratio: float = 0.0
    """Anteil Frikativ-Frames an allen analysierten Frames ∈ [0, 1]."""

    mean_zcr: float = 0.0
    """Mittlere ZCR in Frikativ-Frames."""

    mean_hf_ratio: float = 0.0
    """Mittlerer HF-Energie-Anteil in Frikativ-Frames."""

    sample_rate: int = 48_000
    """Verwendete Sample-Rate."""


# ── Singleton ────────────────────────────────────────────────────────────── #

_instance: ConsonantDetector | None = None
_lock = threading.Lock()


def get_consonant_detector() -> ConsonantDetector:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ConsonantDetector()
    return _instance


def detect_consonants(
    audio: np.ndarray,
    sample_rate: int,
    voice_gender: str = "unknown",
) -> ConsonantDetectionResult:
    """Convenience-Wrapper: Frikativ-Maske ohne Klassen-Instantiierung.

    Args:
        audio:        float32/64 nd-array, mono oder stereo.
        sample_rate:  Sample-Rate in Hz (sollte 48 000 Hz sein, §6.6).
        voice_gender: Stimmtyp-Hinweis für HF-Band-Anpassung (optional).

    Returns:
        ConsonantDetectionResult mit sample-genauer bool-Maske.
    """
    return get_consonant_detector().detect(audio, sample_rate, voice_gender)


# ── Hauptklasse ──────────────────────────────────────────────────────────── #


class ConsonantDetector:
    """Erkennt Frikativ-/Konsonanten-Segmente via ZCR + HF-Energie (§2.8 Step 5b).

    Algorithmus:
        1. Audio zu Mono konvertieren (Mittelung über Kanäle).
        2. Frame-weise Analyse (Fenster 1024 Samples, Hop 512):
           a) ZCR = Σ|sign(x[n]) − sign(x[n−1])| / (2 · frame_len)
              → ZCR > 0.3: Frikativ-Kandidat (stimmlose Konsonanten typisch 0.3–0.6)
           b) HF-Energie-Anteil via RFFT:
              hf_ratio = Σ|X[f]|² (f ∈ [4–16 kHz]) / Σ|X[f]|²
              → hf_ratio > 0.25: Hochenergetisches HF-Band → Konsonant bestätigt
        3. Beide Bedingungen müssen gleichzeitig erfüllt sein (AND-Logik).
        4. Sample-genaue Maske: markierte Frames → Samples übertragen.

    Stimmtyp-Adaptation (optional, voice_gender):
        MALE:        HF-Band   5–10 kHz
        FEMALE:      HF-Band   6–12 kHz
        CHILD:       HF-Band   7–14 kHz
        Androgynous: HF-Band   5.5–11 kHz
        Unknown:     HF-Band   4–16 kHz (breit, konservativ)

    Invarianten:
        - Ausgang mask.shape[0] == audio.shape[0] (mono) / audio.shape[-1] (stereo)
        - NaN/Inf im Eingang → nan_to_num vor Verarbeitung
        - Leeres Audio oder n < _FRAME_SIZE → leere Maske, kein Fehler
        - mask.dtype == bool (immer)

    Referenz:
        §2.8 Stimmtyp-Adaptierung (VoiceGender-System)
        Rabiner & Schafer (1978): Digital Processing of Speech Signals
          (ZCR als Voiced/Unvoiced-Detektor, klassischer DSP-Ansatz)
    """

    # Stimmtyp-adaptive HF-Bänder (Hz-Tupel)
    _HF_BANDS: dict[str, tuple[float, float]] = {
        "male": (5_000.0, 10_000.0),
        "female": (6_000.0, 12_000.0),
        "child": (7_000.0, 14_000.0),
        "androgynous": (5_500.0, 11_000.0),
        "unknown": (HF_LOW_HZ, HF_HIGH_HZ),
    }

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
        voice_gender: str = "unknown",
    ) -> ConsonantDetectionResult:
        """Gibt sample-genaue Frikativ-Maske zurück.

        Args:
            audio:        float32/64 nd-array, mono [n] oder stereo [2, n] / [n, 2].
            sample_rate:  Sample-Rate in Hz.
            voice_gender: Stimmtyp für HF-Band-Adaptation (default: unknown).

        Returns:
            ConsonantDetectionResult (mask, Statistiken). Niemals raise.

        Raises:
            Nichts — alle Ausnahmen werden intern abgefangen.
        """
        try:
            return self._detect_safe(audio, sample_rate, voice_gender)
        except Exception as exc:  # pragma: no cover
            logger.warning("ConsonantDetector: Fehler bei Erkennung (%s) — leere Maske", exc)
            n = (
                audio.shape[0]
                if isinstance(audio, np.ndarray) and audio.ndim == 1
                else (audio.shape[-1] if isinstance(audio, np.ndarray) else 0)
            )
            return ConsonantDetectionResult(
                mask=np.zeros(n, dtype=bool),
                sample_rate=sample_rate,
            )

    # ── Interne Implementierung ──────────────────────────────────────── #

    def _detect_safe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        voice_gender: str,
    ) -> ConsonantDetectionResult:
        # Validierung & NaN-Schutz
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return ConsonantDetectionResult(
                mask=np.zeros(0, dtype=bool),
                sample_rate=sample_rate,
            )

        audio = np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # Mono-Konvertierung
        if audio.ndim == 1:
            mono = audio
        elif audio.ndim == 2:
            # Unterstütze [channels, samples] und [samples, channels]
            mono = audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1)
        else:
            mono = audio.flatten()

        n = len(mono)
        mask = np.zeros(n, dtype=bool)

        if n < _FRAME_SIZE:
            return ConsonantDetectionResult(
                mask=mask,
                sample_rate=sample_rate,
            )

        # Stimmtyp-adaptives HF-Band
        hf_lo, hf_hi = self._HF_BANDS.get(voice_gender.lower(), self._HF_BANDS["unknown"])
        nyq = sample_rate / 2.0
        hf_lo = min(hf_lo, nyq * 0.15)
        hf_hi = min(hf_hi, nyq * 0.95)

        total_frames = 0
        fricative_frames = 0
        zcr_sum = 0.0
        hf_ratio_sum = 0.0

        for start in range(0, n - _FRAME_SIZE, _HOP_SIZE):
            frame = mono[start : start + _FRAME_SIZE]
            total_frames += 1

            # ── ZCR ── #
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2)
            if zcr < ZCR_THRESHOLD:
                continue

            # ── HF-Energie-Anteil ── #
            n_fft = _FRAME_SIZE
            window = np.hanning(n_fft)
            spec = np.abs(np.fft.rfft(frame * window)) ** 2
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
            hf_mask = (freqs >= hf_lo) & (freqs <= hf_hi)
            total_e = float(spec.sum()) + 1e-12
            hf_e = float(spec[hf_mask].sum())
            hf_ratio = hf_e / total_e

            if hf_ratio < HF_ENERGY_THRESHOLD:
                continue

            # ── Frikativ bestätigt ── #
            fricative_frames += 1
            zcr_sum += zcr
            hf_ratio_sum += hf_ratio
            end = min(start + _FRAME_SIZE, n)
            mask[start:end] = True

        fricative_ratio = fricative_frames / max(total_frames, 1)
        mean_zcr = zcr_sum / max(fricative_frames, 1)
        mean_hf = hf_ratio_sum / max(fricative_frames, 1)

        logger.debug(
            "ConsonantDetector: %d/%d Frames Frikativ (ratio=%.2f, mean_zcr=%.2f, mean_hf=%.2f)",
            fricative_frames,
            total_frames,
            fricative_ratio,
            mean_zcr,
            mean_hf,
        )

        return ConsonantDetectionResult(
            mask=mask,
            n_fricative_frames=fricative_frames,
            fricative_ratio=float(fricative_ratio),
            mean_zcr=float(mean_zcr),
            mean_hf_ratio=float(mean_hf),
            sample_rate=sample_rate,
        )
