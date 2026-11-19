"""
PhonemeDetector — DSP-basierter Phonem-/Sibilanten-Segmentierer.

Erkennt stimmhafte (V) / stimmlose (C) Segmente und Sibilanten
anhand von ZCR, Energie und Spektraltilt. Kein ML-Modell nötig.

Referenz: §2.8 Vocal-Restaurierungskette (ConsonantDetector, PhonemeDetector)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhonemeResult:
    """Ergebnis der Phonem-Detektion."""

    phonemes: list[str] = field(default_factory=list)
    """Erkannte Phonem-Labels (z. B. ['V', 'C', 'sib', 'sil'])."""

    confidence: float = 0.0
    """Gesamt-Konfidenz ∈ [0, 1]."""

    timestamps_ms: list[float] = field(default_factory=list)
    """Zeitstempel (ms) pro Phonem-Segment (optional)."""

    sibilant_mask: np.ndarray | None = field(default=None, repr=False)
    """Sample-genaue Sibilanten-Maske (bool), oder None wenn nicht berechnet."""


_phoneme_instance: PhonemeDetector | None = None
_phoneme_lock = threading.Lock()


def get_phoneme_detector() -> PhonemeDetector:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _phoneme_instance
    if _phoneme_instance is None:
        with _phoneme_lock:
            if _phoneme_instance is None:
                _phoneme_instance = PhonemeDetector.__new__(PhonemeDetector)
                _phoneme_instance._initialized = True  # type: ignore[attr-defined]
    return _phoneme_instance


class PhonemeDetector:
    """DSP-basierter Phonem-Detektor für Gesangs- und Sprachsignale.

    Erkennt stimmhafte (V), stimmlose Konsonanten (C), Sibilanten (sib)
    und Stille (sil) anhand von:
        - Zero-Crossing-Rate (ZCR): hoch → C / sib; niedrig → V
        - Kurzzeit-Energie: sehr niedrig → sil
        - Spektraler Tilt (HF-Energie 4–12 kHz): erhöht → sib

    Invarianten:
        - Ausgabe ist niemals None
        - Alle Score-Felder sind finite (NaN/Inf werden auf 0.0 gefangen)
        - Singleton-Pattern (Double-Checked Locking, §3.2)

    Referenz:
        Mauch & Dixon (2014) pYIN — Voicing-Klassifikation via ZCR/Energie
        §2.8 Aurik-Spec: ConsonantDetector-Kriterium ZCR > 0.3 + HF-Energie dominant
    """

    # ── Hyperparameter ──────────────────────────────────────────────────── #
    FRAME_SIZE: int = 1024  # Frames für ZCR/Energie-Analyse
    HOP_SIZE: int = 512
    ENERGY_SILENCE_THR: float = 1e-7  # Unter diesem RMS² → Stille
    ZCR_VOICED_THR: float = 0.10  # ZCR < thr → stimmhaft (V)
    ZCR_SIBILANT_THR: float = 0.30  # ZCR > thr + HF → Sibilant (sib)
    HF_ENERGY_THR: float = 0.25  # HF-Anteil > thr → Sibilant-Kandidat
    # ────────────────────────────────────────────────────────────────────── #

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> PhonemeResult:
        """Erkennt Phonem-Klassen im Audio-Signal.

        Algorithmus:
            1. Frame-weise ZCR + Kurzzeit-Energie (FRAME_SIZE, HOP_SIZE)
            2. Sibilanten-Kandidaten via HF-Energie-Anteil (4–12 kHz / Nyquist)
            3. Pro Frame: sil / V / sib / C
            4. Kompaktierung: konsekutive gleiche Label → ein Eintrag

        Args:
            audio:       Mono-Signal float32/float64, ≥ 1 Sample
            sample_rate: Sample-Rate in Hz (intern ungenutzt außer für HF-Grenze)

        Returns:
            PhonemeResult mit phonemes, confidence, timestamps_ms
        """
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        try:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)

            if audio.size == 0:
                return PhonemeResult(phonemes=["sil"], confidence=1.0)

            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

            n = len(audio)
            phonemes: list[str] = []
            timestamps: list[float] = []

            nyquist = sample_rate / 2.0
            hf_cutoff_bin_frac = min(4000.0 / nyquist, 1.0)

            pos = 0
            while pos < n:
                frame = audio[pos : pos + self.FRAME_SIZE]
                if len(frame) < 2:
                    break

                # Kurzzeit-Energie
                energy = float(np.mean(frame**2))
                energy = np.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

                if energy < self.ENERGY_SILENCE_THR:
                    label = "sil"
                else:
                    # Zero-Crossing-Rate
                    zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
                    zcr = np.nan_to_num(zcr, nan=0.0, posinf=0.0, neginf=0.0)

                    # HF-Energie-Anteil für Sibilanten-Check
                    hf_ratio = self._hf_energy_ratio(frame, hf_cutoff_bin_frac)
                    hf_ratio = np.nan_to_num(hf_ratio, nan=0.0, posinf=0.0, neginf=0.0)

                    if hf_ratio > self.HF_ENERGY_THR:
                        label = "sib"
                    elif zcr < self.ZCR_VOICED_THR:
                        label = "V"
                    else:
                        label = "C"

                # Kompaktierung
                if not phonemes or phonemes[-1] != label:
                    phonemes.append(label)
                    timestamps.append(round(pos / max(sample_rate, 1) * 1000.0, 2))

                pos += self.HOP_SIZE

            if not phonemes:
                phonemes = ["sil"]
                timestamps = [0.0]

            confidence = self._estimate_confidence(audio)

            return PhonemeResult(
                phonemes=phonemes,
                confidence=confidence,
                timestamps_ms=timestamps,
            )

        except Exception as exc:
            logger.warning("PhonemeDetector.detect Fehler: %s", exc)
            return PhonemeResult(phonemes=["unk"], confidence=0.0)

    # ── Hilfsmethoden ────────────────────────────────────────────────────── #

    def _hf_energy_ratio(self, frame: np.ndarray, cutoff_frac: float) -> float:
        """Anteil der Energie oberhalb der Cutoff-Frequenz via FFT-Magnitude."""
        try:
            spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            total = float(np.sum(spec**2))
            if total < 1e-12:
                return 0.0
            cut = max(1, int(cutoff_frac * len(spec)))
            hf = float(np.sum(spec[cut:] ** 2))
            return min(1.0, hf / total)
        except Exception:
            logger.warning("phoneme_detector.py::_hf_energy_ratio fallback", exc_info=True)
            return 0.0

    def _estimate_confidence(self, audio: np.ndarray) -> float:
        """Schätzung der Detektions-Konfidenz aus Signal-SNR-Proxy."""
        try:
            rms = float(np.sqrt(np.mean(audio**2)))
            return float(np.clip(rms * 10.0, 0.0, 1.0))
        except Exception:
            logger.warning("phoneme_detector.py::_estimate_confidence fallback", exc_info=True)
            return 0.0


# Convenience-Wrapper
def detect_phonemes(audio: np.ndarray, sample_rate: int) -> PhonemeResult:
    """Convenience-Wrapper für PhonemeDetector (§3.2 Singleton-Pattern)."""
    return get_phoneme_detector().detect(audio, sample_rate)
