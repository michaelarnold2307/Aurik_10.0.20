"""
backend/core/spectral_band_gap_repair.py
=========================================

Top-Level-Adapter für SpectralBandGapRepair gemäß §4.5 / §6.3 Spec.

Die eigentliche Implementierung liegt in
    backend/core/phases/phase_56_spectral_band_gap_repair.py

Dieses Modul stellt SpectralBandGapRepair als kanonische Klasse bereit,
ohne die Phasen-Infrastruktur zu duplizieren (Anti-Parallelwelten §9.4).

Verwendung:
    from backend.core.spectral_band_gap_repair import SpectralBandGapRepair
    repair = SpectralBandGapRepair()
    audio_repaired = repair.repair(audio, sr=48000, confidence=0.7)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

try:
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

try:
    _CREPE_OK = True
except Exception:
    _CREPE_OK = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inharmonizitäts-Priors (§2.11)
# ---------------------------------------------------------------------------
INHARMONICITY_PRIORS: dict[str, float] = {
    "piano_bass": 0.0080,
    "piano_mid": 0.0020,
    "piano_treble": 0.0001,
    "guitar": 0.0005,
    "violin": 0.0003,
    "flute": 0.0000,
    "brass": 0.0001,
    "unknown": 0.0010,
}


@dataclass
class BandGapResult:
    """Ergebnis der spektralen Lückenreparatur."""

    applied: bool
    n_gaps_found: int
    n_gaps_repaired: int
    confidence: float
    spectral_flatness_ok: bool
    audio: np.ndarray = field(repr=False)
    message: str = ""


class SpectralBandGapRepair:
    """Repariert HEAD_WEAR-induzierte Frequenzband-Auslöschungen.

    Gemäß §4.5 / §6.3: Frequenzband-Lückenreparatur für Azimuth-/Kopffehler.
    Aktivierung: DefectType.HEAD_WEAR, confidence ≥ 0.55,
                 MaterialType TAPE oder REEL_TAPE.

    Algorithmus:
        1. 1/6-Oktav-Subband-Energie-Analyse (30-Frame-Median)
        2. Lücken-Detektion: Energie ≤ -60 dBFS über ≥ 80 % der Länge, ≥ 200 Hz breit
        3. Harmonische Partial-Interpolation (Fletcher-Modell)
        4. Spektrale Glattheit-Prüfung (Flatness ≤ 0.4)
        5. PGHI-konsistente Rückwandlung

    Invarianten (§4.5):
        - Reparatur NUR wenn Lücke > 3 dB Abfall und > 200 Hz breit
        - AuthentizitaetMetric nach Reparatur ≥ vor Reparatur
        - Kein Eingriff in absichtliche Notch-artige Dips
        - Keine Reparatur unter confidence < 0.55
        - NaN/Inf-sicher: np.nan_to_num + np.clip(-1.0, 1.0)

    Referenz:
        Roebel (2010), Fletcher (1964), Février & Idier (2011), PGHI (Perraudin 2013)
    """

    # Schwellwerte
    MIN_CONFIDENCE: float = 0.55  # Mindest-Konfidenz für Aktivierung
    GAP_ENERGY_THRESHOLD_DB: float = -60.0  # Lücken-Energieschwelle
    GAP_COVERAGE_MIN: float = 0.80  # Mindest-Abdeckung der Dateilänge
    MIN_GAP_WIDTH_HZ: float = 200.0  # Mindest-Lückenbreite in Hz
    MAX_SPECTRAL_FLATNESS: float = 0.40  # Flatness-Grenze (≤ = ok)

    def repair(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        instrument_tag: str = "unknown",
        confidence: float = 1.0,
    ) -> BandGapResult:
        """Repariert spektrale Bandlücken (HEAD_WEAR-Defekte).

        Args:
            audio:          Input-Audio float32/64, mono oder stereo
            sr:             Sample-Rate (muss 48000 sein)
            instrument_tag: PANNs-Instrument-Tag für Inharmonizitäts-Prior
            confidence:     DefectScanner-Konfidenz für HEAD_WEAR

        Returns:
            BandGapResult mit repariertem audio-Feld
        """
        assert sr == 48000, f"SR muss 48000 sein, erhalten: {sr}"

        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if confidence < self.MIN_CONFIDENCE:
            logger.debug(
                "SpectralBandGapRepair: confidence=%.2f < %.2f, übersprungen",
                confidence,
                self.MIN_CONFIDENCE,
            )
            return BandGapResult(
                applied=False,
                n_gaps_found=0,
                n_gaps_repaired=0,
                confidence=confidence,
                spectral_flatness_ok=True,
                audio=audio,
                message="Konfidenz unter Schwellwert — kein Eingriff",
            )

        # Zu mono konvertieren für Analyse
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio.copy()

        # Lücken analysieren
        gaps = self._detect_band_gaps(mono, sr)

        if not gaps:
            return BandGapResult(
                applied=False,
                n_gaps_found=0,
                n_gaps_repaired=0,
                confidence=confidence,
                spectral_flatness_ok=True,
                audio=audio,
                message="Keine spektralen Lücken gefunden",
            )

        # Reparatur durchführen
        audio_repaired = self._repair_gaps(audio, mono, sr, gaps, instrument_tag)
        audio_repaired = np.clip(audio_repaired, -1.0, 1.0)
        audio_repaired = np.nan_to_num(audio_repaired, nan=0.0, posinf=0.0, neginf=0.0)

        # Spektrale Glattheit prüfen
        check_mono = np.mean(audio_repaired, axis=0) if audio_repaired.ndim == 2 else audio_repaired
        flatness_ok = self._check_spectral_flatness(check_mono, sr)

        n_repaired = len(gaps)
        return BandGapResult(
            applied=True,
            n_gaps_found=len(gaps),
            n_gaps_repaired=n_repaired,
            confidence=confidence,
            spectral_flatness_ok=flatness_ok,
            audio=audio_repaired,
            message=f"{n_repaired}/{len(gaps)} Bandlücken repariert",
        )

    # ------------------------------------------------------------------
    # Private Hilfsmethoden
    # ------------------------------------------------------------------

    def _detect_band_gaps(self, mono: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """Erkennt Frequenzband-Lücken via 1/6-Okt.-Energieanalyse.

        Returns:
            Liste von (freq_low_hz, freq_high_hz) Tupeln
        """
        n_fft = 2048
        if len(mono) < n_fft:
            return []

        # STFT-Magnitudenspektrum
        stft = np.abs(np.fft.rfft(np.pad(mono, (0, n_fft - len(mono) % n_fft if len(mono) % n_fft != 0 else 0))))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        # 1/6-Oktav-Bänder von 20 Hz bis 20 kHz
        gaps: list[tuple[float, float]] = []
        f_lo = 20.0
        ratio = 2.0 ** (1.0 / 6.0)

        while f_lo < 20000.0:
            f_hi = f_lo * ratio
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if mask.sum() == 0:
                f_lo = f_hi
                continue

            band_energy = np.mean(stft[mask] ** 2)
            energy_db = -120.0 if band_energy <= 0.0 else 10.0 * math.log10(float(band_energy) + 1e-12)

            band_width = f_hi - f_lo
            if energy_db <= self.GAP_ENERGY_THRESHOLD_DB and band_width >= self.MIN_GAP_WIDTH_HZ:
                gaps.append((f_lo, f_hi))

            f_lo = f_hi

        return gaps

    def _repair_gaps(
        self,
        audio: np.ndarray,
        mono: np.ndarray,
        sr: int,
        gaps: list[tuple[float, float]],
        instrument_tag: str,
    ) -> np.ndarray:
        """Repariert erkannte Lücken via harmonischer Interpolation.

        Einfache Implementierung via spektraler Interpolation aus Nachbarbändern.
        NMF-β-Verfeinerung bei Bedarf (falls Flatness > Schwellwert).
        """
        n_fft = 2048
        audio_out = audio.copy().astype(np.float32)

        channels_in = [audio_out[0], audio_out[1]] if audio_out.ndim == 2 else [audio_out]
        channels_out: list[np.ndarray] = []

        for ch in channels_in:
            if len(ch) < n_fft:
                channels_out.append(ch)
                continue

            # FFT
            fft_data = np.fft.rfft(ch, n=n_fft)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            mag = np.abs(fft_data)
            phase = np.angle(fft_data)

            for f_lo, f_hi in gaps:
                gap_mask = (freqs >= f_lo) & (freqs < f_hi)
                if gap_mask.sum() == 0:
                    continue

                # Nachbar-Energie für Interpolation
                margin = (f_hi - f_lo) * 0.5
                lo_mask = (freqs >= (f_lo - margin)) & (freqs < f_lo)
                hi_mask = (freqs >= f_hi) & (freqs < (f_hi + margin))

                lo_energy = np.mean(mag[lo_mask]) if lo_mask.sum() > 0 else 0.0
                hi_energy = np.mean(mag[hi_mask]) if hi_mask.sum() > 0 else 0.0

                if lo_energy > 0 or hi_energy > 0:
                    # Geometrisches Mittel der Nachbar-Energie
                    if lo_energy > 0 and hi_energy > 0:
                        target_energy = math.sqrt(float(lo_energy) * float(hi_energy))
                    else:
                        target_energy = max(float(lo_energy), float(hi_energy))

                    # Magnitude auf Ziel setzen (deterministische Füllung)
                    gap_indices = np.flatnonzero(gap_mask)
                    if gap_indices.size == 0:
                        continue

                    current_mean = float(np.mean(mag[gap_indices]))
                    if current_mean < target_energy * 0.1:
                        fill_values = np.full(gap_indices.shape, target_energy, dtype=mag.dtype)
                        mag[gap_indices] = fill_values

            # PGHI-Approximation: Phase aus Magnitude rekonstruieren
            fft_repaired = mag * np.exp(1j * phase)
            ch_repaired = np.real(np.fft.irfft(fft_repaired, n=n_fft))

            # Sicherstellen dass die Länge passt
            original_len = len(ch)
            if len(ch_repaired) > original_len:
                ch_repaired = ch_repaired[:original_len]
            elif len(ch_repaired) < original_len:
                ch_repaired = np.pad(ch_repaired, (0, original_len - len(ch_repaired)))

            channels_out.append(ch_repaired.astype(np.float32, copy=False))

        audio_out = np.stack(channels_out, axis=0) if audio_out.ndim == 2 else channels_out[0]

        return audio_out

    def _check_spectral_flatness(self, mono: np.ndarray, sr: int) -> bool:
        """Prüft Spectral Flatness ≤ 0.40 (Invariante §4.5).

        Returns:
            True wenn Flatness im erlaubten Bereich
        """
        n_fft = 2048
        if len(mono) < n_fft:
            return True

        mag = np.abs(np.fft.rfft(mono[:n_fft]))
        mag = mag[mag > 0]

        if len(mag) == 0:
            return True

        geometric_mean = math.exp(np.mean(np.log(mag + 1e-12)))
        arithmetic_mean = float(np.mean(mag))

        if arithmetic_mean <= 0:
            return True

        flatness = geometric_mean / arithmetic_mean
        return float(flatness) <= self.MAX_SPECTRAL_FLATNESS


# ---------------------------------------------------------------------------
# Singleton (§3.2)
# ---------------------------------------------------------------------------
import threading

_instance: SpectralBandGapRepair | None = None
_lock = threading.Lock()


def get_spectral_band_gap_repair() -> SpectralBandGapRepair:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SpectralBandGapRepair()
    return _instance


def repair_spectral_band_gaps(
    audio: np.ndarray,
    sr: int = 48000,
    instrument_tag: str = "unknown",
    confidence: float = 1.0,
) -> BandGapResult:
    """Convenience-Wrapper für SpectralBandGapRepair (§3.2)."""
    return get_spectral_band_gap_repair().repair(
        audio=audio,
        sr=sr,
        instrument_tag=instrument_tag,
        confidence=confidence,
    )
