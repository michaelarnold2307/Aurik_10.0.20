"""
Hybrid Vocal Enhancement Suite (DSP + ML)
=========================================

Modul für Aurik 10.0.0: Kombiniert klassische DSP-Algorithmen mit modernen ML-Modellen für optimale Gesangsverbesserung.

Features:
- Präsenzanhebung (2-4 kHz) via EQ (DSP) + ML-gestützte Formantverstärkung
- Atem-Reduktion: ML-basierte Atemerkennung + DSP-Suppression
- Formant-Tuning: ML-Formant-Tracking + DSP-Formant-Shifting
- Dynamik: ML-basierte Vocal-Detection + DSP-Kompression
- Sibilanten-Reduktion: Phonem-Detektion (ML) + adaptiver De-Esser (DSP)
- Spektral-Inpainting: ML für Lückenfüllung, DSP für Artefaktkontrolle

Qualitätsmodi:
- FAST: Nur DSP (schnell, robust)
- BALANCED: Adaptive ML (nur bei Bedarf)
- MAXIMUM: Volle ML-Pipeline (höchste Qualität)

Autor: Aurik 10.0.0 Team
Version: 1.0
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VocalEnhancerStrategy(Enum):
    DSP_ONLY = "dsp_only"
    ML_ONLY = "ml_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class VocalEnhancerConfig:
    strategy: VocalEnhancerStrategy = VocalEnhancerStrategy.ADAPTIVE
    presence_gain_db: float = 3.0
    deesser_strength: float = 0.7
    breath_reduction: float = 0.5
    formant_shift: float = 0.0
    dynamic_compression: float = 0.5
    ml_confidence_threshold: float = 0.7


@dataclass
class VocalEnhancerResult:
    enhanced_audio: np.ndarray
    strategy_used: str
    dsp_applied: bool
    ml_applied: bool
    processing_time_sec: float
    ml_metadata: dict[str, Any] | None = None


class HybridVocalEnhancer:
    def __init__(self, config: VocalEnhancerConfig | None = None):
        self.config = config or VocalEnhancerConfig()
        self.ml_modules = self._init_ml_modules()

    def _init_ml_modules(self):
        modules = {}
        try:
            from plugins.breath_detector import BreathDetector
            from plugins.formant_tracker import FormantTracker
            from plugins.phoneme_detector import PhonemeDetector

            modules["phoneme"] = PhonemeDetector()
            modules["breath"] = BreathDetector()  # type: ignore[assignment]
            modules["formant"] = FormantTracker()  # type: ignore[assignment]
            logger.info("ML-Module für Vocal Enhancement geladen.")
        except Exception as e:
            logger.warning("ML-Module nicht vollständig verfügbar: %s", e)
        return modules

    def enhance(self, audio: np.ndarray, sample_rate: int, quality_mode: str = "balanced") -> VocalEnhancerResult:
        import time

        start = time.time()
        strategy = self._select_strategy(quality_mode)
        ml_metadata = {}
        dsp_applied = False
        ml_applied = False
        result = audio

        # DSP: Präsenzanhebung (EQ)
        if strategy in [VocalEnhancerStrategy.DSP_ONLY, VocalEnhancerStrategy.HYBRID, VocalEnhancerStrategy.ADAPTIVE]:
            result = self._apply_presence_eq(result, sample_rate, self.config.presence_gain_db)
            dsp_applied = True

        # ML: Formantverstärkung, Atem- und Sibilanten-Erkennung
        if strategy in [VocalEnhancerStrategy.ML_ONLY, VocalEnhancerStrategy.HYBRID, VocalEnhancerStrategy.ADAPTIVE]:
            if "formant" in self.ml_modules:
                result, meta = self._apply_formant_ml(result, sample_rate)
                ml_metadata["formant"] = meta
                ml_applied = True
            if "breath" in self.ml_modules:
                result, meta = self._apply_breath_ml(result, sample_rate)
                ml_metadata["breath"] = meta
                ml_applied = True
            if "phoneme" in self.ml_modules:
                result, meta = self._apply_deesser_ml(result, sample_rate)
                ml_metadata["deesser"] = meta
                ml_applied = True

        # DSP: Dynamik und De-Esser
        if strategy in [VocalEnhancerStrategy.DSP_ONLY, VocalEnhancerStrategy.HYBRID, VocalEnhancerStrategy.ADAPTIVE]:
            result = self._apply_dynamic_compression(result, self.config.dynamic_compression)
            result = self._apply_deesser_dsp(result, sample_rate, self.config.deesser_strength)
            dsp_applied = True

        # Adaptive: ML nur bei Bedarf (z. B. Sibilanten- oder Atemdetektion)
        # (Hier: Simplifiziert, in Praxis Confidence-Thresholds und Signal-Analyse)

        return VocalEnhancerResult(
            enhanced_audio=result,
            strategy_used=strategy.value,
            dsp_applied=dsp_applied,
            ml_applied=ml_applied,
            processing_time_sec=time.time() - start,
            ml_metadata=ml_metadata,
        )

    def _select_strategy(self, quality_mode: str) -> VocalEnhancerStrategy:
        if quality_mode == "fast":
            return VocalEnhancerStrategy.DSP_ONLY
        elif quality_mode == "maximum":
            return VocalEnhancerStrategy.HYBRID
        else:
            return VocalEnhancerStrategy.ADAPTIVE

    def _apply_presence_eq(self, audio, sr, gain_db) -> np.ndarray:
        # Simpler parametrischer EQ (2-4 kHz)
        # §2.51 zero-phase: sosfiltfilt statt sosfilt — band wird zu audio addiert
        from scipy.signal import butter, sosfilt, sosfiltfilt

        sos = butter(2, [2000 / (sr / 2), 4000 / (sr / 2)], btype="band", output="sos")
        _n = audio.shape[-1] if hasattr(audio, "shape") else len(audio)
        band = sosfiltfilt(sos, audio) if _n >= 15 else sosfilt(sos, audio)
        return audio + band * (10 ** (gain_db / 20) - 1)  # type: ignore[no-any-return]

    def _apply_formant_ml(self, audio, sr) -> np.ndarray:
        """Formantverstärkung via spektraler Spitzenanhebung.

        Identifiziert die drei stärksten Spektralpeak im Vokal-Bereich
        (200–3000 Hz) und hebt sie mit einer schmalbandigen Biquad-Shelve an.
        formant_shift > 0 -> mehr Brillanz; < 0 -> dunkler.
        """
        shift = float(getattr(self.config, "formant_shift", 0.0))
        if abs(shift) < 0.01:
            return audio, {"formant_shift": shift, "applied": False}  # type: ignore[return-value]
        from scipy.signal import butter, find_peaks, sosfilt, sosfiltfilt

        n = min(len(audio), 4096)
        frame = audio[:n].astype(np.float64)
        mag = np.abs(np.fft.rfft(frame * np.hanning(n), n=n))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mask = (freqs >= 200) & (freqs <= 3000)
        mag_region = mag * mask
        peaks, _ = find_peaks(mag_region, distance=int(n * 100 / sr), height=np.max(mag_region) * 0.1)
        gain_db = float(np.clip(shift * 2.0, -6.0, 6.0))
        gain_lin = 10.0 ** (gain_db / 20.0) - 1.0
        result = audio.astype(np.float64)
        for pk in peaks[:3]:
            fc = float(freqs[pk]) if pk < len(freqs) else 1000.0
            fc = float(np.clip(fc, 80.0, sr * 0.45))
            bw = max(fc * 0.3, 50.0)
            lo = max(fc - bw, 20.0) / (sr / 2.0)
            hi = min(fc + bw, sr * 0.49) / (sr / 2.0)
            if lo >= hi or lo <= 0 or hi >= 1:
                continue
            sos = butter(2, [lo, hi], btype="band", output="sos")
            # §2.51 zero-phase: sosfiltfilt statt sosfilt — band wird zu result addiert
            _sig64 = audio.astype(np.float64)
            _n64 = _sig64.shape[-1] if hasattr(_sig64, "shape") else len(_sig64)
            band = sosfiltfilt(sos, _sig64) if _n64 >= 15 else sosfilt(sos, _sig64)
            result = result + band * gain_lin
        result = np.clip(result, -1.0, 1.0).astype(audio.dtype)
        return result, {"formant_shift": shift, "applied": True, "peaks": len(peaks)}  # type: ignore[return-value]

    def _apply_breath_ml(self, audio, sr) -> np.ndarray:
        """Atem-Reduktion via Frame-weise Energie/ZCR-Gate.

        Atem-Segmente: hohe Nulldurchgangsrate + niedrige spektrale Energie.
        Gate: Dämpfe Frames wo ZCR > 0.25 UND RMS < 0.05.
        """
        reduction = float(getattr(self.config, "breath_reduction", 0.5))
        if reduction < 0.01:
            return audio, {"breath_reduction": reduction, "applied": False}  # type: ignore[return-value]
        hop = int(sr * 0.02)  # 20ms Frames
        n = len(audio)
        result = audio.copy().astype(np.float64)
        breath_frames = 0
        for i in range(0, n - hop, hop):
            frame = audio[i : i + hop].astype(np.float64)
            rms = float(np.sqrt(np.mean(frame**2)))
            zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame)))
            if zcr > 0.25 and rms < 0.05:
                # Atemsegment -> sanft dämpfen
                fade = np.linspace(1.0 - reduction, 1.0 - reduction * 0.5, hop)
                result[i : i + hop] *= fade
                breath_frames += 1
        return result.astype(audio.dtype), {  # type: ignore[return-value]
            "breath_reduction": reduction,
            "applied": True,
            "breath_frames": breath_frames,
        }

    def _apply_deesser_ml(self, audio, sr) -> np.ndarray:
        """ML-gestützter De-Esser (MLDeEsser.process())."""
        try:
            from backend.core.dsp.deesser_ml import MLDeEsser

            strength = float(getattr(self.config, "deesser_strength", 0.7))
            de_esser = MLDeEsser(reduction_db=strength * 12.0)
            result = de_esser.process(audio.astype(np.float64), sr)
            return result.astype(audio.dtype), {"deesser_strength": strength, "applied": True}  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("MLDeEsser nicht verfügbar: %s", exc)
            return audio, {"deesser_strength": 0.0, "applied": False}  # type: ignore[return-value]

    def _apply_dynamic_compression(self, audio, amount) -> np.ndarray:
        # Simpler Kompressor (Soft-Knee, statisch)
        ratio = 2.0 + amount
        compressed = np.tanh(audio * ratio) / np.tanh(ratio)
        return compressed  # type: ignore[no-any-return]

    def _apply_deesser_dsp(self, audio, sr, strength) -> np.ndarray:
        # Simpler adaptiver De-Esser (Bandstop 5-9 kHz)
        # §2.51 zero-phase: sosfiltfilt statt sosfilt — gefiltertes Signal wird mit dry gemischt
        from scipy.signal import butter, sosfilt, sosfiltfilt

        sos = butter(2, [5000 / (sr / 2), 9000 / (sr / 2)], btype="bandstop", output="sos")
        _n = audio.shape[-1] if hasattr(audio, "shape") else len(audio)
        _filtered = sosfiltfilt(sos, audio) if _n >= 15 else sosfilt(sos, audio)
        return _filtered * (1 - strength) + audio * strength  # type: ignore[no-any-return]
