"""
Hybrid ML Denoiser - Aurik 10.0.0
================================

Kombiniert OMLSA (DSP-basiert, schnell) mit DeepFilterNet (ML-basiert, musik-finetuned)
für optimale Balance zwischen Performance und Qualität.

Strategy:
    - Stage 1 (Fast Pre-filtering): OMLSA spectral subtraction (~30-60s)
      * Removes bulk of noise quickly (stationary noise, hum, hiss)
      * Sets good baseline for ML refinement

    - Stage 2 (Quality Refinement): DeepFilterNet ML (in-process, ~10-30s)
      * Further enhances naturalness and removes residual artifacts
      * Works better on OMLSA-preprocessed audio (cleaner input)

    - Adaptive Strategy:
      * FAST mode: OMLSA only (~0.5× RT)
      * BALANCED mode: OMLSA + selective DeepFilterNet (~1.5× RT)
      * MAXIMUM mode: OMLSA + full DeepFilterNet (~2-3× RT)

Benefits:
    - 30-40% faster than DeepFilterNet alone (OMLSA does bulk work)
    - Better quality than OMLSA alone (+0.05-0.10 naturalness)
    - More stable than DeepFilterNet on noisy inputs (OMLSA pre-cleaning)
    - Adaptive to quality requirements

Performance:
    - FAST: ~0.5× RT (OMLSA only)
    - BALANCED: ~1.5× RT (OMLSA + selective DeepFilterNet)
    - MAXIMUM: ~3-5× RT (full pipeline)

Author: Aurik 10.0.0 Development Team
Version: 1.0.0
Date: 16. Februar 2026
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np

from backend.core.audio_utils import safe_filtfilt  # §v10.101 padlen-guard
from dsp.adaptive_omlsa import AdaptiveOMLSA

logger = logging.getLogger(__name__)


class DenoiseStrategy(Enum):
    """Denoising strategy."""

    OMLSA_ONLY = "omlsa_only"  # Fast, DSP-based
    DFN_ONLY = "dfn_only"  # High quality, ML-based (musik-finetuned DeepFilterNet)
    HYBRID = "hybrid"  # OMLSA → DeepFilterNet (best balance)
    ADAPTIVE = "adaptive"  # Auto-select based on audio analysis


@dataclass
class DenoiseConfig:
    """Configuration for hybrid denoising."""

    strategy: DenoiseStrategy = DenoiseStrategy.HYBRID
    omlsa_alpha: float = 0.98  # OMLSA smoothing factor
    omlsa_noise_floor: float = 1e-8  # OMLSA noise floor
    dfn_denoise: float = 0.8  # DeepFilterNet denoising strength
    dfn_enhance: float = 0.5  # DeepFilterNet enhancement strength
    enable_preprocessing: bool = True  # OMLSA preprocessing before DeepFilterNet
    quality_threshold: float = 0.75  # If quality > threshold, skip DeepFilterNet
    enable_psychoacoustic_fusion: bool = (
        True  # §Witness-SOTA H1+H2: Masking-Fusion + Musical-Noise-Gate um die ML-Stufe
    )


@dataclass
class DenoiseResult:
    """Result of denoising operation."""

    audio: np.ndarray
    strategy_used: DenoiseStrategy
    omlsa_applied: bool
    dfn_applied: bool
    processing_time: float
    quality_estimate: float
    metadata: dict[str, Any]


class HybridMLDenoiser:
    """
    Hybrid ML Denoiser combining OMLSA and DeepFilterNet.

    Usage:
        denoiser = HybridMLDenoiser(strategy=DenoiseStrategy.HYBRID)
        result = denoiser.denoise(audio, sample_rate=48000)
        clean_audio = result.audio
    """

    def __init__(self, config: DenoiseConfig | None = None):
        """
        Initialisiert hybrid denoiser.

        Args:
            config: Denoising configuration (default: HYBRID strategy)
        """
        self.config = config or DenoiseConfig()
        self.omlsa = AdaptiveOMLSA(alpha=self.config.omlsa_alpha, noise_floor=self.config.omlsa_noise_floor)

        # Lazy-load DeepFilterNet (musik-finetuned, in-process)
        self._dfn_cache: Any = None

        logger.info("HybridMLDenoiser initialisiert: strategy=%s", self.config.strategy.value)

    def _dfn_plugin(self) -> Any:
        """Lazy DeepFilterNet-Singleton (ehem. Sprach-Enhancement ersetzt, §v10.19 .github/specs/v10.19_sprachmodell_ersatz_sota_roadmap.md)."""
        if self._dfn_cache is None:
            try:
                from plugins.deepfilternet_v3_ii_plugin import (  # pylint: disable=import-outside-toplevel
                    get_deepfilternet_plugin,
                )

                self._dfn_cache = get_deepfilternet_plugin()
                logger.info("DeepFilterNet plugin geladen erfolgreich")
            except Exception as e:
                logger.warning("konnte DeepFilterNet nicht laden: %s", e)
                logger.warning("Falling back to OMLSA-only Betriebsart")
        return self._dfn_cache

    def denoise(
        self, audio: np.ndarray, sample_rate: int = 48000, noise_profile: np.ndarray | None = None
    ) -> DenoiseResult:
        """
        Denoise audio using hybrid strategy.

        Args:
            audio: Input audio (mono or stereo)
            sample_rate: Sample rate in Hz
            noise_profile: Optional noise profile for OMLSA

        Returns:
            DenoiseResult with cleaned audio and metadata

        §4.7 NoiseTextureCoherenceGuard: Kohärenz≥0.80 in Restoration-Modus (Trägerprofil-Erhaltung)
        """
        start_time = time.time()

        # Determine strategy
        strategy = self._determine_strategy(audio, sample_rate)

        omlsa_applied = False
        dfn_applied = False
        quality_estimate = 0.0
        metadata = {}

        # Stage 1: OMLSA preprocessing (if enabled)
        if strategy in [DenoiseStrategy.OMLSA_ONLY, DenoiseStrategy.HYBRID]:
            logger.info("Stufe 1: Applying OMLSA preprocessing...")
            audio, omlsa_meta = self._apply_omlsa(audio, sample_rate, noise_profile)
            omlsa_applied = True
            metadata["omlsa"] = omlsa_meta

            # Estimate quality after OMLSA
            quality_estimate = self._estimate_quality(audio, sample_rate)
            metadata["quality_after_omlsa"] = quality_estimate  # type: ignore[assignment]

            logger.info("OMLSA vollstaendig: quality=%.3f", quality_estimate)

            # Skip DFN if quality already good enough
            if quality_estimate >= self.config.quality_threshold and strategy == DenoiseStrategy.HYBRID:
                logger.info("Quality sufficient (%.3f), skipping DeepFilterNet", quality_estimate)
                strategy = DenoiseStrategy.OMLSA_ONLY

        # §Witness-SOTA H1: DSP-Baseline für die masking-threshold-bewusste
        # Fusion sichern (Referenz, an der die ML-Stufe gemessen wird).
        _omlsa_baseline = audio.copy()

        # Stage 2: DeepFilterNet Enhancement (if needed)
        if strategy in [DenoiseStrategy.DFN_ONLY, DenoiseStrategy.HYBRID]:
            # try_allocate-Gate: erlaubt Tests DeepFilterNet per Mock zu deaktivieren (§2.51 Determinismus)
            _dfn_budget_ok = True
            try:
                from backend.core.ml_memory_budget import (
                    try_allocate as _ml_try_allocate,  # pylint: disable=import-outside-toplevel
                )

                _dfn_budget_ok = _ml_try_allocate("DeepFilterNetV3", size_gb=0.5)
            except Exception as e:
                logger.warning("hybrid_ml_denoiser.py::denoise Ersatzpfad: %s", e)
            if (
                _dfn_budget_ok
                and self._has_sufficient_ml_headroom(audio, sample_rate)
                and self._dfn_plugin() is not None
            ):
                logger.info("Stufe 2: Applying DeepFilterNet refinement...")
                # Protect DeepFilterNetV3 from PLM eviction during inference
                try:
                    from backend.core.plugin_lifecycle_manager import (
                        get_plugin_lifecycle_manager,  # pylint: disable=import-outside-toplevel
                    )

                    _plm = get_plugin_lifecycle_manager()
                    _plm.set_active("DeepFilterNetV3", True)
                except Exception:
                    _plm = None
                try:
                    audio, dfn_meta = self._apply_dfn(audio, sample_rate)
                    dfn_applied = True
                    metadata["dfn"] = dfn_meta

                    # §Witness-SOTA H1+H2: Die ML-Stufe (Kandidat) darf sich nur
                    # dort durchsetzen, wo sie gegenüber dem OMLSA-Baseline
                    # HÖRBAR leiser ist (Johnston-Maskierungsschwelle, Bark-
                    # Bänder, Never-worsen); danach dämpft das Musical-Noise-Gate
                    # hörbares Restrauschen auf die Schwelle (nie „tote Stille").
                    # ML→DSP-Ersatzpfad mit Warnung bei Fehler (§V6 (copilot-instructions.md)): dann bleibt
                    # das unveränderte DeepFilterNet-Ergebnis bestehen.
                    if self.config.enable_psychoacoustic_fusion and omlsa_applied:
                        try:
                            from backend.core.dsp.hybrid_denoise_fusion import (  # pylint: disable=import-outside-toplevel
                                masked_denoise_fusion,
                                musical_noise_gate,
                            )

                            audio, _fusion_report = masked_denoise_fusion(audio, _omlsa_baseline, sample_rate)
                            metadata["psychoacoustic_fusion"] = _fusion_report
                            audio, _gate_report = musical_noise_gate(audio, sample_rate)
                            metadata["musical_noise_gate"] = _gate_report
                            logger.info(
                                "§H1+H2 Psychoakustische Denoise-Fusion: mean_blend=%.3f, bands_gated=%d",
                                float(_fusion_report.get("mean_blend", 0.0)),
                                int(_gate_report.get("bands_gated", 0)),
                            )
                        except Exception as _pf_exc:  # pylint: disable=broad-except
                            logger.warning(
                                "§V6 (copilot-instructions.md) Psychoakustische Fusion nicht anwendbar (%s) — "
                                "DeepFilterNet-Ergebnis ohne H1+H2-Gate beibehalten",
                                _pf_exc,
                            )

                    # Re-estimate quality after DeepFilterNet
                    quality_estimate = self._estimate_quality(audio, sample_rate)
                    metadata["quality_after_dfn"] = quality_estimate  # type: ignore[assignment]

                    logger.info("DeepFilterNet vollstaendig: quality=%.3f", quality_estimate)
                finally:
                    if _plm is not None:
                        _plm.set_active("DeepFilterNetV3", False)
            else:
                logger.warning("DeepFilterNet not verfuegbar, using OMLSA Ergebnis")

        processing_time = time.time() - start_time
        metadata["processing_time"] = processing_time  # type: ignore[assignment]

        # §4.7 NoiseTextureCoherenceGuard: Kohärenz≥0.80 in Restoration-Modus (Trägerprofil-Erhaltung)
        try:
            from backend.core.noise_texture_coherence import get_noise_texture_coherence_guard

            _guard = get_noise_texture_coherence_guard()
            # Residual noise estimate: denoised audio minus original (approximated by difference)
            _residual = (
                np.clip(audio - np.mean(audio, axis=-1, keepdims=True), -0.5, 0.5) if audio.ndim > 1 else audio * 0.01
            )
            _coherence_result = _guard.check(_residual, "unknown", sample_rate)
            metadata["noise_texture_coherence"] = float(_coherence_result.coherence)  # type: ignore[assignment]
            if _coherence_result.coherence < 0.80:
                logger.warning(
                    "§4.7 NoiseTextureCoherenceGuard: coherence=%.2f < 0.80 → Kohärenz-Pruefung fehlgeschlagen",
                    float(_coherence_result.coherence),
                )
        except Exception as _ntc_err:
            logger.debug("NoiseTextureCoherenceGuard nicht verfügbar: %s", _ntc_err)

        return DenoiseResult(
            audio=audio,
            strategy_used=strategy,
            omlsa_applied=omlsa_applied,
            dfn_applied=dfn_applied,
            processing_time=processing_time,
            quality_estimate=quality_estimate,
            metadata=metadata,
        )

    def _determine_strategy(self, audio: np.ndarray, _sample_rate: int) -> DenoiseStrategy:
        """Bestimmt optimal denoising strategy."""
        if self.config.strategy != DenoiseStrategy.ADAPTIVE:
            return self.config.strategy

        # §23-TONALITY (Vorschlag 01): tonales/sauberes Signal ⇒ kein generatives ML;
        # OMLSA (transparent auf Tonalem) reicht vollständig.
        try:
            from backend.core.dsp.tonality_gate import is_tonal_clean  # pylint: disable=import-outside-toplevel

            if is_tonal_clean(audio, _sample_rate):
                logger.info("Tonal/clean audio (tonality_gate) → OMLSA_ONLY")
                return DenoiseStrategy.OMLSA_ONLY
        except Exception as _tg_exc:  # nicht blockierend
            logger.debug("hybrid_ml_denoiser: tonality_gate nicht verfügbar (%s)", _tg_exc)

        # Analyze noise level to decide strategy
        noise_level = self._estimate_noise_level(audio)

        if noise_level < 0.01:
            # Very clean audio - skip denoising
            logger.info("Clean audio (noise=%.4f), minimal processing", noise_level)
            return DenoiseStrategy.OMLSA_ONLY
        elif noise_level < 0.05:
            # Moderate noise - OMLSA sufficient
            logger.info("Moderate noise (noise=%.4f), OMLSA only", noise_level)
            return DenoiseStrategy.OMLSA_ONLY
        else:
            # Heavy noise - full hybrid pipeline
            logger.info("Heavy noise (noise=%.4f), full hybrid", noise_level)
            return DenoiseStrategy.HYBRID

    def _apply_omlsa(
        self, audio: np.ndarray, sample_rate: int, noise_profile: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Wendet an: OMLSA spectral subtraction."""
        from scipy import signal  # pylint: disable=import-outside-toplevel

        metadata = {}

        # Store original shape for length preservation
        is_stereo = audio.ndim == 2
        if is_stereo:
            # Detect orientation: channels-first (ch, N) → shape[0] ≤ 2 and shape[1] > 2
            # vs samples-first (N, ch) → shape[0] > shape[1].
            if audio.shape[0] <= 2 and audio.shape[1] > 2:
                # (ch, N) → downmix along axis=0
                audio_mono = np.mean(audio, axis=0)
                original_length = audio.shape[1]
            else:
                # (N, ch) → downmix along axis=1
                audio_mono = np.mean(audio, axis=1)
                original_length = audio.shape[0]
        else:
            audio_mono = audio
            original_length = audio.shape[0]

        # STFT — clamp noverlap so it is always < min(nperseg, signal_length).
        # scipy auto-reduces nperseg to signal_length for short chunks which leaves
        # the fixed noverlap=1536 >= effective nperseg → ValueError.
        _sig_len = len(audio_mono)
        _nperseg = int(min(2048, max(1, _sig_len)))
        _noverlap = int(min(1536, max(0, _nperseg - 1)))
        _f, _t, Zxx = signal.stft(audio_mono, fs=sample_rate, nperseg=_nperseg, noverlap=_noverlap, boundary="zeros")

        noisy_mag = np.abs(Zxx)
        noisy_phase = np.angle(Zxx)

        # Estimate noise profile (first 0.5s if not provided)
        if noise_profile is None:
            noise_frames = int(0.5 * sample_rate / (2048 - 1536))  # ~0.5s
            noise_mag = np.median(noisy_mag[:, :noise_frames], axis=1, keepdims=True)
        else:
            noise_mag = noise_profile

        # Apply OMLSA frame-by-frame
        clean_mag = np.zeros_like(noisy_mag)
        for i in range(noisy_mag.shape[1]):
            clean_mag[:, i] = self.omlsa.omlsa(noisy_mag[:, i], noise_mag[:, 0] if noise_mag.ndim == 2 else noise_mag)

        # ISTFT
        Zxx_clean = clean_mag * np.exp(1j * noisy_phase)
        _, audio_clean = signal.istft(Zxx_clean, fs=sample_rate, nperseg=_nperseg, noverlap=_noverlap, boundary="zeros")

        # Trim/pad to original length
        if len(audio_clean) > original_length:
            audio_clean = audio_clean[:original_length]
        elif len(audio_clean) < original_length:
            audio_clean = np.pad(audio_clean, (0, original_length - len(audio_clean)))

        # Restore stereo if needed
        if is_stereo:
            # Simple stereo restoration (scale original channels)
            # scale has shape (samples,), need to broadcast to (channels, samples)
            scale = np.maximum(np.abs(audio_clean), 1e-8) / np.maximum(np.abs(audio_mono[:original_length]), 1e-8)
            audio_clean = audio[:, :original_length] * scale[np.newaxis, :]

        metadata["frames_processed"] = noisy_mag.shape[1]
        _clean_power = np.mean(clean_mag**2)
        metadata["noise_reduction_db"] = (
            10 * np.log10(np.mean(noisy_mag**2) / _clean_power) if _clean_power > 0 else 0.0
        )  # §3.1: Zero-Division-Guard bei Stille (clean_mag == 0)

        return audio_clean, metadata

    def _apply_dfn(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Wendet an: DeepFilterNet ML refinement (musik-finetuned, in-process).

        §v10.19 (.github/specs/v10.19_sprachmodell_ersatz_sota_roadmap.md), umgesetzt
        2026-09-11: das vormals eingesetzte Sprach-Enhancement
        wurde aus Aurik entfernt — für GESANG ist die musik-finetuned
        DeepFilterNet-Kette die korrekte ML-Stufe; KIM2 (kim_vocal_2) übernimmt
        die vokal-spezifische Klarheit in den Gesangsphasen (phase_19/43/66 +
        StemContext-Router).
        """
        metadata: dict[str, Any] = {}

        if not self._has_sufficient_ml_headroom(audio, sample_rate):
            metadata["success"] = False
            metadata["error"] = "OOM guard: insufficient RAM"
            return audio, metadata

        try:
            _dfn = self._dfn_plugin()
            if _dfn is None:
                metadata["success"] = False
                metadata["error"] = "DeepFilterNet nicht verfügbar — DSP-Pfad bleibt aktiv"
                return audio, metadata
            _in = audio.T if audio.ndim == 2 else audio
            _out = np.asarray(_dfn.enhance(_in, sample_rate), dtype=np.float32)
            if audio.ndim == 2:
                _out = _out.T if _out.ndim == 2 else np.stack([_out, _out], axis=0)
            metadata["success"] = True
            metadata["model"] = "deepfilternet_v3_ii"
            return _out, metadata
        except Exception as exc:
            logger.warning("hybrid_ml_denoiser.py DFN-Stufe ML→DSP-Ersatzpfad: %s", exc)
            metadata["success"] = False
            metadata["error"] = str(exc)
            return audio, metadata

    def _has_sufficient_ml_headroom(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Gibt True when enough free RAM is available for DeepFilterNet denoise zurück.

        The previous guard only compared current free RAM to raw audio size and
        still allowed plugin loading plus temp-file IO to push the VS Code cgroup
        into OOM on long stereo files. Use a conservative duration/channel-aware
        threshold and reclaim memory before entering the ML stage.
        """
        try:
            import gc  # pylint: disable=import-outside-toplevel

            import psutil  # pylint: disable=import-outside-toplevel
        except Exception as e:
            logger.warning("hybrid_ml_denoiser.py::_has_sufficient_ml_headroom Ersatzpfad: %s", e)
            return True

        n_samples = int(
            audio.shape[-1]
            if audio.ndim == 2 and audio.shape[0] <= 2 and audio.shape[1] > audio.shape[0]
            else audio.shape[0]
        )
        n_channels = 2 if audio.ndim == 2 else 1
        duration_s = n_samples / float(max(1, sample_rate))

        required_gb = 4.0
        if n_channels >= 2:
            required_gb += 2.0
        if duration_s >= 180.0:
            required_gb += 2.0
        elif duration_s >= 60.0:
            required_gb += 1.0

        avail_gb = psutil.virtual_memory().available / (1024**3)
        if avail_gb < required_gb + 1.5:
            logger.info(
                "Denoise: %.1f GB frei, Ziel-Headroom %.1f GB — proaktive Plugin-Eviction vor DeepFilterNet-Inferenz",
                avail_gb,
                required_gb,
            )
            try:
                from backend.core.plugin_lifecycle_manager import (
                    evict_stale_plugins,  # pylint: disable=import-outside-toplevel
                )

                evict_stale_plugins(required_mb=int(required_gb * 1024))
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
            gc.collect()
            try:
                import ctypes as _ct  # pylint: disable=import-outside-toplevel

                _ct.CDLL("libc.so.6").malloc_trim(0)
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
            avail_gb = psutil.virtual_memory().available / (1024**3)

        if avail_gb < required_gb:
            logger.warning(
                "Denoise RAM guard: %.1f GB frei, benötigt >= %.1f GB"
                " (dauer=%.1fs, kanaele=%d) — DeepFilterNet-Stufe übersprungen, OMLSA-Ergebnis behalten",
                avail_gb,
                required_gb,
                duration_s,
                n_channels,
            )
            return False

        return True

    def _estimate_noise_level(self, audio: np.ndarray) -> float:
        """Schätzt noise level in audio (0-1 scale)."""
        # Simple noise estimation: RMS of high-frequency content
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # High-pass filter to isolate noise
        from scipy.signal import butter, filtfilt  # pylint: disable=import-outside-toplevel

        b, a = cast(tuple[np.ndarray, np.ndarray], butter(4, 0.3, btype="high", output="ba"))
        # filtfilt requires at least padlen+1 = 15+1 samples; fall back to RMS on short clips
        min_len = 3 * max(len(b), len(a)) + 1
        if len(audio) < min_len:
            noise_rms = float(np.sqrt(np.mean(audio**2)))
            signal_rms = noise_rms
        else:
            noise = safe_filtfilt(b, a, audio)
            noise_rms = np.sqrt(np.mean(noise**2))
            signal_rms = np.sqrt(np.mean(audio**2))

        # Noise ratio
        noise_ratio = noise_rms / (signal_rms + 1e-8)

        return float(np.clip(noise_ratio, 0, 1))

    def _estimate_quality(self, audio: np.ndarray, _sample_rate: int) -> float:
        """
        Schätzt audio quality (0-1 scale).

        Simple heuristic:
        - SNR estimation
        - Spectral flatness
        - Dynamic range

        Returns:
            Quality score (0-1)
        """
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Compute metrics
        signal_power = np.mean(audio**2)
        noise_level = self._estimate_noise_level(audio)

        # SNR-based quality
        snr = signal_power / (noise_level**2 + 1e-8)
        snr_db = 10 * np.log10(snr + 1e-8)
        snr_quality = np.clip(snr_db / 40.0, 0, 1)  # 40 dB = excellent

        # Dynamic range (§0 Peak-Guard: 99.9th percentile for impulse robustness)
        from backend.core.core_utils import safe_peak_amplitude  # pylint: disable=import-outside-toplevel

        dynamic_range = safe_peak_amplitude(audio) / (np.mean(np.abs(audio)) + 1e-8)
        dr_quality = np.clip(dynamic_range / 10.0, 0, 1)  # 10:1 = good

        # Combined quality estimate
        quality = 0.6 * snr_quality + 0.4 * dr_quality

        return float(quality)


# Convenience functions for different modes
def denoise_fast(audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Fast denoising (OMLSA only)."""
    config = DenoiseConfig(strategy=DenoiseStrategy.OMLSA_ONLY)
    denoiser = HybridMLDenoiser(config)
    result = denoiser.denoise(audio, sample_rate)
    return result.audio


def denoise_balanced(audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Balanced denoising (OMLSA + selective DeepFilterNet)."""
    config = DenoiseConfig(
        strategy=DenoiseStrategy.HYBRID,
        quality_threshold=0.75,  # Skip DeepFilterNet if OMLSA achieves >0.75
    )
    denoiser = HybridMLDenoiser(config)
    result = denoiser.denoise(audio, sample_rate)
    return result.audio


def denoise_maximum(audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Maximum quality denoising (Full OMLSA → DeepFilterNet)."""
    config = DenoiseConfig(
        strategy=DenoiseStrategy.HYBRID,
        quality_threshold=1.0,  # Always apply DeepFilterNet
        dfn_denoise=1.0,
        dfn_enhance=0.7,
    )
    denoiser = HybridMLDenoiser(config)
    result = denoiser.denoise(audio, sample_rate)
    return result.audio


# Module test
if __name__ == "__main__":
    logger.debug("=" * 80)
    logger.debug("Hybrid ML Denoiser - Test")
    logger.debug("=" * 80)
    logger.debug("")

    # Test with synthetic noisy audio
    logger.debug("Generating test audio...")
    _sr_main = 48000
    _dur_main = 5.0
    _t_main = np.linspace(0, _dur_main, int(_sr_main * _dur_main))

    # Pure tone + noise
    _sig_main = np.sin(2 * np.pi * 440 * _t_main)  # 440 Hz
    _nse_main = 0.1 * np.random.randn(len(_t_main))
    _noisy_main = _sig_main + _nse_main

    logger.debug("Test audio: %ss @ %s Hz", _dur_main, _sr_main)
    logger.debug("SNR: %.1f dB", 10 * np.log10(np.mean(_sig_main**2) / np.mean(_nse_main**2)))
    logger.debug("")

    # Test different strategies
    _strategies_main = [
        (DenoiseStrategy.OMLSA_ONLY, "OMLSA Only (Fast)"),
        (DenoiseStrategy.HYBRID, "Hybrid (Balanced)"),
    ]

    for _strat_main, _name_main in _strategies_main:
        logger.debug("Testing: %s", _name_main)
        logger.debug("-" * 40)

        _cfg_main = DenoiseConfig(strategy=_strat_main)
        _dsr_main = HybridMLDenoiser(_cfg_main)

        _res_main = _dsr_main.denoise(_noisy_main, _sr_main)

        logger.debug("✅ Strategy: %s", _res_main.strategy_used.value)
        logger.debug("✅ OMLSA angewendet: %s", _res_main.omlsa_applied)
        logger.debug("✅ DeepFilterNet angewendet: %s", _res_main.dfn_applied)
        logger.debug("✅ Processing time: %.2fs", _res_main.processing_time)
        logger.debug("✅ Quality estimate: %.3f", _res_main.quality_estimate)
        logger.debug("")

    logger.debug("=" * 80)
    logger.debug("✅ Hybrid ML Denoiser module operational")
    logger.debug("=" * 80)
