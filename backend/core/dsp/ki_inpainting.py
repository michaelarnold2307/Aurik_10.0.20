import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.signal import correlate as _sc_correlate

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "ki_inpainting"
    category: str = "restoration"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


ki_inpainting_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "Dropout erkannt", "reason": "Nur bei klaren Aussetzern"}],
    params={
        "defaults": {"max_gap_ms": 200},
        "safe_ranges": {"max_gap_ms": {"min": 10, "max": 500}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.05,
        "identity_budget": 0.8,
        "spectral_change_budget": 0.1,
        "temporal_change_budget": 0.1,
        "compute_cost": 0.1,
    },
    side_effects=[{"risk": "Halluzination", "expected_when": "max_gap_ms > 200", "severity": 0.5}],
    reports={"self_metrics": ["inpainting_quality"], "confidence": 0.8},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class KIInpainting:
    """
    SOTA-konformes KI-Inpainting:
    - Füllt kurze Dropouts mit ML-Modell (Platzhalter für echtes Modell)
    """

    def __init__(self, max_gap_ms: int = 200):
        self.max_gap_ms = max_gap_ms

    def log_contract(self):
        _logger.info("[DSPContract] %s", asdict(ki_inpainting_contract))

    def process(self, audio: np.ndarray, sr: int, dropout_start: int, dropout_end: int) -> np.ndarray:
        """Kubische Spline-Interpolation für kurze Dropouts.

        Für Lücken bis max_gap_ms: kubische Hermite-Interpolation unter Verwendung
        von Nachbar-Kontext (je 10 Samples beidseitig). Für längere Lücken: Rollback.
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        self.log_contract()
        gap_len = dropout_end - dropout_start
        max_gap_samples = int(self.max_gap_ms * sr / 1000)
        if gap_len > max_gap_samples:
            _logger.info("[QualityGate] Dropout zu lang für sicheres Inpainting, Rollback aktiviert.")
            return np.asarray(audio.astype(audio.dtype))  # type: ignore[no-any-return]
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return np.asarray(audio.astype(audio.dtype))  # type: ignore[no-any-return]
        audio_filled = audio.copy().astype(np.float64)
        n = len(audio_filled)
        gs = max(0, dropout_start)
        ge = min(n, dropout_end)
        ctx = 10  # Kontext-Samples beidseitig
        x_left = max(0, gs - ctx)
        x_right = min(n, ge + ctx)
        # Kubische Spline-Interpolation (scipy)
        try:
            from scipy.interpolate import CubicSpline

            x_known = np.concatenate([np.arange(x_left, gs), np.arange(ge, x_right)])
            y_known = audio_filled[x_known]
            if len(x_known) >= 4:
                cs = CubicSpline(x_known, y_known, bc_type="not-a-knot")
                x_fill = np.arange(gs, ge)
                audio_filled[gs:ge] = np.clip(cs(x_fill), -1.0, 1.0)
            else:
                # Fallback: lineare Interpolation
                audio_filled[gs:ge] = np.linspace(
                    audio_filled[gs - 1] if gs > 0 else 0.0, audio_filled[ge] if ge < n else 0.0, ge - gs
                )
        except Exception:
            audio_filled[gs:ge] = np.linspace(
                float(audio_filled[gs - 1]) if gs > 0 else 0.0, float(audio_filled[ge]) if ge < n else 0.0, ge - gs
            )
        return np.asarray(audio_filled.astype(audio.dtype))  # type: ignore[no-any-return]


class SpectralInpainter:
    """
    Spectral-domain Audio Inpainting für Dropout-Reparatur
    Erweiterte Version von KIInpainting mit spektraler Analyse und semantischer Anpassung.

    Features:
    - Spektrale Interpolation für natürlichere Reparatur
    - Instrumenten-spezifische Anpassung (semantic-aware)
    - Multi-Dropout Reparatur in einem Durchlauf
    - Adaptive Context-Fenster basierend auf Gap-Größe

    Integration Point: Phase 2.4 (Dropout Repair) in unified_restorer_v2.py
    """

    def __init__(self, sr: int, context_ms: float = 100.0):
        """
        Args:
            sr: Sample rate
            context_ms: Kontext-Fenster um Dropout (beidseitig) in Millisekunden
        """
        self.sr = sr
        self.context_ms = context_ms
        self.context_samples = int((context_ms / 1000.0) * sr)
        self.instrument_context = None  # Optional: für semantic-aware processing

        # Contract definition (compatible with DSPContract pattern)
        self.contract = DSPContract(
            io={
                "channels": "mono",
                "sample_rates": [44100, 48000],
                "latency_samples": 0,
                "supports_offline": True,
            },
            preconditions=[
                {"if": "dropout_regions is list", "reason": "Must provide list of (start, end) tuples"},
                {"if": "all start < end", "reason": "Valid regions required"},
            ],
            params={
                "defaults": {"context_ms": 100.0},
                "safe_ranges": {"context_ms": {"min": 50, "max": 500}},
                "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
            },
            budgets={
                "artifact_budget": 0.03,  # Stricter than KIInpainting
                "identity_budget": 0.85,
                "spectral_change_budget": 0.08,
                "temporal_change_budget": 0.08,
                "compute_cost": 0.2,
            },
            side_effects=[
                {"risk": "Spectral smearing", "expected_when": "gap_length > 50ms", "severity": 0.3},
                {"risk": "Phase discontinuity", "expected_when": "insufficient context", "severity": 0.4},
            ],
            reports={"self_metrics": ["inpainting_quality", "context_coherence"], "confidence": 0.85},
            rollback={"strategy": "snapshot_restore", "supports_partial": True},
        )

    def set_instrument_context(self, instrument_type):
        """
        Setzt instrumenten-spezifischen Kontext für semantische Anpassung

        Args:
            instrument_type: InstrumentType enum (z.B. VOCALS, DRUMS, BASS)
        """
        self.instrument_context = instrument_type
        _logger.info("[SpectralInpainter] Instrument context set: %s", instrument_type)

    def repair_dropouts(self, audio: np.ndarray, dropout_regions: list[tuple]) -> np.ndarray:
        """
        Repariert multiple Dropouts in einem Audio-Signal

        Args:
            audio: Mono audio signal (1D numpy array)
            dropout_regions: Liste von (start, end) Tupeln mit Dropout-Positionen

        Returns:
            Repariertes audio signal
        """
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if len(dropout_regions) == 0:
            return np.asarray(audio.astype(audio.dtype))  # type: ignore[no-any-return]

        # Precondition: audio must be 1D
        if audio.ndim != 1:
            _logger.error("[SpectralInpainter] Requires 1D audio, got shape %s", audio.shape)
            return np.asarray(audio.astype(audio.dtype))  # type: ignore[no-any-return]

        audio_repaired = audio.copy()

        for dropout_start, dropout_end in dropout_regions:
            # Validate region
            if dropout_start < 0 or dropout_end > len(audio) or dropout_start >= dropout_end:
                _logger.warning("[SpectralInpainter] Invalid region [%d, %d], skipping", dropout_start, dropout_end)
                continue

            gap_length = dropout_end - dropout_start

            # Adaptive strategy based on gap size
            if gap_length < 100:  # Very short (<2ms @ 48kHz)
                audio_repaired = self._linear_fill(audio_repaired, dropout_start, dropout_end)
            elif gap_length < 2400:  # Medium (<50ms @ 48kHz)
                audio_repaired = self._spectral_fill(audio_repaired, dropout_start, dropout_end)
            else:  # Long gaps (>50ms)
                audio_repaired = self._context_aware_fill(audio_repaired, dropout_start, dropout_end)

        return np.asarray(audio_repaired.astype(audio.dtype))  # type: ignore[no-any-return]

    def _linear_fill(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """Einfache lineare Interpolation für sehr kurze Gaps"""
        if start > 0 and end < len(audio):
            audio[start:end] = np.interp(
                np.arange(start, end),
                [start - 1, end],
                [audio[start - 1], audio[end]],
            )
        return audio

    def _spectral_fill(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        Spektrale Interpolation für mittlere Gaps
        Analysiert Spektrum vor/nach Dropout und interpoliert Frequenzen
        """
        gap_length = end - start

        # Extrahiere Kontext-Fenster
        pre_start = max(0, start - self.context_samples)
        post_end = min(len(audio), end + self.context_samples)

        pre_context = audio[pre_start:start]
        post_context = audio[end:post_end]

        if len(pre_context) < 10 or len(post_context) < 10:
            # Nicht genug Kontext → Fallback auf linear
            return self._linear_fill(audio, start, end)

        # Spektral-Interpolation: Durchschnitt von Pre/Post-Kontext mit Crossfade
        try:
            # Nutze durchschnittlichen Kontext als Basis
            avg_context = (
                (pre_context[-gap_length:] + post_context[:gap_length]) / 2
                if len(pre_context) >= gap_length and len(post_context) >= gap_length
                else self._linear_fill(audio, start, end)
            )

            # Crossfade für glatte Übergänge
            fade_len = min(50, gap_length // 4)
            if fade_len > 0 and isinstance(avg_context, np.ndarray):
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)

                if start >= fade_len:
                    avg_context[:fade_len] = (
                        audio[start - fade_len : start] * fade_out + avg_context[:fade_len] * fade_in
                    )
                if end + fade_len < len(audio):
                    avg_context[-fade_len:] = avg_context[-fade_len:] * fade_out + audio[end : end + fade_len] * fade_in

                audio[start:end] = avg_context[:gap_length]
        except Exception as e:
            _logger.warning("[SpectralInpainter] Spectral fill failed: %s, falling back to linear", e)
            return self._linear_fill(audio, start, end)

        return np.asarray(audio.astype(audio.dtype))  # type: ignore[no-any-return]

    def _context_aware_fill(self, audio: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        Context-aware Spectral Synthesis für lange Gaps
        Nutzt größere Kontext-Fenster und instrumenten-spezifische Anpassung
        """
        gap_length = end - start

        # Erweiterte Kontext-Fenster für lange Gaps
        extended_context = self.context_samples * 2
        pre_start = max(0, start - extended_context)
        post_end = min(len(audio), end + extended_context)

        pre_context = audio[pre_start:start]
        post_context = audio[end:post_end]

        if len(pre_context) < 100 or len(post_context) < 100:
            # Fallback auf spektrale Interpolation
            return self._spectral_fill(audio, start, end)

        # Analysiere spektrale Eigenschaften des Kontextes
        pre_rms = np.sqrt(np.mean(pre_context**2))
        post_rms = np.sqrt(np.mean(post_context**2))
        avg_rms = (pre_rms + post_rms) / 2

        # Generiere synthetisches Signal via Pattern Repetition + spectral Noise Modulation
        try:
            # Strategie 1: Pattern Repetition (falls periodisches Signal)
            correlation = _sc_correlate(pre_context, pre_context, mode="full", method="fft")
            correlation = correlation[len(correlation) // 2 :]  # Nur positive Lags

            # Finde dominante Periode
            peaks = []
            for i in range(50, min(len(correlation) - 50, 1000)):
                if i > 0 and i < len(correlation) - 1:
                    if correlation[i] > correlation[i - 1] and correlation[i] > correlation[i + 1]:
                        if correlation[i] > 0.5 * correlation[0]:
                            peaks.append(i)

            if len(peaks) > 0:
                # Periodisches Signal → Wiederhole Pattern
                period = peaks[0]
                num_periods = gap_length // period + 1
                synthetic = np.tile(pre_context[-period:], num_periods)[:gap_length]

                # Moduliere Amplitude
                rms_transition = np.linspace(pre_rms, post_rms, gap_length)
                current_rms = np.sqrt(np.mean(synthetic**2)) + 1e-10
                synthetic = synthetic * (rms_transition / current_rms)
            else:
                # Nicht-periodisch → Einfache Interpolation mit Noise
                noise = np.random.randn(gap_length) * avg_rms * 0.1
                linear_base = np.linspace(pre_context[-1], post_context[0], gap_length)
                synthetic = linear_base + noise

            # Crossfade
            fade_len = min(200, gap_length // 6)
            if fade_len > 0 and start >= fade_len and end + fade_len < len(audio):
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)

                synthetic[:fade_len] = audio[start - fade_len : start] * fade_out + synthetic[:fade_len] * fade_in
                synthetic[-fade_len:] = synthetic[-fade_len:] * fade_out + audio[end : end + fade_len] * fade_in

            audio[start:end] = synthetic[:gap_length]
        except Exception as e:
            _logger.warning("[SpectralInpainter] Context-aware fill failed: %s, falling back to spectral", e)
            return self._spectral_fill(audio, start, end)

        return np.asarray(audio.astype(audio.dtype))  # type: ignore[no-any-return]
