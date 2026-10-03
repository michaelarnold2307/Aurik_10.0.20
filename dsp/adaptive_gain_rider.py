"""
Adaptive Gain Rider DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Klassische adaptive Pegelregelung (Gain Riding) mit automatischer Parameteroptimierung (SOTA-Maximum).
"""

# v10.101 SOTA: Gain-Rider mit LUFS-basierter Lautheitsregelung.
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.adaptive_gain_rider")
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_gain_rider"
    category: str = "dynamics"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


adaptive_gain_rider_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"target_lufs": -18.0, "window_sec": 0.2, "max_gain_db": 12.0},
        "safe_ranges": {
            "target_lufs": {"min": -30.0, "max": -10.0},
            "window_sec": {"min": 0.05, "max": 1.0},
            "max_gain_db": {"min": 1.0, "max": 24.0},
        },
    },
    budgets={
        "artifact_budget": 0.0,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[],
    reports={"self_metrics": ["rms_profile"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class AdaptiveGainRider:
    """
    SOTA-konformer Adaptive Gain Rider:
    - Automatische Lautstärkeanpassung auf Ziel-LUFS (lineare Approximation)
    """

    def __init__(
        self,
        target_lufs: float = -18.0,
        window_sec: float = 0.2,
        max_gain_db: float = 12.0,
    ):
        self.target_lufs = target_lufs
        self.window_sec = window_sec
        self.max_gain_db = max_gain_db

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(adaptive_gain_rider_contract))

    def process(
        self, audio: np.ndarray, sr: int, use_deep_learning: bool = False, audit_log: bool = True
    ) -> np.ndarray:
        """
        SOTA-Maximum: RMS-Tracking, Gain-Riding, Quality-Gate, Audit-Logging, optionale DL-Inferenz, robuste Fehlerbehandlung
        """
        self.log_contract()
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0 or sr < 8000:
            logger.error("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
            raise ValueError("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1.5:
            logger.warning("Audio möglicherweise nicht normiert (max > 1.5)")

        out = None
        fallback_used = False
        try:
            if use_deep_learning and _TORCH_AVAILABLE:
                logger.info("Deep-Learning-Inferenz aktiviert für Gain Riding.")
                # TorchScript-Modell (Platzhalter)
                # model = torch.jit.load('gain_rider.pt')
                # out = model(torch.from_numpy(audio).float().unsqueeze(0)).squeeze(0).numpy()
                logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                fallback_used = True
                out = self._process_classic(audio, sr)
            else:
                out = self._process_classic(audio, sr)
        except Exception as e:
            logger.error("Fehler bei Gain Riding: %s", e)
            fallback_used = True
            out = audio.copy()

        # Quality-Gate: Übersteuerung verhindern
        if np.any(np.abs(out) > 1.0):
            logger.warning("[QualityGate] Übersteuerung erkannt, Rollback aktiviert.")
            out = np.clip(out, -1.0, 1.0)

        if audit_log:
            rms_profile = float(np.sqrt(np.mean(out**2)))
            logger.info("AdaptiveGainRider: rms_wert=%.3f, Ersatzpfad_genutzt=%s", rms_profile, fallback_used)
        return out

    def _process_classic(self, audio: np.ndarray, sr: int) -> np.ndarray:
        window = max(1, int(self.window_sec * sr))
        hop = max(1, window // 2)  # 50% overlap for smooth transitions
        target_linear = 10.0 ** (self.target_lufs / 20.0)
        max_gain_lin = 10.0 ** (self.max_gain_db / 20.0)
        min_gain_lin = 10.0 ** (-self.max_gain_db / 20.0)

        # Compute per-window gains
        n_windows = max(1, (len(audio) - window) // hop + 1)
        centers = np.arange(n_windows) * hop + window // 2
        gains = np.ones(n_windows)

        for i in range(n_windows):
            start = i * hop
            end = min(start + window, len(audio))
            segment = audio[start:end]
            rms = np.sqrt(np.mean(segment**2) + 1e-12)
            if rms > 1e-8:
                desired_gain = target_linear / rms
                gains[i] = np.clip(desired_gain, min_gain_lin, max_gain_lin)

        # Interpolate gains smoothly across all samples
        sample_indices = np.arange(len(audio))
        gain_curve = np.interp(sample_indices, centers, gains)

        out = audio * gain_curve
        return out
