import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "broadband_dynamics_stabilizer"
    category: str = "dynamics"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


broadband_dynamics_stabilizer_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"target_lufs": -18.0, "window_ms": 100.0},
        "safe_ranges": {
            "target_lufs": {"min": -30.0, "max": -8.0},
            "window_ms": {"min": 10.0, "max": 500.0},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.02,
    },
    side_effects=[{"risk": "Pumpen", "expected_when": "window_ms < 30.0", "severity": 0.2}],
    reports={"self_metrics": ["dynamics_stability"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class BroadbandDynamicsStabilizer:
    """
    SOTA-konformer Broadband Dynamics Stabilizer:
    - Stabilisiert die Dynamik ohne Pumpen (z.B. RMS-Tracking, sanfte Gain-Riding)
    """

    def __init__(self, target_lufs: float = -18.0, window_ms: float = 100.0):
        self.target_lufs = target_lufs
        self.window_ms = window_ms

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(broadband_dynamics_stabilizer_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """LUFS-based dynamics stabilization with soft pump prevention.

        Replaces RMS with K-weighted LUFS-inspired measurement and uses
        soft gain reduction instead of hard rollback when pumping is detected.

        Args:
            audio: Audio signal (mono)
            sr: Sample rate
        Returns:
            Stabilized audio
        """
        self.log_contract()
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float64))

        window = max(1, int(self.window_ms * sr / 1000.0))
        target_lin = 10.0 ** (self.target_lufs / 20.0)

        # K-weighting approximation (BS.1770 inspired):
        # High-shelf boost at 1681 Hz + high-pass at 38 Hz
        try:
            from scipy.signal import butter, sosfilt

            # High-shelf approximation (2nd order Butterworth high-boost at 1.5 kHz)
            wn_k = np.clip(1500.0 / (sr / 2.0), 0.001, 0.99)
            sos_k = butter(2, wn_k, btype="high", output="sos")
            k_weighted = sosfilt(sos_k, audio) * 1.3 + audio * 0.7  # blend
        except Exception:
            k_weighted = audio

        # RMS envelope on K-weighted signal
        rms_kernel = np.ones(window) / window
        rms = np.sqrt(np.convolve(k_weighted**2, rms_kernel, mode="same") + 1e-12)

        # Compute gain
        gain = np.where(rms > 1e-8, target_lin / rms, 1.0)

        # Limit gain range (max ±12 dB)
        max_gain = 10.0 ** (12.0 / 20.0)
        min_gain = 10.0 ** (-12.0 / 20.0)
        gain = np.clip(gain, min_gain, max_gain)

        # Smooth gain with adaptive smoothing
        smooth_kernel = np.ones(window) / window
        smoothed_gain = np.convolve(gain, smooth_kernel, mode="same")

        # Soft pump prevention: if gain variance > threshold,
        # progressively blend toward unity gain (not hard rollback)
        gain_std = float(np.std(smoothed_gain))
        if gain_std > 0.5:
            # Severe pumping — reduce gain riding intensity proportionally
            blend = np.clip(0.5 / (gain_std + 0.01), 0.1, 0.9)
            smoothed_gain = blend * smoothed_gain + (1.0 - blend) * 1.0
            logger.warning(
                "[BroadbandDynamicsStabilizer] Pump detected (std=%.3f), blending to %.0f%% gain riding",
                gain_std,
                blend * 100,
            )
        elif gain_std > 0.3:
            # Mild pumping — slight blend
            blend = 0.7
            smoothed_gain = blend * smoothed_gain + (1.0 - blend) * 1.0

        out = audio * smoothed_gain
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)
        return np.asarray(out)  # type: ignore[no-any-return]
