import logging

"""
automatic_denoiser.py - SOTA-konformer automatischer Denoiser für Aurik 6.0
Dieses Modul entfernt Rauschen automatisch aus Audiosignalen (klassische DSP, SOTA-Maximum).
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import istft, stft

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "automatic_denoiser"
    category: str = "noise_removal"
    version: str = "1.0.0"
    io: dict | None = None
    preconditions: list | None = None
    params: dict | None = None
    budgets: dict | None = None
    side_effects: list | None = None
    reports: dict | None = None
    rollback: dict | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
denoiser_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold": 0.5},
        "safe_ranges": {"threshold": {"min": 0.1, "max": 1.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.1,
        "identity_budget": 0.98,
        "spectral_change_budget": 0.2,
        "temporal_change_budget": 0.1,
        "compute_cost": 0.1,
    },
    side_effects=[{"risk": "musical_noise", "expected_when": "threshold > 0.8", "severity": 0.3}],
    reports={"self_metrics": ["noise_reduction_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AutomaticDenoiser:
    """
    Klassischer automatischer Denoiser (SOTA-Maximum):
    - Spektrales Gating/Noise-Reduction ohne ML
    """

    contract: DSPContract = denoiser_contract

    def __init__(self, noise_floor_db: float = -40.0):
        self.noise_floor_db = noise_floor_db

    def log_contract(self) -> None:
        """
        Gibt den DSPContract für Auditierbarkeit aus.
        """
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Entfernt Rauschen aus dem Audiosignal per spektralem Gating (SOTA, keine ML/AI).
        :param audio: Eingabesignal (np.ndarray)
        :param sr: Abtastrate
        :return: Denoised Signal (np.ndarray)
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        _f, _t, Zxx = stft(audio, fs=sr, nperseg=1024)
        mag = np.abs(Zxx)
        noise_thresh = 10 ** (self.noise_floor_db / 20)
        mask = mag > noise_thresh
        Zxx_clean = Zxx * mask
        _, audio_out = istft(Zxx_clean, fs=sr, nperseg=1024)
        audio_out = audio_out[: len(audio)]
        return np.asarray(audio_out.astype(audio.dtype))  # type: ignore[no-any-return]
