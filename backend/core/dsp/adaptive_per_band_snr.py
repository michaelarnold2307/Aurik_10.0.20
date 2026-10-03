import logging

"""
adaptive_per_band_snr.py - SOTA-konformes adaptives Per-Band-SNR für Aurik 6.0
Dieses Modul berechnet adaptiv das SNR pro Frequenzband (klassische DSP, SOTA-Maximum).
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_per_band_snr"
    category: str = "snr"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_per_band_snr_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"eps": 1e-8},
        "safe_ranges": {"eps": {"min": 1e-12, "max": 1e-3}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[
        {
            "risk": "Fehlmessung bei schlechtem Noise-Estimate",
            "expected_when": "noise_spectrogram falsch",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["per_band_snr"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptivePerBandSNR:
    """
    SOTA-konformes adaptives Per-Band-SNR (klassisch):
    - Berechnet das SNR pro Frequenzband aus Signal- und Noise-Spektrogramm
    - Quality-Gates, Auditierbarkeit, Rollback
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(adaptive_per_band_snr_contract))

    def per_band_snr(
        self,
        signal_spectrogram: np.ndarray[Any, Any],
        noise_spectrogram: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """
        Berechnet das SNR pro Frequenzband.
        :param signal_spectrogram: Signal-Spektrogramm (np.ndarray)
        :param noise_spectrogram: Noise-Spektrogramm (np.ndarray)
        :return: SNR pro Band (np.ndarray)
        """
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        signal_power = np.mean(np.abs(signal_spectrogram) ** 2, axis=0)
        noise_power = np.mean(np.abs(noise_spectrogram) ** 2, axis=0)
        snr = 10 * np.log10((signal_power + self.eps) / (noise_power + self.eps))
        return np.asarray(snr)  # type: ignore[no-any-return]

    def auto_optimize(
        self,
        signal_spectrogram: np.ndarray,
        noise_spectrogram: np.ndarray,
    ) -> None:
        """Adaptiert eps anhand des minimalen Rauschpegels im Spektrogramm.

        Sehr leises Rauschen -> kleines eps (präzise Messung).
        Starkes Rauschen -> größeres eps (numerische Stabilität).
        """
        self.log_contract()
        noise_power = np.mean(np.abs(noise_spectrogram) ** 2)
        # eps mindestens 1e-12, maximal 1e-4
        self.eps = float(np.clip(noise_power * 0.01, 1e-12, 1e-4))
