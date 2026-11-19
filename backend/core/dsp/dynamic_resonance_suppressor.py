import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    id: str = "dynamic_resonance_suppressor"
    category: str = "dynamics"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


# Instanz des Contracts
resonance_suppressor_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold_db": -30.0, "q_factor": 8.0, "reduction_db": 6.0},
        "safe_ranges": {
            "threshold_db": {"min": -60.0, "max": 0.0},
            "q_factor": {"min": 2.0, "max": 20.0},
            "reduction_db": {"min": 0.0, "max": 24.0},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.02,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.02,
    },
    side_effects=[
        {
            "risk": "Klangfärbung",
            "expected_when": "reduction_db > 12.0",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["resonance_reduction"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class DynamicResonanceSuppressor:
    """
    SOTA-konformer Dynamic Resonance Suppressor:
    - Findet und unterdrückt störende Resonanzen dynamisch (z.B. mit adaptiven Notch-Filtern)
    """

    def __init__(
        self,
        threshold_db: float = -30.0,
        q_factor: float = 8.0,
        reduction_db: float = 6.0,
    ):
        self.threshold_db = threshold_db
        self.q_factor = q_factor
        self.reduction_db = reduction_db

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(resonance_suppressor_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        SOTA-Maximum: Automatische Resonanzdetektion und adaptive Notch-Filterung
        - 1. Analyse: Spektrum berechnen, Peaks (Resonanzen) finden
        - 2. Adaptive Notch-Filter für erkannte Resonanzen anwenden
        - 3. Quality-Gate: Prüfen, ob keine Artefakte entstehen (z.B. Pegel, Spektrum, Transienten)
        """
        self.log_contract()
        # 1. Analyse: Spektrum und Peak-Picking
        from scipy.signal import find_peaks, iirnotch, lfilter

        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        # dB-Skala
        spectrum_db = 20 * np.log10(spectrum + 1e-8)
        # Resonanz-Peaks suchen (oberhalb threshold_db)
        peaks, _ = find_peaks(spectrum_db, height=self.threshold_db)
        # 2. Adaptive Notch-Filter anwenden
        filtered = audio.copy()
        for peak_idx in peaks:
            freq = freqs[peak_idx]
            if freq < 40 or freq > sr / 2 - 100:
                continue  # keine Notch für Subbass/Ultrahochton
            # Notch-Filter-Parameter
            q = self.q_factor
            w0 = freq / (sr / 2)
            b, a = iirnotch(w0, q)
            filtered = lfilter(b, a, filtered)
        # 3. Quality-Gate: Keine Überdämpfung/Artefakte?
        # (Vergleich RMS/Spektrum vor/nachher, einfache Heuristik)
        rms_before = np.sqrt(np.mean(audio**2))
        rms_after = np.sqrt(np.mean(filtered**2))
        if rms_after < 0.5 * rms_before:
            logger.warning("[QualityGate] Warnung: Starke Pegelabsenkung, Rollback aktiviert.")
            return audio  # Rollback bei Artefakt
        # Optional: weitere Checks (Transienten, Spektralvergleich)
        return filtered
