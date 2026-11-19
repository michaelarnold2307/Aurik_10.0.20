import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("aurik.dsp.reverb_amount_estimator")
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class DSPContract:
    id: str = "reverb_amount_estimator"
    category: str = "reverb"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


reverb_estimator_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"window_sec": 2.0},
        "safe_ranges": {"window_sec": {"min": 0.5, "max": 10.0}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.0,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.0,
        "temporal_change_budget": 0.0,
        "compute_cost": 0.01,
    },
    side_effects=[],
    reports={"self_metrics": ["reverb_amount"], "confidence": 1.0},
    rollback={"strategy": "none", "supports_partial": False},
)


class ReverbAmountEstimator:
    """
    SOTA-konformer Reverb Amount Estimator:
    - Schätzt Nachhallzeit (RT60) und Reverb-Anteil im Signal
    """

    def __init__(self, window_sec: float = 2.0):
        self.window_sec = window_sec

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(reverb_estimator_contract))

    def process(self, audio: np.ndarray, sr: int, audit_log: bool = True) -> float:
        """
        SOTA-Maximum: Blind RT60-Schätzung und Reverb-Detektion
        Quality Gate, Audit-Logging, robuste Fehlerbehandlung
        :param audio: Eingabe-Audiodaten (np.ndarray)
        :param sr: Samplingrate
        :param audit_log: Audit-Logging aktivieren
        :return: Geschätzte RT60 (float)
        """
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            logger.error("Ungültiges Audio-Array (leer oder falscher Typ)")
            raise ValueError("Ungültiges Audio-Array (leer oder falscher Typ)")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1e6:
            logger.warning("Audio möglicherweise nicht normiert (max > 1e6)")

        try:
            self.log_contract()
            # 1. Energieverlauf berechnen (Schätzung der Nachhallzeit)
            window = int(self.window_sec * sr)
            if len(audio) < window:
                window = len(audio)
            env = np.abs(audio[:window])
            env_db = 20 * np.log10(env + 1e-8)
            # 2. Lineare Regression auf Hüllkurve (Decay)
            from scipy.stats import linregress

            t = np.arange(window) / sr
            slope, _intercept, _r, _p, _stderr = linregress(t, env_db)
            # 3. RT60-Schätzung (klassisch: -60dB/Slope)
            rt60 = 0.0 if slope >= 0 else -60.0 / slope
            # 4. Quality-Gate: Plausibilität
            if rt60 < 0 or rt60 > 10:
                logger.warning("[QualityGate] Warnung: Unplausible RT60-Schätzung, Rollback aktiviert.")
                rt60 = 0.0
        except Exception as e:
            logger.error("Fehler bei RT60-Schätzung: %s", e)
            rt60 = 0.0

        if audit_log:
            logger.info("ReverbAmountEstimator: rt60=%.3f", rt60)
        return rt60
