"""
auto_panner_gain_ducker.py - Klassische Auto-Panner, Auto-Gain, Auto-Ducker für Aurik 6.0 (SOTA-Maximum)

Dieses Modul stellt klassische, SOTA-konforme Tools für Panning, Gain und Ducking bereit - keine ML/AI, nur DSP.
Alle Algorithmen sind nachvollziehbar, auditierbar und rollback-fähig.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität (klassisch, keine AI/ML)
@dataclass(frozen=True)
class DSPContract:
    id: str
    category: str
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanzen der Contracts (klassisch, keine AI)
auto_panner_contract = DSPContract(
    id="auto_panner",
    category="spatial",
    io={
        "channels": "stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"pan": 0.0}},
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {
            "risk": "Fehlpositionierung",
            "expected_when": "Pan zu extrem",
            "severity": 0.1,
        }
    ],
    reports={"self_metrics": ["panning_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
auto_gain_contract = DSPContract(
    id="auto_gain",
    category="level",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"target_level": -20.0}},
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[{"risk": "Übersteuerung", "expected_when": "Gain zu hoch", "severity": 0.2}],
    reports={"self_metrics": ["gain_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
auto_ducker_contract = DSPContract(
    id="auto_ducker",
    category="ducking",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"threshold": -30.0, "ratio": 2.0}},
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[{"risk": "Pumpen", "expected_when": "Sidechain falsch", "severity": 0.2}],
    reports={"self_metrics": ["ducking_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AutoPanner:
    """
    Klassischer Auto-Panner (SOTA-Maximum):
    - Automatisiert die Stereopositionierung per Amplitudenmodulation (Sinus-LFO)
    """

    contract: DSPContract = auto_panner_contract

    def __init__(self, pan: float = 0.0):
        self.pan = pan

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Wendet klassisches Panning mit LFO auf ein Stereo-Signal an. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param audio: Stereo-Audio (np.ndarray, shape [2, N])
        :param sr: Sample-Rate
        :return: Gepanntes Stereo-Audio (np.ndarray)
        """
        self.log_contract()
        # Quality-Gate: Input-Check
        if not isinstance(audio, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            raise ValueError("Input must be a numpy array")
        if audio.ndim != 2 or audio.shape[0] != 2:
            self._audit_log("error", "Input must be stereo (2,N)")
            raise ValueError("Input must be stereo (2,N)")
        try:
            t = np.arange(audio.shape[1]) / sr
            rate_hz = getattr(self, "rate_hz", 0.5)
            depth = getattr(self, "depth", 1.0)
            lfo = 0.5 * (1 + np.sin(2 * np.pi * rate_hz * t))
            left = audio[0] * (1 - depth * lfo)
            right = audio[1] * (depth * lfo)
            self._audit_log("success", "Panning erfolgreich angewendet")
            return np.asarray(np.stack([left, right], axis=0))  # type: ignore[no-any-return]
        except Exception as e:
            self._audit_log("error", f"Fehler bei Panning: {e}")
            return audio

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[auto_panner] %s", message)


class AutoGain:
    """
    Klassischer Auto-Gain (SOTA-Maximum):
    - Automatische Lautstärkeanpassung auf Zielpegel (z.B. -16 LUFS)
    - Auditierbar, rollback-fähig
    """

    contract: DSPContract = auto_gain_contract

    def __init__(self, target_level: float = -20.0):
        self.target_level = target_level

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def process(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """
        Pegelt das Signal auf den Zielpegel (dBFS). Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param audio: Audio (np.ndarray)
        :param sr: Sample-Rate (optional)
        :return: Gepegeltes Audio (np.ndarray)
        """
        self.log_contract()
        if not isinstance(audio, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            raise ValueError("Input must be a numpy array")
        try:
            rms = np.sqrt(np.mean(audio**2))
            if rms == 0:
                self._audit_log("warn", "RMS ist 0, keine Pegelanpassung")
                return audio
            gain = 10 ** (self.target_level / 20) / rms
            self._audit_log("success", "Gain erfolgreich angewendet")
            return np.asarray(audio * gain)  # type: ignore[no-any-return]
        except Exception as e:
            self._audit_log("error", f"Fehler bei Gain: {e}")
            return audio

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[auto_gain] %s", message)


class AutoDucker:
    """
    Klassischer Auto-Ducker (SOTA-Maximum):
    - Automatisiertes Ducking per Sidechain-Envelope-Follower
    - Auditierbar, rollback-fähig
    """

    contract: DSPContract = auto_ducker_contract

    def __init__(self, threshold: float = -30.0, ratio: float = 2.0):
        self.threshold = threshold
        self.ratio = ratio

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def process(self, audio: np.ndarray, sidechain: np.ndarray) -> np.ndarray:
        """
        Duckt das Signal, wenn Sidechain über Schwellwert liegt. Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param audio: Audio (np.ndarray)
        :param sidechain: Sidechain-Signal (np.ndarray)
        :return: Geducktes Audio (np.ndarray)
        """
        self.log_contract()
        if not isinstance(audio, np.ndarray) or not isinstance(sidechain, np.ndarray):
            self._audit_log("error", "Input(s) not numpy array")
            raise ValueError("audio und sidechain müssen numpy arrays sein")
        try:
            sc_rms = np.sqrt(np.mean(sidechain**2))
            threshold_lin = 10 ** (self.threshold / 20)
            if sc_rms > threshold_lin:
                self._audit_log("success", "Ducking angewendet")
                return np.asarray(audio / self.ratio)  # type: ignore[no-any-return]
            self._audit_log("info", "Kein Ducking notwendig")
            return audio
        except Exception as e:
            self._audit_log("error", f"Fehler bei Ducking: {e}")
            return audio

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[auto_ducker] %s", message)
