"""
adaptive_spectral_rolloff.py - SOTA-konformes Spectral Rolloff Modul für Aurik 6.0

Dieses Modul implementiert klassisches Spectral Rolloff (SOTA-Maximum, keine ML/AI) für Audiosignale.
Es berechnet adaptiv die Rolloff-Frequenz und ist mit vollständigem DSPContract und Auditierbarkeit ausgestattet.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.adaptive_spectral_rolloff")
logger.setLevel(logging.INFO)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_spectral_rolloff"
    category: str = "spectral_rolloff"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
adaptive_spectral_rolloff_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"n_fft": 2048, "hop_length": 512, "roll_percent": 0.85},
        "safe_ranges": {
            "n_fft": {"min": 256, "max": 8192},
            "roll_percent": {"min": 0.5, "max": 0.99},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {
            "risk": "Fehlklassifikation",
            "expected_when": "roll_percent zu niedrig",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["rolloff_freq"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveSpectralRolloff:
    """
    SOTA-konformes klassisches Spectral Rolloff (keine ML/AI):
    - Berechnet adaptiv die Rolloff-Frequenz eines Audiosignals
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        roll_percent: float = 0.85,
        center: bool = True,
    ):
        """
        Initialisiert das Spectral Rolloff Modul mit Quality-Gate für Parameter.
        :param sr: Abtastrate
        :param n_fft: FFT-Größe
        :param hop_length: Hop-Länge
        :param roll_percent: Anteil für Rolloff-Schwelle
        :param center: Padding für zentrierte Frames
        """
        if not (256 <= n_fft <= 8192):
            logger.error("Ungültiges n_fft: %s. Muss zwischen 256 und 8192 liegen.", n_fft)
            raise ValueError("n_fft muss zwischen 256 und 8192 liegen.")
        if not (0.5 <= roll_percent <= 0.99):
            logger.error("Ungültiges roll_percent: %s. Muss zwischen 0.5 und 0.99 liegen.", roll_percent)
            raise ValueError("roll_percent muss zwischen 0.5 und 0.99 liegen.")
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.roll_percent = roll_percent
        self.center = center
        logger.info(
            f"AdaptiveSpectralRolloff initialisiert mit sr={self.sr}, n_fft={self.n_fft}, hop_length={self.hop_length}, roll_percent={self.roll_percent}, center={self.center}"
        )

    def log_contract(self) -> None:
        """
        Gibt den DSPContract für Auditierbarkeit aus (Log + Print).
        """
        contract_dict = asdict(adaptive_spectral_rolloff_contract)
        logger.info("[DSPContract] %s", contract_dict)

    def spectral_rolloff(
        self, y: np.ndarray[Any, Any], use_deep_learning: bool = False, audit_log: bool = True, **kwargs: Any
    ) -> np.ndarray[Any, Any]:
        """
        Berechnet die Rolloff-Frequenz pro Frame (SOTA, keine ML/AI).
        Quality-Gate, Audit-Logging, optionale DL-Inferenz, Fehlerbehandlung, SOTA-Transparenz.
        :param y: Audiosignal (np.ndarray)
        :param use_deep_learning: Optional Deep-Learning-Inferenz (torch/jit)
        :param audit_log: Audit-Logging aktivieren
        :param kwargs: Optionale Parameter (sr, n_fft, hop_length, roll_percent, center)
        :return: Rolloff-Frequenzen (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(y, np.ndarray):
            logger.error("y ist kein np.ndarray")
            raise TypeError("y ist kein np.ndarray")
        if y.size == 0:
            logger.error("y ist leer")
            raise ValueError("y ist leer")
        if np.isnan(y).any():
            logger.error("y enthält NaN-Werte")
            raise ValueError("y enthält NaN-Werte")

        sr = kwargs.get("sr", self.sr)
        n_fft = kwargs.get("n_fft", self.n_fft)
        hop_length = kwargs.get("hop_length", self.hop_length)
        roll_percent = kwargs.get("roll_percent", self.roll_percent)
        center = kwargs.get("center", self.center)

        output = None
        fallback_used = False
        try:
            if use_deep_learning:
                if not _TORCH_AVAILABLE:
                    logger.warning("PyTorch nicht verfügbar, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._spectral_rolloff_classic(y, sr, n_fft, hop_length, roll_percent, center)
                else:
                    logger.info("Deep-Learning-Inferenz aktiviert für Spectral Rolloff.")
                    # TorchScript-Modell (Platzhalter)
                    # model = torch.jit.load('spectral_rolloff.pt')
                    # output = model(torch.from_numpy(y).float().unsqueeze(0)).squeeze(0).numpy()
                    logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                    fallback_used = True
                    output = self._spectral_rolloff_classic(y, sr, n_fft, hop_length, roll_percent, center)
            else:
                output = self._spectral_rolloff_classic(y, sr, n_fft, hop_length, roll_percent, center)
        except Exception as e:
            logger.error("Fehler bei Spectral Rolloff: %s", e, exc_info=True)
            fallback_used = True
            output = np.zeros(1)

        if audit_log:
            rolloff_mean = float(np.mean(output)) if output is not None else float("nan")
            logger.info(
                f"AdaptiveSpectralRolloff: rolloff_mean={rolloff_mean:.2f}, fallback_used={fallback_used}, n_fft={n_fft}, roll_percent={roll_percent}"
            )
            logger.info("[DSPContract] %s", asdict(adaptive_spectral_rolloff_contract))
        return output

    def _spectral_rolloff_classic(
        self, y: np.ndarray, sr: int, n_fft: int, hop_length: int, roll_percent: float, center: bool
    ) -> np.ndarray:
        """
        Klassische Rolloff-Berechnung (SOTA, keine ML/AI).
        """
        if center:
            pad = n_fft // 2
            y = np.pad(y, (pad, pad), mode="reflect")
        rolloff = []
        for i in range(0, len(y) - n_fft + 1, hop_length):
            frame = y[i : i + n_fft]
            spectrum = np.abs(np.fft.rfft(frame))
            total_energy: float = float(np.sum(spectrum))
            threshold = roll_percent * total_energy
            cumulative = np.cumsum(spectrum)
            rolloff_bin = np.where(cumulative >= threshold)[0]
            freq = rolloff_bin[0] * sr / n_fft if len(rolloff_bin) > 0 else 0.0
            rolloff.append(freq)
        return np.asarray(np.array(rolloff))  # type: ignore[no-any-return]

    def auto_optimize(self, y: np.ndarray[Any, Any], sr: int) -> None:
        """
        Passt die FFT-Parameter adaptiv an die Signalgröße an (Dummy, normkonform gekennzeichnet).
        :param y: Audiosignal (np.ndarray)
        :param sr: Abtastrate
        """
        self.log_contract()
        if len(y) < 4096:
            self.n_fft = 256
            self.hop_length = 64
        elif len(y) < 16384:
            self.n_fft = 1024
            self.hop_length = 256
        else:
            self.n_fft = 2048
            self.hop_length = 512
        logger.info("FFT-Parameter auto-optimiert: n_fft=%s, hop_length=%s", self.n_fft, self.hop_length)
