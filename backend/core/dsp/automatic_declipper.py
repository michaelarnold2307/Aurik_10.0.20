"""
automatic_declipper.py - SOTA-konformer automatischer Declipper für Aurik 6.0

Dieses Modul entfernt Clipping-Artefakte automatisch aus Audiosignalen per klassischer Interpolation (SOTA-Maximum, keine ML/AI, nur DSP).
Alle Algorithmen sind nachvollziehbar, auditierbar und rollback-fähig.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.automatic_declipper")
logger.setLevel(logging.INFO)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "automatic_declipper"
    category: str = "declipper"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


automatic_declipper_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"clip_threshold": 0.98}},
    budgets={"compute_cost": 0.01},
    side_effects=[{"risk": "Artefakte", "expected_when": "hohe Verstärkung", "severity": 0.2}],
    reports={"self_metrics": ["clipping_reduction"]},
    rollback={"strategy": "bypass", "supports_partial": True},
)


class AutomaticDeclipper:
    """
    Klassischer automatischer Declipper (SOTA-Maximum, keine ML/AI):
    - Entfernt Clipping-Artefakte mit klassischer Interpolation (z.B. PCHIP)
    - Keine KI, keine Blackbox, auditierbar und rollback-fähig
    """

    contract: DSPContract = automatic_declipper_contract

    def __init__(self, clip_threshold: float = 0.98):
        self.clip_threshold = clip_threshold

    def log_contract(self) -> None:
        """
        Gibt den DSPContract für Auditierbarkeit aus.
        """
        import logging

        logging.info("[DSPContract] %s", asdict(self.contract))

    def declip(self, audio: np.ndarray, sr: int, use_deep_learning: bool = False, audit_log: bool = True) -> np.ndarray:
        """
        Entfernt Clipping-Artefakte per klassischer Interpolation (SOTA, keine ML/AI).
        Quality Gate, Audit-Logging, optionale DL-Inferenz, robuste Fehlerbehandlung
        :param audio: Eingabesignal (np.ndarray)
        :param sr: Abtastrate
        :param use_deep_learning: Optional Deep-Learning-Inferenz (torch/jit)
        :param audit_log: Audit-Logging aktivieren
        :return: Degeclipptes Signal (np.ndarray)
        """
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0 or sr < 8000:
            logger.error("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
            raise ValueError("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
        if np.isnan(audio).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1.5:
            logger.warning("Audio möglicherweise nicht normiert (max > 1.5)")

        audio_out = None
        fallback_used = False
        try:
            if use_deep_learning and _TORCH_AVAILABLE:
                logger.info("Deep-Learning-Inferenz aktiviert für Declipper.")
                # TorchScript-Modell (Platzhalter)
                # model = torch.jit.load('declipper.pt')
                # audio_out = model(torch.from_numpy(audio).float().unsqueeze(0)).squeeze(0).numpy()
                logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                fallback_used = True
                audio_out = self._declip_classic(audio)
            else:
                audio_out = self._declip_classic(audio)
        except Exception as e:
            logger.error("Fehler bei Declipper: %s", e)
            fallback_used = True
            audio_out = audio.copy()

        if audit_log:
            clipping_reduction = float(np.mean(np.abs(audio - audio_out)))
            logger.info(
                f"AutomaticDeclipper: clipping_reduction={clipping_reduction:.4f}, fallback_used={fallback_used}"
            )
        logger.info("[DSPContract] %s", asdict(self.contract))
        return np.asarray(audio_out.astype(audio.dtype))  # type: ignore[no-any-return]

    def _declip_classic(self, audio: np.ndarray) -> np.ndarray:
        clipped = np.abs(audio) >= self.clip_threshold
        if not np.any(clipped):
            return audio
        from scipy.interpolate import PchipInterpolator

        x = np.arange(len(audio))
        unclipped = ~clipped
        interp = PchipInterpolator(x[unclipped], audio[unclipped], extrapolate=True)
        audio_out = audio.copy()
        audio_out[clipped] = interp(x[clipped])
        maxval: float = float(np.max(np.abs(audio_out)))
        if maxval > 1.0:
            audio_out = np.clip(audio_out, -1.0, 1.0)
        return audio_out
