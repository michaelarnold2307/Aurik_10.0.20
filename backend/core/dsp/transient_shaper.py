import logging
from dataclasses import asdict, dataclass
from typing import Any


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "transient_shaper"
    category: str = "dynamics"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
transient_shaper_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {
            "band": (100, 8000),
            "attack": 1.5,
            "sustain": 1.0,
            "formant_preserving": False,
        },
        "safe_ranges": {
            "attack": {"min": 0.5, "max": 3.0},
            "sustain": {"min": 0.5, "max": 3.0},
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
    side_effects=[{"risk": "Klicks", "expected_when": "attack > 2.5", "severity": 0.1}],
    reports={"self_metrics": ["transient_enhancement"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, lfilter

logger = logging.getLogger(__name__)


class TransientShaper:
    """
    SOTA-konformer Transient Shaper:
    - Separates Envelope/Sustain-Processing, Bandwahl, ML-ready
    """

    def __init__(
        self,
        band: tuple[float, float] = (100, 8000),
        attack: float = 1.5,
        sustain: float = 1.0,
        formant_preserving: bool = False,
    ) -> None:
        """
        band: Frequenzbereich für Transientenbearbeitung (Hz)
        attack: Verstärkung der Einschwingphase (>1 = mehr Transient)
        sustain: Verstärkung der Sustain-Phase (>1 = mehr Sustain)
        formant_preserving: Wenn True, wird ein Formant-Preserving-Ansatz genutzt (empfohlen für Vocals)
        """
        self.band = band
        self.attack = attack
        self.sustain = sustain
        self.formant_preserving = formant_preserving

    # Audit: Contract-Infos loggen (optional)
    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(transient_shaper_contract))

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet das Eingangssignal mit Transientenbearbeitung.
        audio: 1D numpy-Array (Mono)
        sr: Abtastrate (Hz)
        Rückgabe: geformtes Signal (gleicher Typ wie audio)
        """
        # Audit: Contract-Infos loggen (optional)
        self.log_contract()
        # Band extrahieren
        b, a = butter(4, [self.band[0] / (sr / 2), self.band[1] / (sr / 2)], btype="band", output="ba")
        band_sig = lfilter(b, a, audio)
        # Envelope-Detection (z.B. mit abs + Lowpass)
        env = np.abs(band_sig)
        b_env, a_env = butter(2, 10 / (sr / 2), btype="low", output="ba")
        env_smooth = lfilter(b_env, a_env, env)
        # Transient/Sustain-Trennung
        trans = np.diff(env_smooth, prepend=env_smooth[0])
        trans_gain: np.ndarray = np.clip(1 + self.attack * trans, 0, None)
        sustain_gain: np.ndarray = np.clip(1 + self.sustain * (env_smooth - np.mean(env_smooth)), 0, None)
        # Anwendung
        shaped: np.ndarray = band_sig * trans_gain * sustain_gain
        # Formant-Preserving-Option (vereinfachtes Beispiel: Dry/Wet nur auf Obertöne, nicht auf Grundtonbereich)
        if self.formant_preserving:
            # Obertöne extrahieren (vereinfachtes Beispiel: Hochpass ab 2 kHz)
            bh, ah = butter(2, 2000 / (sr / 2), btype="high", output="ba")
            overtones = lfilter(bh, ah, shaped - band_sig)
            out = audio + overtones
        else:
            # Standard: Mischung mit gesamtem Band
            out = audio + (shaped - band_sig)
        # Pegel normalisieren
        maxval: float = float(np.max(np.abs(out)))
        if maxval > 1.0:
            out = np.clip(out, -1.0, 1.0)
        return np.asarray(out.astype(audio.dtype))
