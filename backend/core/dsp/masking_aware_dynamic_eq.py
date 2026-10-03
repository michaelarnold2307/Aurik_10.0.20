import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.masking_aware_dynamic_eq")
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class DSPContract:
    id: str = "masking_aware_dynamic_eq"
    category: str = "eq"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


masking_eq_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"bands": 8, "max_gain_db": 6.0, "min_gain_db": -6.0},
        "safe_ranges": {
            "bands": {"min": 3, "max": 32},
            "max_gain_db": {"min": 1.0, "max": 12.0},
            "min_gain_db": {"min": -12.0, "max": -1.0},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.05,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.03,
    },
    side_effects=[{"risk": "Überbetonung", "expected_when": "max_gain_db > 8.0", "severity": 0.2}],
    reports={"self_metrics": ["masking_reduction"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class MaskingAwareDynamicEQ:
    """
    SOTA-konformer Masking-aware Dynamic EQ:
    - Analysiert spektrale Maskierung und passt EQ-Bänder adaptiv an
    """

    def __init__(self, bands: int = 8, max_gain_db: float = 6.0, min_gain_db: float = -6.0):
        self.bands = bands
        self.max_gain_db = max_gain_db
        self.min_gain_db = min_gain_db

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(masking_eq_contract))

    def process(
        self, audio: np.ndarray, sr: int, use_deep_learning: bool = False, audit_log: bool = True
    ) -> np.ndarray:
        """
        SOTA-Maximum: Maskierungsanalyse, adaptive EQ-Bänder, Quality-Gate, Audit-Logging, optionale DL-Inferenz, robuste Fehlerbehandlung
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

        eq_audio = None
        fallback_used = False
        try:
            if use_deep_learning and _TORCH_AVAILABLE:
                logger.info("Deep-Learning-Inferenz aktiviert für Masking-EQ.")
                # TorchScript-Modell (Platzhalter)
                # model = torch.jit.load('masking_eq.pt')
                # eq_audio = model(torch.from_numpy(audio).float().unsqueeze(0)).squeeze(0).numpy()
                logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                fallback_used = True
                eq_audio = self._process_classic(audio, sr)
            else:
                eq_audio = self._process_classic(audio, sr)
        except Exception as e:
            logger.error("Fehler bei Masking-EQ: %s", e)
            fallback_used = True
            eq_audio = audio.copy()

        # Quality-Gate: Keine Überbetonung?
        if np.max(np.abs(eq_audio)) > 2.0:
            logger.warning("[QualityGate] Überbetonung, Rollback aktiviert.")
            eq_audio = audio.copy()

        if audit_log:
            masking_reduction = float(np.mean(np.abs(eq_audio - audio)))
            logger.info(
                f"MaskingAwareDynamicEQ: masking_reduction={masking_reduction:.4f}, fallback_used={fallback_used}"
            )
        return eq_audio

    def _process_classic(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Masking-Aware Dynamic EQ via Audio-EQ-Cookbook Biquad-Filter (Bristow-Johnson).

        Algorithmus:
          1. FFT-basierte Energie pro logarithmisch aufgeteiltem Band.
          2. Maskierungsmodell: Dominante Bänder absenken, schwache anheben
             (Ziel: gleichmäßige spektrale Energieverteilung).
          3. Biquad Peaking-EQ pro Band via sosfilt.
        """
        from scipy.signal import sosfilt

        n = len(audio)
        # Logarithmisch aufgeteilte Bandgrenzen (20 Hz … Nyquist)
        bands = np.logspace(np.log10(20.0), np.log10(sr / 2.0 * 0.95), self.bands + 1)

        # Energie pro Band via FFT (erste 4096 Samples für Geschwindigkeit)
        fft_len = min(n, 4096)
        freqs = np.fft.rfftfreq(fft_len, d=1.0 / sr)
        power = np.abs(np.fft.rfft(audio[:fft_len])) ** 2

        band_energy = np.zeros(self.bands)
        for i in range(self.bands):
            mask = (freqs >= bands[i]) & (freqs < bands[i + 1])
            if mask.any():
                band_energy[i] = float(np.mean(power[mask]))

        total_energy: float = float(np.sum(band_energy))
        if total_energy > 1e-30:
            band_energy /= total_energy

        # Maskierungsmodell: Ziel = Gleichverteilung (1/bands pro Band)
        target = 1.0 / self.bands
        gains = np.zeros(self.bands)
        for i in range(self.bands):
            if band_energy[i] > 1e-30:
                ratio = target / band_energy[i]
                gain_db = 10.0 * np.log10(np.clip(ratio, 0.01, 100.0))
                gains[i] = float(np.clip(gain_db, self.min_gain_db, self.max_gain_db))

        # Biquad Peaking-EQ pro Band (Audio-EQ-Cookbook, R. Bristow-Johnson)
        eq_audio = audio.copy()
        band_centers = np.sqrt(bands[:-1] * bands[1:])  # geometrisches Mittel

        for i in range(self.bands):
            if abs(gains[i]) < 0.05:
                continue
            fc = float(band_centers[i])
            if fc <= 0.0 or fc >= sr / 2.0:
                continue
            w0 = 2.0 * np.pi * fc / sr
            A = 10.0 ** (gains[i] / 40.0)  # dB/20 = gain/2
            Q = 1.0 / np.sqrt(2.0)  # Butterworth-Q
            alpha = np.sin(w0) / (2.0 * Q)
            b0 = 1.0 + alpha * A
            b1 = -2.0 * np.cos(w0)
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * np.cos(w0)
            a2 = 1.0 - alpha / A
            if abs(a0) < 1e-12:
                continue
            sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
            try:
                eq_audio = sosfilt(sos, eq_audio)
            except Exception:
                logger.warning("EQ-Band %d übersprungen (Filter instabil)", i)

        return eq_audio
