import logging

"""
ai_chain_recommendation.py - Adaptive Chain Recommendation (AI) für Aurik 6.0

Dieses Modul schlägt automatisch die optimale DSP-Kette für ein Audiosignal vor (Stub).
"""

"""
chain_recommendation.py - Adaptive Chain Recommendation (AI) für Aurik 6.0

SOTA-konforme DSP-Kettenempfehlung mit DSPContract und Auditierbarkeit.
"""
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContractChainRecommendation:
    id: str = "chain_recommendation"
    category: str = "chain_recommendation"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


chain_recommendation_contract = DSPContractChainRecommendation(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"model_path": None}},
    budgets={"compute_cost": 0.01},
    side_effects=[
        {
            "risk": "Fehlempfehlung",
            "expected_when": "Modell nicht geladen",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["chain_quality"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AiChainRecommendation:
    """
    SOTA-konforme Adaptive Chain Recommendation (AI):
    - Analysiert das Audiosignal und empfiehlt eine DSP-Kette (AI-gestützt)
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractChainRecommendation = chain_recommendation_contract

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def recommend(self, audio: np.ndarray, sr: int) -> list[str]:
        """Heuristische DSP-Kettenempfehlung basierend auf Signalanlyse.

        Algorithmus (rein DSP, kein Modell erforderlich):
            1. Mono-Konvertierung und NaN-Schutz
            2. SNR-Schätzung via Spektral-Floor (20. Perzentil der Frame-RMS)
            3. Click-Detektion: Spitzenwert > 6σ des Mittels
            4. Brumm-Detektion: Energie bei 50/100/150 Hz relativ zu Gesamt
            5. Clipping-Detektion: Anteil der Samples nahe 0 dBFS
            6. Sibilanten: relative Energie 6–16 kHz
            7. Kette aufsteigend nach Schwere zusammenstellen

        Wenn ein ONNX-Modell geladen ist, wird dessen Ausgabe als priorisierte
        Label-Liste interpretiert und zur heuristischen Kette hinzugefügt.

        Returns:
            List[str]: Geordnete DSP-Kette, z. B. ['ClickRemover', 'Denoiser'].
        """
        self.log_contract()

        # ── Mono + NaN-Schutz ──────────────────────────────────────────
        mono = audio.mean(axis=0).astype(np.float64) if audio.ndim == 2 else audio.astype(np.float64)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(mono)
        if n == 0:
            return ["Denoiser"]

        chain: list[str] = []

        # ── 1. Click-Detektion ─────────────────────────────────────────
        abs_mono = np.abs(mono)
        click_thr = np.mean(abs_mono) + 6.0 * np.std(abs_mono)
        click_ratio = float(np.mean(abs_mono > click_thr))
        if click_ratio > 0.001:  # > 0,1 % Samples sind Clicks
            chain.append("ClickRemover")

        # ── 2. SNR-Schätzung (Rauschboden via Frame-RMS-Perzentil) ─────
        frame = max(int(0.03 * sr), 128)
        n_fr = n // frame
        if n_fr > 0:
            rms_frames = np.sqrt(np.mean(mono[: n_fr * frame].reshape(n_fr, frame) ** 2, axis=1) + 1e-12)
            noise_floor = np.percentile(rms_frames, 20)
            signal_rms = np.sqrt(np.mean(mono**2) + 1e-12)
            snr_db = 20.0 * np.log10((signal_rms + 1e-12) / (noise_floor + 1e-12))
        else:
            snr_db = 60.0

        if snr_db < 30.0:
            chain.append("Denoiser")

        # ── 3. Brumm-Detektion (50/100/150 Hz) ────────────────────────
        ps = np.abs(np.fft.rfft(mono, n=min(n, 65536))) ** 2
        freqs = np.fft.rfftfreq(min(n, 65536), d=1.0 / sr)
        total_power = np.sum(ps) + 1e-12
        hum_freqs = [50.0, 60.0, 100.0, 120.0, 150.0, 180.0]
        hum_power = sum(float(np.sum(ps[(freqs >= f - 3) & (freqs <= f + 3)])) for f in hum_freqs)
        if hum_power / total_power > 0.05:  # > 5 % Energie im Brumm
            chain.append("HumRemover")

        # ── 4. Clipping-Detektion ──────────────────────────────────────
        clip_ratio = float(np.mean(abs_mono >= 0.98))
        if clip_ratio > 0.001:
            chain.append("Declip")

        # ── 5. Sibilanten / De-Esser (6–16 kHz relativ zu 1–6 kHz) ───
        hi_mask = (freqs >= 6000) & (freqs <= 16000)
        mid_mask = (freqs >= 1000) & (freqs < 6000)
        hi_e = float(np.sum(ps[hi_mask])) + 1e-12
        mid_e = float(np.sum(ps[mid_mask])) + 1e-12
        if hi_e / mid_e > 0.6:  # HF deutlich erhöht
            chain.append("DeEsser")

        # ── 6. Dynamik / Limiter immer am Ende ────────────────────────
        peak = float(np.max(abs_mono))
        if peak > 0.90:
            chain.append("Limiter")

        if not chain:
            chain = ["Denoiser"]

        # ── 7. Optional: ONNX-Modell priorisiert einfügen ─────────────
        if self.model is not None:
            try:
                _in_name = self.model.get_inputs()[0].name
                _inp = mono[np.newaxis, :].astype(np.float32)
                _out = self.model.run(None, {_in_name: _inp})
                # Interpretation: Ausgabe ist Liste von Label-Indices
                if isinstance(_out, list) and len(_out) > 0:
                    _labels = list(_out[0].flatten().astype(int))
                    _label_map = {
                        0: "Denoiser",
                        1: "ClickRemover",
                        2: "HumRemover",
                        3: "Declip",
                        4: "DeEsser",
                        5: "Limiter",
                    }
                    ml_chain = [_label_map[i] for i in _labels if i in _label_map]
                    # Führe ML-Empfehlung an, behalte DSP-Kette als Fallback
                    chain = ml_chain + [c for c in chain if c not in ml_chain]
            except Exception as _onnx_err:
                logger.warning(
                    "AiChainRecommendation: ONNX fehlgeschlagen (%s) — heuristische Kette aktiv.",
                    _onnx_err,
                )

        logger.debug("ChainRecommendation: empfohlene Kette=%s", chain)
        return chain
