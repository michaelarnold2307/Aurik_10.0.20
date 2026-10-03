"""
ultra_low_latency.py — Ultra-Low-Latency-DSPs für Aurik.

Latenzoptimierte Varianten der Kern-DSPs für Echtzeit- und Monitoring-
Pfade, mit DSPContract für Auditierbarkeit und SOTA-Konformität.

Wohlklang-Prinzipien (§III (copilot-instructions.md)):
  - Limiter: Soft-Knee-Gain statt tanh-Waveshaping (kein harmonischer
    Klirrfaktor in leisen/mittleren Passagen), harte Ceiling nur als
    Gain-Guard, Wellenform bleibt unverzerrt.
  - Denoiser: WOLA mit Fensterleistungs-Normierung (keine Pegelwelligkeit),
    globale Rauschboden-Schätzung statt Pro-Frame-Sprung (kein Pumpen).
  - Gate: weiche Gain-Kennlinie (Attack/Release) statt hartem Muten
    (keine Schalt-Klicks an den Flanken).
  - Identität unterhalb der Wirkschwelle (Contract identity_budget).
Stereo-Layout-Invariante (AGENTS.md): Verarbeitung pro Kanal-View,
nie mean(axis=0) ohne Layout-Check.
Determinismus (§G5 (copilot-instructions.md)): keine Zufallszahlen,
keine Zeitstempel in Entscheidungen.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str
    category: str
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanzen der Contracts
ull_limiter_contract = DSPContract(
    id="ultra_low_latency_limiter",
    category="limiter",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"ceiling": 0.9}, "safe_ranges": {"ceiling": {"min": 0.5, "max": 1.0}}, "trial_profile": {}},
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Limiter-Pumpen",
            "expected_when": "Release zu langsam für Peaks",
            "severity": 0.1,
        }
    ],
    reports={"self_metrics": ["limiting_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
ull_denoiser_contract = DSPContract(
    id="ultra_low_latency_denoiser",
    category="denoiser",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"floor_gain": 0.4},
        "safe_ranges": {"floor_gain": {"min": 0.2, "max": 1.0}},
        "trial_profile": {},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[{"risk": "Artefakte", "expected_when": "Rauschboden überschätzt", "severity": 0.2}],  # type: ignore[list-item]
    reports={"self_metrics": ["denoising_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
ull_gate_contract = DSPContract(
    id="ultra_low_latency_gate",
    category="gate",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold_dbfs": -40.0},
        "safe_ranges": {"threshold_dbfs": {"min": -80.0, "max": -10.0}},
        "trial_profile": {},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Falschabschaltung",
            "expected_when": "Threshold zu hoch",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["gating_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


def _channel_views(audio: np.ndarray) -> tuple[list[np.ndarray], bool]:
    """Kanal-Views je nach Layout (Stereo-Layout-Invariante, AGENTS.md).

    Rückgabe: (Views als 1-D-Slices, transposed_flag). Mono → eine View.
    """
    arr = np.asarray(audio)
    if arr.ndim == 1:
        return [arr], False
    from backend.core.audio_layout import is_channels_first

    if is_channels_first(arr):
        return [np.ascontiguousarray(arr[c]) for c in range(arr.shape[0])], False
    return [np.ascontiguousarray(arr[:, c]) for c in range(arr.shape[1])], True


def _rebuild(views: list[np.ndarray], transposed: bool, template: np.ndarray) -> np.ndarray:
    stacked = np.stack(views, axis=1 if transposed else 0)
    return stacked.reshape(template.shape).astype(template.dtype)  # type: ignore[no-any-return]


def _smooth_gain(target: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Randkorrekturierte Hann-Glättung einer Gain-Kurve (kein Pegelsprung an den Rändern)."""
    if target.size < 2:
        return target.copy()
    if kernel.size >= target.size:
        # Fenster nie länger als das Signal (np.convolve 'same' liefert sonst
        # max(M, N) Samples) — ungerade Kürzung auf höchstens target.size.
        width = target.size if target.size % 2 == 1 else target.size - 1
        if width < 3:
            return target.copy()
        kernel = np.hanning(width)
    num = np.convolve(target, kernel, mode="same")
    den = np.convolve(np.ones_like(target), kernel, mode="same")
    return np.asarray(num / np.maximum(den, 1e-12))  # type: ignore[no-any-return]


class UltraLowLatencyLimiter:
    """Ultra-Low-Latency-Limiter: Soft-Knee-Gain-Limiter ohne Look-ahead.

    Unterhalb der Soft-Knee (−6 dB re Ceiling) bleibt das Signal bit-identisch
    (keine tanh-Verzerrung); darüber läuft die Gain-Reduktion weich in die
    Ceiling. Die Ceiling wird zusätzlich als per-sample Gain-Guard erzwungen
    (nie harte Wellenform-Klemmung).
    """

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(ull_limiter_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """ULL-Limiter: Soft-Knee-Limiter (Ceiling = 0.9, Latenz 0)."""
        self.log_contract()
        ceiling = 0.9
        knee = 0.5 * ceiling  # 6 dB Soft-Knee
        views, transposed = _channel_views(audio)
        out_views: list[np.ndarray] = []
        smooth_kernel = np.hanning(max(3, int(sr * 0.010) | 1))
        for x_any in views:
            x = x_any.astype(np.float64)
            peak = np.abs(x)
            g_target = np.ones_like(x)
            over_knee = peak > knee
            if np.any(over_knee):
                u = np.clip((peak[over_knee] - knee) / (ceiling - knee), 0.0, 1.0)
                g_target[over_knee] = np.minimum(1.0, (ceiling / peak[over_knee]) ** (u * u))
            g = _smooth_gain(g_target, smooth_kernel)
            # Instantanter Ceiling-Guard: Gain darf nie über die Ceiling treiben.
            guard = np.ones_like(g)
            loud = peak > ceiling
            if np.any(loud):
                guard[loud] = np.minimum(1.0, ceiling / peak[loud])
            g = np.minimum(g, guard)
            out_views.append((x * g).astype(x_any.dtype))
        result = _rebuild(out_views, transposed, np.asarray(audio))
        self._audit(result)
        result_arr: np.ndarray = result
        return result_arr

    def _audit(self, result: np.ndarray) -> None:
        logger.debug("[Audit][UltraLowLatencyLimiter] peak=%.5f", float(np.max(np.abs(result))) if result.size else 0.0)


class UltraLowLatencyDenoiser:
    """Ultra-Low-Latency-Denoiser: WOLA-Spektralsubtraktion (128er-FFT).

    50 %-OLA mit Fensterleistungs-Normierung (keine Pegelwelligkeit),
    globaler Rauschboden pro Bin (20. Perzentil über alle Frames) und
    Gain-Floor −8 dB (sehr sanfte Subtraktion, Artefakt-Budget 0.01).
    """

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(ull_denoiser_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """ULL-Denoiser: Spektrale Unterdrückung mit 128-Punkt-FFT-Frame."""
        self.log_contract()
        views, transposed = _channel_views(audio)
        out_views: list[np.ndarray] = []
        for x_any in views:
            x = x_any.astype(np.float64)
            out_views.append(self._denoise_channel(x).astype(x_any.dtype))
        result = _rebuild(out_views, transposed, np.asarray(audio))
        result = np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        self._audit(result)
        result_arr: np.ndarray = result.astype(np.asarray(audio).dtype)
        return result_arr

    def _denoise_channel(self, x: np.ndarray) -> np.ndarray:
        n_fft = 128  # Minimale FFT-Größe für ULL
        hop = n_fft // 2
        n = x.size
        if n < n_fft:
            # Zu kurzes Fenster: unangetastet statt stiller Null-Strecke.
            logger.debug("ULL-Denoiser: Signal kürzer als n_fft (%d < %d) — Passthrough", n, n_fft)
            return x.copy()
        window = np.hanning(n_fft)
        n_frames = (n - n_fft) // hop + 1
        spec = np.empty((n_frames, n_fft // 2 + 1), dtype=np.complex128)
        for i in range(n_frames):
            seg = x[i * hop : i * hop + n_fft] * window
            spec[i] = np.fft.rfft(seg)
        mag = np.abs(spec)
        phase = np.angle(spec)
        # Globaler Rauschboden pro Bin (stationär, kein Pro-Frame-Pumpen)
        floor = np.percentile(mag, 20, axis=0)
        gain = np.maximum(1.0 - floor[None, :] / (mag + 1e-12), 0.4)
        result = np.zeros(n, dtype=np.float64)
        norm = np.zeros(n, dtype=np.float64)
        w_pow = window * window
        for i in range(n_frames):
            frame = np.fft.irfft(mag[i] * gain[i] * np.exp(1j * phase[i]))
            s = i * hop
            result[s : s + n_fft] += frame * window
            norm[s : s + n_fft] += w_pow
        return np.asarray(result / np.maximum(norm, 1e-12))  # type: ignore[no-any-return]

    def _audit(self, result: np.ndarray) -> None:
        logger.debug(
            "[Audit][UltraLowLatencyDenoiser] rms=%.6f", float(np.sqrt(np.mean(result**2))) if result.size else 0.0
        )


class UltraLowLatencyGate:
    """Ultra-Low-Latency-Gate: weiche Gain-Kennlinie statt hartem Muten.

    Envelope-Follower mit 4 ms Attack / 20 ms Release; die Gate-Gain-Kurve
    folgt mit denselben Zeitkonstanten und multipliziert das Signal — die
    Flanken erzeugen keine Schalt-Klicks (Artifacts < Budget 0.01).
    """

    def log_contract(self) -> None:
        logger.debug("[DSPContract] %s", asdict(ull_gate_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """ULL-Gate: Sample-genauer Envelope-Follower + weiches Schwellen-Gate."""
        self.log_contract()
        threshold = 10 ** (-40 / 20)  # -40 dBFS
        attack = max(1, int(sr * 0.004))  # 4 ms
        release = max(1, int(sr * 0.020))  # 20 ms
        alpha_att = 1 - np.exp(-1.0 / attack)
        alpha_rel = 1 - np.exp(-1.0 / release)
        views, transposed = _channel_views(audio)
        out_views: list[np.ndarray] = []
        for x_any in views:
            x = x_any.astype(np.float64)
            env = 0.0
            gate = 0.0
            out = np.empty_like(x)
            for i in range(x.size):
                peak = abs(x[i])
                if peak > env:
                    env += alpha_att * (peak - env)
                else:
                    env += alpha_rel * (peak - env)
                target = 1.0 if env >= threshold else 0.0
                alpha = alpha_att if target > gate else alpha_rel
                gate += alpha * (target - gate)
                out[i] = x[i] * gate
            out_views.append(out.astype(x_any.dtype))
        result = _rebuild(out_views, transposed, np.asarray(audio))
        self._audit(result)
        result_arr: np.ndarray = result
        return result_arr

    def _audit(self, result: np.ndarray) -> None:
        logger.debug("[Audit][UltraLowLatencyGate] peak=%.6f", float(np.max(np.abs(result))) if result.size else 0.0)
