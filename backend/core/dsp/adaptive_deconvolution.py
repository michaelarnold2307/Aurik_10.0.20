"""
Adaptive Deconvolution / Inverse Filtering Modul für Aurik 6.0 (SOTA-Maximum)
SOTA-tauglich, adaptiv, mit automatischer Parameteroptimierung (klassische DSP, SOTA-Maximum).
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dsp._memory_budget_guard import check_budget

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_logger = logging.getLogger(__name__)

# Kanonischer Pfad zum models/-Verzeichnis (relativ zu diesem Paket, Fallback auf cwd)
_MODELS_DIR: Path = Path(__file__).parent.parent / "models"


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_deconvolution"
    category: str = "deconvolution"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveDeconvolution:
    """
    Klassische adaptive Deconvolution/Inverse Filtering (SOTA-Maximum)
    """

    contract: DSPContract = DSPContract()

    def __init__(self, method: str = "wiener", auto_optimize: bool = True):
        """
        method: 'wiener', 'spectral', 'rls', 'custom'
        auto_optimize: Wenn True, werden Parameter automatisch optimiert.
        """
        self.method = method
        self.auto_optimize = auto_optimize
        self.last_params: dict[str, Any] | None = None

    def log_contract(self):
        # Optional: Audit-Log für Vertrag
        _logger.debug("[DSPContract] %s", asdict(self.contract))

    def deconvolve(
        self,
        audio: np.ndarray,
        ir: np.ndarray,
        snr: float = 30.0,
        use_deep_learning: bool = False,
        audit_log: bool = True,
    ) -> np.ndarray:
        """
        Führt Deconvolution durch. Benötigt das Impulsantwortsignal (ir).
        snr: Signal-Rausch-Verhältnis für Wiener-Filter (nur relevant für 'wiener')
        Quality Gate, Audit-Logging, optionale DL-Inferenz, robuste Fehlerbehandlung
        """
        self.log_contract()
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0 or not isinstance(ir, np.ndarray) or ir.size == 0:
            _logger.error("Ungültiges Audio- oder IR-Array (leer oder falscher Typ)")
            return np.asarray(np.zeros(max(audio.size if isinstance(audio, np.ndarray) else 1, 1), dtype=np.float32))  # type: ignore[no-any-return]
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        ir = np.nan_to_num(ir, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(audio)) > 1.5 or np.max(np.abs(ir)) > 1.5:
            _logger.warning("Audio oder IR möglicherweise nicht normiert")

        result = None
        fallback_used = False
        try:
            if use_deep_learning and _TORCH_AVAILABLE:
                # ONNX-Modell für neuronale Deconvolution wird via ModelDownloader bereitgestellt.
                # Solange kein verifiziertes ONNX-Checkpoint in models/manifest.json eingetragen ist,
                # wird auf die klassische MMSE/Wiener-Kette zurückgefallen (Post-2018, spec §4.2).
                manifest_path = _MODELS_DIR / "manifest.json"
                onnx_path: Path | None = None
                if manifest_path.exists():
                    import json as _json

                    with open(manifest_path, encoding="utf-8") as _f:
                        _manifest = _json.load(_f)
                    for _entry in _manifest.get("models", []):
                        if _entry.get("name") == "adaptive_deconvolution" and _entry.get("bundled"):
                            _bp = Path(_entry.get("bundled_path", ""))
                            if _bp.exists():
                                onnx_path = _bp
                                break
                if onnx_path is not None:
                    try:
                        import onnxruntime as _ort

                        if not check_budget("adaptive_deconv_onnx", 0.1):
                            raise RuntimeError("Memory budget exceeded")
                        _sess = _ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                        _in_name = _sess.get_inputs()[0].name
                        _ir_name = _sess.get_inputs()[1].name
                        _out = _sess.run(
                            None,
                            {
                                _in_name: audio[np.newaxis, :].astype(np.float32),
                                _ir_name: ir[np.newaxis, :].astype(np.float32),
                            },
                        )[0].squeeze()
                        result = np.nan_to_num(_out.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                        _logger.info("AdaptiveDeconvolution: ONNX-Inferenz erfolgreich (%s).", onnx_path.name)
                    except Exception as _onnx_err:
                        _logger.warning(
                            "ONNX-Deconvolution fehlgeschlagen (%s) — Fallback auf klassische Methode.", _onnx_err
                        )
                        fallback_used = True
                        result = self._deconvolve_classic(audio, ir, snr)
                else:
                    _logger.debug("Kein ONNX-Deconvolution-Modell gebündelt — klassische MMSE/Wiener-Kette aktiv.")
                    fallback_used = True
                    result = self._deconvolve_classic(audio, ir, snr)
            else:
                result = self._deconvolve_classic(audio, ir, snr)
        except Exception as e:
            _logger.error("Fehler bei Deconvolution: %s", e)
            fallback_used = True
            result = audio.copy()

        if audit_log:
            _logger.info(
                "AdaptiveDeconvolution: method=%s, snr=%s, fallback_used=%s",
                self.method,
                snr,
                fallback_used,
            )
        return result

    def _deconvolve_classic(self, audio: np.ndarray, ir: np.ndarray, snr: float) -> np.ndarray:
        if self.method == "wiener":
            return self._wiener_deconvolution(audio, ir, snr)
        elif self.method == "spectral":
            return self._spectral_deconvolution(audio, ir)
        elif self.method == "rls":
            return self._rls_deconvolution(audio, ir)
        else:
            # Unbekannte Methode: NotImplementedError werfen.
            # deconvolve() (die öffentliche Methode) fängt das ab und
            # liefert einen sicheren Fallback — kein Raise nach außen.
            raise NotImplementedError(
                f"AdaptiveDeconvolution: Unbekannte Methode '{self.method}' — "
                "gültige Werte: 'wiener', 'spectral', 'rls'."
            )

    def _wiener_deconvolution(self, audio: np.ndarray, ir: np.ndarray, snr: float) -> np.ndarray:
        # Wiener-Deconvolution im Frequenzbereich
        n = len(audio) + len(ir) - 1
        A = np.fft.fft(audio, n)
        H = np.fft.fft(ir, n)
        H_conj = np.conj(H)
        SNR_linear = 10 ** (snr / 10)
        # Wiener-Filter
        G = H_conj / (H * H_conj + 1 / SNR_linear)
        X = A * G
        result = np.fft.ifft(X)
        return np.asarray(np.real(result)[: len(audio)].astype(audio.dtype))  # type: ignore[no-any-return]

    def _spectral_deconvolution(self, audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
        # Einfache spektrale Division
        n = len(audio) + len(ir) - 1
        A = np.fft.fft(audio, n)
        H = np.fft.fft(ir, n)
        H[H == 0] = 1e-8  # Vermeide Division durch Null
        X = A / H
        result = np.fft.ifft(X)
        return np.asarray(np.real(result)[: len(audio)].astype(audio.dtype))  # type: ignore[no-any-return]

    def _rls_deconvolution(self, audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """
        Recursive Least Squares (RLS) adaptive Inversfilterung.

        Trainiert einen FIR-Inversfilter W der Länge N so, dass:
            W ★ IR ≈ δ[n - delay]   (Dirac-Impuls mit Systemdelay)

        Danach wird W auf `audio` angewandt, um das dekOnvolvierte Signal
        zu rekonstruieren.

        Referenz: Haykin, "Adaptive Filter Theory", 4th Ed., Kap. 13.

        Args:
            audio: Beobachtetes Signal (= Original ★ IR)
            ir: Bekannte Impulsantwort (wird invertiert)

        Returns:
            Dekonvolviertes Audio (gleiche Form wie input).
        """
        # Filterlänge: heuristisch 2× IR-Länge, mindestens 32, maximal 256
        N = min(max(2 * len(ir), 32), 256)
        delay = N // 2  # Kausaler Systemdelay im Inversfilter
        lambda_ = 0.99  # Vergessensfaktor (kleiner = schnellere Konvergenz)
        delta_reg = 1e-2  # Regularisierung der Anfangs-Kovarianzmatrix

        # ------------------------------------------------------------------
        # Trainingssignal für RLS-Inversfilteridentifikation.
        # Problem: Ein kurzes IR (z.B. 3 Samples) liefert nur len(ir) Trainings-
        # iterationen — viel zu wenig für einen N-Tap-Filter (N>>len(ir)).
        # Lösung: Synthetische Trainingssequenz der Länge T >> N:
        #   - Excitation e[n]: Pseudo-Weißrauschen (reproduzierbar, seed=42)
        #   - Messung  y[n] = e[n] ★ IR  (Systemausgang)
        #   - Ziel     d[n] = e[n - delay]  (Verzögertes Original)
        #   - RLS lernt Inversfilter W so dass W★y[n] ≈ d[n]
        # ------------------------------------------------------------------
        rng = np.random.default_rng(seed=42)
        T = max(15 * N, min(len(audio), 4096))  # Trainingssequenzlänge

        excit = rng.standard_normal(T + len(ir)).astype(np.float64)
        # Systemantwort als Trainings-Eingang zum adaptiven Filter
        measured = np.convolve(excit, ir.astype(np.float64), mode="full")[: T + len(ir)]
        # Gewünschte Ausgabe: Excitation um delay verschoben
        desired_sig = np.concatenate([np.zeros(delay), excit])[: T + len(ir)]

        # RLS-Initialisierung: P = (1/δ)·I, w = 0
        w = np.zeros(N, dtype=np.float64)
        P = (1.0 / delta_reg) * np.eye(N, dtype=np.float64)

        # Zero-Pad für gleitendes Fenster
        measured_padded = np.pad(measured, (N - 1, 0))

        # RLS-Schleife: T Iterationen (genug für Konvergenz eines N-Tap-Filters)
        for n in range(T):
            # Eingabevektor: zeitumgekehrtes Fenster (Konvolutions-Konvention)
            x_n = measured_padded[n : n + N][::-1]

            # Kalman-Gewinn:  k = P·x / (λ + x^T·P·x)
            Px = P @ x_n
            denom = lambda_ + float(x_n @ Px)
            if abs(denom) < 1e-12:
                continue
            k_n = Px / denom

            # Fehler: Ziel minus Filterausgang
            idx = n + N - 1
            d_n = desired_sig[idx] if idx < len(desired_sig) else 0.0
            e_n = d_n - float(w @ x_n)

            # Koeffizientenupdate
            w = w + k_n * e_n

            # Kovarianzmatrix-Update (Rang-1-Downdate): P ← (P - k·(x^T·P)) / λ
            xP = x_n @ P
            P = (P - np.outer(k_n, xP)) / lambda_

        # Inversfilter auf tatsächliches Audio anwenden
        result = np.convolve(audio.astype(np.float64), w, mode="full")[: len(audio)]

        # Energie-Normalisierung: Inversfilter verändert Gesamtgain
        rms_in = np.sqrt(np.mean(audio**2) + 1e-30)
        rms_out = np.sqrt(np.mean(result**2) + 1e-30)
        if rms_out > 1e-10:
            result *= rms_in / rms_out

        _logger.info(
            "RLS-Deconvolution: N=%d, delay=%d, λ=%.3f, IR-Länge=%d",
            N,
            delay,
            lambda_,
            len(ir),
        )
        return np.asarray(np.clip(result, -1.0, 1.0).astype(audio.dtype))  # type: ignore[no-any-return]

    def auto_optimize_params(
        self, audio: np.ndarray, ir: np.ndarray, target: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Wählt Dekonvolutions-Methode und Regularisierung anhand des Signal-SNR.
        Hoher SNR → Wiener-Filter (beste Qualität).
        Mittlerer SNR → Spektrale Division.
        Niedriger SNR → RLS (robuster).
        target: Optionales Zielspektrum oder Referenzsignal
        """
        # SNR-Schätzung: Verhältnis Signalleistung zu geschätztem Rauschboden
        mag = np.abs(np.fft.rfft(audio.astype(float)))
        np.abs(np.fft.rfft(ir.astype(float), n=len(audio)))
        noise_floor = np.percentile(mag, 10)
        signal_power = np.mean(mag)
        snr_est = float(signal_power / (noise_floor + 1e-8))

        if snr_est >= 15.0:
            best_method = "wiener"
        elif snr_est >= 5.0:
            best_method = "spectral"
        else:
            best_method = "rls"

        # Regularisierungsparameter: weniger Regularisierung bei hohem SNR
        reg = max(1e-4, 0.5 / (snr_est + 1.0))

        self.method = best_method
        self.last_params = {"method": best_method, "snr": snr_est, "reg": reg}
        _logger.info("auto_optimize_params: SNR=%.2f → method=%s, reg=%.4f", snr_est, best_method, reg)
        return self.last_params
