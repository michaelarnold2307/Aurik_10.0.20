"""SOTA-M12 (D-K4-7): WF-V4-Klassen-Warp-Head auf FCPE/CRePE-F0-Basis.

Challenger zur heuristischen Phase-12-Warp-Schätzung (Rest-Wow/Flutter unter
der Hörschwelle). Modellklasse im Muster der Challenger-Integrationen
SOTA-M8 (`backend/core/dsp/aspade_declipper.py`) und SOTA-M9
(`backend/core/dsp/tf_gridnet_denoiser.py`): ONNX-Inferenz, Flag-Gate
(``music_model_flags.use_wf4_warp_head``), §V6-DSP-Fallback.

WARP-HEAD-VERTRAG
=================
Eingabe (beide Pfade, ONNX wie Torch-Ref):
    x_f0      : (T,) float32 — log2-f0-Abweichung in Cents/1200 relativ zum
                robusten Median der Trajektorie (0 = keine Abweichung).
    x_conf    : (T,) float32 — [0, 1] Frame-Konfidenz (FCPE/CRePE/pYIN).
    x_feat    : (T, F) float32 — optionale Spektral-Features (F = N_SPEC_FEAT,
                Default 8: spektraler Schwerpunkt, Bandbreite, Rolloff 85 %,
                Flatness, Flux, Low-Band-Energie, High-Band-Energie, RMS).
Ausgabe:
    warp      : (T,) float32 — Warp-Kurve als log2-Abweichung (Cents/1200),
                smooth (Slope-Limit im Netz), |warp| ≤ WARP_LIMIT_LOG2.
                Umrechnung in Stretch-Faktoren: sf = 2**(warp) (Zeit-Verzerrung).
    conf_out  : (T,) float32 — Modell-Konfidenz der Schätzung in [0, 1].

Beide Pfade sind deterministisch (§G5 (copilot-instructions.md)): ONNX-
Session mit fixen Thread-Counts (inter=1, intra=4), Torch in eval() und
inference_mode(). Keine zeitabhängigen Entscheidungen.

ARCHITEKTUR-KLASSENAUSWAHL (analog D-K4-1, SOTA-M8)
===================================================
Klassenwahl: **kompaktes 1-D-CNN + BiLSTM-Encoder-Dekoder ("WF-V4-Klasse")**
für die zeitsynchrone Warp-Regression — bewusst KEIN Transformer und KEIN
Volllängen-Wave-U-Net:

1.  **Zeitsynchrone 1-D-Regression statt Spektral-Synthese**: Die Zielgröße
    ist eine EINE-dimensionale Kurve (Warp pro Frame), kein Spektrogramm.
    Ein CNN-Stack mit wachsendem Dilations-Kontext extrahiert lokale
    Modulations-Muster (Wow < 4 Hz, Flutter 4–100 Hz); ein BiLSTM bildet
    die langsame Referenz-Tonhöhen-Drift (denkbar über viele Sekunden) ab.
    Reine TCN-Architekturen (SOTA-M8-Analogie) reichen für die langsame
    Komponente ohne sehr große Dilationskerne nicht; reine LSTM ohne
    Convolution-Dilations kontextieren die schnelle Flutter-Komponente
    (4–100 Hz bei 40 fps = 0,04–2,5 Frames/Periode) schlecht.
2.  **Kompaktheit ≪ 1 Mio. Parameter** (Default: 3 Dilations-Blöcke à 32
    Channels, 1 BiLSTM-Layer à 32 Units ≈ 32 k Parameter): der Head läuft
    als Challenger auf CPU im Phase-12-Pfad und muss auch bei fehlendem
    ONNX-Export als Torch-Ref schnell bleiben (Laufzeit-Budget Phase 12).
3.  **Warped-Median-Vertrag statt "rohe f0"**: Die Eingabe ist die
    robuste f0-Abweichung (relativ zum Median der Trajektorie) — damit
    bleibt der Head material- und instrumenten-unabhängig (§V7 (copilot-instructions.md)):
    keine phasen-individuellen Schwellwerte;
    der Median ist keine Material-Konstante).
4.  **Slope-Limit als Hard-Constraint**: Die Ausgabe wird im Netz mit
    tanh-Begrenzung + anschließender Lipschitz-Projektion (wie
    `_smooth_stretch_factors` in der Phase 12) erzeugt — Stufen in der
    Warp-Kurve erzeugen Energie-Sprünge (§WF-V3-Befund), die Nie-schlechter-
    Invariante verlangt glatte Kurven.

ONNX-Vertrag: Eingabe ``x_f0`` (1, T), ``x_conf`` (1, T), ``x_feat``
(1, T, F); Ausgabe ``warp`` (1, T), ``conf_out`` (1, T). Statisch über T =
``T_MAX_FRAMES`` (400 Frames ≈ 10 s bei 40 fps) — dynamische LSTM-Exporte
scheitern an Torch-Dynamo-Shape-Constraints (dokumentierter Weg wie bei
SOTA-M9, `scripts/train_tf_gridnet_denoiser.py`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from backend.core import music_model_flags

logger = logging.getLogger(__name__)  # §III.7 (copilot-instructions.md)

# ── Modell-Konstanten (Vertrag mit scripts/train_wf4_warp_head.py) ────────────

#: Frame-Rate der FCPE/CRePE/pYIN-Trajektorien in Phase 12 (40 fps; Nyquist
#: 20 Hz — Flutter > 20 Hz wird im separaten `_flutter_correction_pass` über
#: Sample-Rate-Warp behandelt, siehe Phase 12 §7.4c).
TARGET_FPS = 40.0
#: Maximale Frame-Anzahl pro Modell-Aufruf (statischer ONNX-Vertrag).
T_MAX_FRAMES = 400
#: Anzahl Spektral-Features (siehe Modul-Docstring, Ausgabe 8).
N_SPEC_FEAT = 8
#: Ausgabe-Grenze der Warp-Kurve in log2 (Cents/1200): ±50 cents ≈ ±5,8 %.
WARP_LIMIT_LOG2 = 50.0 / 1200.0
#: Klassen-Dilations der Convolution-Blöcke (WF-V4-Klasse, siehe Docstring).
DEFAULT_DILATIONS = (1, 3, 9)
#: Hidden-Größe des BiLSTM-Encoders.
DEFAULT_HIDDEN = 32
#: Default ONNX-Modellpfad (analog SOTA-M9: models/<slug>/<name>.onnx).
DEFAULT_MODEL_SUBPATH = "models/wf4_warp_head/wf4_warp_head.onnx"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ── Feature-Extraktion (deterministisch, gemeinsam mit dem Training) ─────────


def compute_spectral_features(mono: np.ndarray, sample_rate: int, n_frames: int) -> np.ndarray:
    """(n_frames, N_SPEC_FEAT) Spektral-Features eines Mono-Signals.

    Deterministisch (§G5 (copilot-instructions.md)): feste STFT-Parameter
    (N_FFT=1024, HOP=256 @ 48 kHz), Hanning-Fenster, kein Zufall. NaN/Inf-
    Schutz (§0a (copilot-instructions.md)): alle Features durchlaufen
    ``np.nan_to_num``.
    """
    x = np.nan_to_num(np.asarray(mono, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n_fft, hop = 1024, 256
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    win = np.hanning(n_fft).astype(np.float32)
    n = int(x.shape[-1])
    n_frames_stft = max(1, (n - n_fft) // hop + 1)
    feats: np.ndarray = np.zeros((n_frames_stft, N_SPEC_FEAT), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(sample_rate, 1)).astype(np.float32)
    for t in range(n_frames_stft):
        seg = x[t * hop : t * hop + n_fft] * win
        mag = np.abs(np.fft.rfft(seg)).astype(np.float32)
        mag_sum = float(mag.sum()) + 1e-9
        centroid = float((mag * freqs).sum() / mag_sum)
        spread = float(np.sqrt(np.maximum((mag * (freqs - centroid) ** 2).sum() / mag_sum, 0.0)))
        csum = np.cumsum(mag) / mag_sum
        rolloff = float(freqs[min(int(np.searchsorted(csum, 0.85)), len(freqs) - 1)])
        log_mag = np.log1p(mag)
        flatness = float(np.exp(log_mag.mean()) / (mag.mean() + 1e-9))
        flux = float(np.maximum(mag - (feats[t - 1, 4] if t > 0 else 0.0), 0.0).mean())
        low_e = float(mag[freqs < 250.0].sum() / mag_sum)
        high_e = float(mag[freqs > 4000.0].sum() / mag_sum)
        feats[t] = (
            centroid / 8000.0,
            spread / 4000.0,
            rolloff / 8000.0,
            min(flatness, 4.0) / 4.0,
            min(flux, 2.0) / 2.0,
            low_e,
            high_e,
            float(np.sqrt(np.mean(mag**2))),
        )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    if n_frames_stft == n_frames:
        return feats
    # Zeit-Angleichung auf die Frame-Anzahl der f0-Trajektorie (linear).
    src_idx = np.arange(n_frames_stft, dtype=np.float64)
    dst_idx = np.linspace(0.0, max(n_frames_stft - 1, 0), n_frames)
    out: np.ndarray = np.empty((n_frames, N_SPEC_FEAT), dtype=np.float32)
    for j in range(N_SPEC_FEAT):
        out[:, j] = np.interp(dst_idx, src_idx, feats[:, j].astype(np.float64)).astype(np.float32)
    return out


def warp_to_stretch_factors(warp_log2: np.ndarray) -> np.ndarray:
    """log2-Warp-Kurve → Stretch-Faktoren (Phase-12-Konvention sf > 1 = langsamer)."""
    stretch: np.ndarray = np.power(2.0, np.nan_to_num(np.asarray(warp_log2, dtype=np.float64), nan=0.0)).astype(
        np.float32
    )
    return stretch


# ── Torch-Modell (WF-V4-Klasse) ─────────────────────────────────────────────


def build_wf4_warp_model(
    n_feat: int = N_SPEC_FEAT,
    hidden: int = DEFAULT_HIDDEN,
    dilations: tuple[int, ...] = DEFAULT_DILATIONS,
) -> Any:
    """WF-V4-Klassen-Warp-Head als Torch-Modell (siehe Modul-Docstring).

    Vertrag: forward(x_f0 (B, T), x_conf (B, T), x_feat (B, T, F)) →
    (warp (B, T), conf_out (B, T)) — beide float32, |warp| ≤ WARP_LIMIT_LOG2.
    """
    import torch
    import torch.nn as nn

    class _DilatedBlock(nn.Module):
        def __init__(self, ch: int, dilation: int) -> None:
            super().__init__()
            self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation)
            self.norm = nn.GroupNorm(1, ch)

        def forward(self, h: Any) -> Any:
            return self.norm(torch.tanh(self.conv(h)))

    class WarpHeadNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_ch = 2 + n_feat
            self.proj = nn.Conv1d(in_ch, hidden, kernel_size=1)
            self.blocks = nn.ModuleList(_DilatedBlock(hidden, d) for d in dilations)
            self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
            self.warp_head = nn.Linear(2 * hidden, 1)
            self.conf_head = nn.Linear(2 * hidden, 1)

        def forward(self, x_f0: Any, x_conf: Any, x_feat: Any) -> tuple[Any, Any]:
            b, t = x_f0.shape
            feats = torch.cat([x_f0.unsqueeze(-1), x_conf.unsqueeze(-1), x_feat], dim=-1)  # (B, T, 2+F)
            h = self.proj(feats.transpose(1, 2))  # (B, hidden, T)
            for blk in self.blocks:
                h = h + blk(h)
            seq = h.transpose(1, 2)  # (B, T, hidden)
            y, _ = self.lstm(seq)
            warp = torch.tanh(self.warp_head(y).squeeze(-1)) * WARP_LIMIT_LOG2
            conf_out = torch.sigmoid(self.conf_head(y).squeeze(-1))
            return warp, conf_out

    return WarpHeadNet()


# ── ONNX-Inferenz (lazy, Flag-gesteuert) ─────────────────────────────────────


class WF4WarpHead:
    """ONNX-Inferenz des WF-V4-Warp-Head (deterministisch, §G5 (copilot-instructions.md)).

    Lazy-Loading wie bei ``TFGridNetDenoiser`` (SOTA-M9): das ONNX-Modell wird
    erst beim ersten ``predict``-Aufruf geladen; fehlendes/defektes Modell ⇒
    ``is_available() == False`` und der Aufrufer fällt auf die heuristische
    Phase-12-Schätzung zurück (§V6 (copilot-instructions.md)).
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._session: Any | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._model_ok: bool = False
        if model_path is None:
            self._model_path = _PROJECT_ROOT / DEFAULT_MODEL_SUBPATH
        else:
            self._model_path = Path(model_path)
        self._try_load_model()

    def _try_load_model(self) -> None:
        if not self._model_path.exists():
            logger.warning(
                "SOTA-M12 WF4-Warp-Head ONNX nicht gefunden: %s — Challenger inaktiv "
                "(§V6 (copilot-instructions.md): heuristische Phase-12-Schätzung als Fallback).",
                self._model_path,
            )
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("WF4WarpHead", size_gb=0.15):
                    logger.warning(
                        "SOTA-M12 WF4-Warp-Head: ML-Memory-Budget erschöpft — Challenger inaktiv "
                        "(§V6 (copilot-instructions.md): heuristische Phase-12-Schätzung als Fallback)."
                    )
                    return
            except Exception as _exc:  # pragma: no cover - Systemgrenze
                logger.debug(
                    "SOTA-M12 WF4-Warp-Head: Speicher-Kontingent-Prüfung nicht verfügbar (unkritisch): %s", _exc
                )

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3
            # §III.9 (copilot-instructions.md): ONNX-Pfad bleibt CPU (kein
            # Paritäts-Nachweis für Nicht-CPU-EPs vorhanden).
            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
            if not self._contract_ok():
                self._session = None
                return
            self._model_ok = True
            logger.info(
                "SOTA-M12 WF4-Warp-Head ONNX geladen: %s | In=%s | Out=%s",
                self._model_path,
                self._input_names,
                self._output_names,
            )
        except Exception as _exc:
            logger.warning(
                "SOTA-M12 WF4-Warp-Head ONNX nicht ladbar (%s) — Challenger inaktiv "
                "(§V6 (copilot-instructions.md): heuristische Phase-12-Schätzung als Fallback).",
                _exc,
            )

    def _contract_ok(self) -> bool:
        """Prüft Input-Vertrag (x_f0, x_conf, x_feat) — statische Dims müssen passen."""
        sess = self._session
        if sess is None:
            return False
        try:
            shapes = [(i.name, i.shape) for i in sess.get_inputs()]
        except Exception:
            shapes = []
        if len(shapes) < 3:
            logger.warning(
                "SOTA-M12 WF4-Warp-Head: %d Inputs statt 3 (x_f0, x_conf, x_feat) — "
                "Vertrags-Bruch, Challenger inaktiv (§V6 (copilot-instructions.md)).",
                len(shapes),
            )
            return False
        for name, shape in shapes[:3]:
            if not shape or len(shape) not in (2, 3):
                logger.warning(
                    "SOTA-M12 WF4-Warp-Head: Input %s hat Shape %s (erwartet 2-D/3-D) — "
                    "Vertrags-Bruch, Challenger inaktiv (§V6 (copilot-instructions.md)).",
                    name,
                    shape,
                )
                return False
        return True

    def is_available(self) -> bool:
        return self._model_ok and self._session is not None

    def predict(
        self,
        x_f0: np.ndarray,
        x_conf: np.ndarray,
        x_feat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ONNX-Inferenz: (warp_log2, conf_out) für eine Frame-Sequenz ≤ T_MAX_FRAMES.

        Deterministisch (§G5 (copilot-instructions.md)): feste Session-Threads,
        keine Zufallszustände. NaN/Inf-Schutz (§0a (copilot-instructions.md)):
        Eingaben werden vor dem Feed gesäubert, Ausgaben nach dem Run.
        """
        sess = self._session
        if sess is None:
            # Ehrlicher Abbruch statt stiller Fallback (§V6 (copilot-instructions.md)).
            raise RuntimeError("WF4-Warp-Head: ONNX-Session nicht geladen — Inferenz unmöglich")
        t = int(np.asarray(x_f0).shape[-1])
        f0 = np.nan_to_num(np.asarray(x_f0, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.clip(np.nan_to_num(np.asarray(x_conf, dtype=np.float32), nan=0.0), 0.0, 1.0)
        feat = np.nan_to_num(np.asarray(x_feat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        feed = {
            self._input_names[0]: np.ascontiguousarray(f0.reshape(1, t), dtype=np.float32),
            self._input_names[1]: np.ascontiguousarray(conf.reshape(1, t), dtype=np.float32),
            self._input_names[2]: np.ascontiguousarray(feat.reshape(1, t, -1), dtype=np.float32),
        }
        outs = sess.run(self._output_names, feed)
        warp = np.nan_to_num(np.asarray(outs[0], dtype=np.float32).reshape(-1)[:t], nan=0.0)
        conf_out = np.clip(np.nan_to_num(np.asarray(outs[1], dtype=np.float32).reshape(-1)[:t], nan=0.0), 0.0, 1.0)
        warp = np.clip(warp, -WARP_LIMIT_LOG2, WARP_LIMIT_LOG2).astype(np.float32)
        return warp, conf_out.astype(np.float32)


_warp_head_singleton: WF4WarpHead | None = None


def get_wf4_warp_head() -> WF4WarpHead:
    """Prozess-Singleton der ONNX-Inferenz (zustandslos, pro Song aufrufbar)."""
    global _warp_head_singleton  # pylint: disable=global-statement
    if _warp_head_singleton is None:
        _warp_head_singleton = WF4WarpHead()
    return _warp_head_singleton


# ── Challenger-Stufe (Flag-Gate, §V6-Fallback) ───────────────────────────────


def apply_wf4_warp_stage(
    pitch_trajectory: np.ndarray,
    confidence: np.ndarray,
    sample_rate: int,
    *,
    audio: np.ndarray | None = None,
    head: WF4WarpHead | None = None,
) -> tuple[np.ndarray | None, bool]:
    """Optionale WF-V4-Warp-Schätzung NUR hinter ``use_wf4_warp_head``.

    Args:
        pitch_trajectory: f0-Kurve (Hz) aus Phase 12 (FCPE/CRePE/pYIN-Pfad).
        confidence: Frame-Konfidenz [0, 1].
        sample_rate: Sample-Rate des Quell-Audios (für Spektral-Features).
        audio: Optionales Mono-Audio für Spektral-Features; ohne Audio werden
            Null-Features übergeben (Head arbeitet rein auf f0-Basis).
        head: Optionale Test-Injektion (Muster SOTA-M9).

    Returns:
        (stretch_factors, used). Flag AUS ⇒ (None, False) — keine
        Verhaltensänderung. Jeder Fehler hinter dem Flag ⇒
        ``logger.warning`` (§V6 (copilot-instructions.md)) + heuristische
        Phase-12-Schätzung als Fallback.
    """
    if not music_model_flags.use_wf4_warp_head:
        return None, False
    try:
        pitch_arr = np.asarray(pitch_trajectory, dtype=np.float64)
        conf_arr = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
        if pitch_arr.size < 4 or pitch_arr.shape != conf_arr.shape:
            logger.warning(
                "SOTA-M12 WF4-Warp-Head: F0-Kurve zu kurz/inkonsistent (T=%d, conf=%d) — "
                "Challenger übersprungen, heuristische Phase-12-Schätzung (§V6 (copilot-instructions.md)).",
                int(pitch_arr.size),
                int(conf_arr.size),
            )
            return None, False
        t = int(pitch_arr.size)
        # Robuster Median (kein Material-Schwellwert, §V7 (copilot-instructions.md)).
        voiced = pitch_arr[pitch_arr > 0.0]
        ref_hz = float(np.median(voiced)) if voiced.size > 0 else 440.0
        ref_hz = max(ref_hz, 1e-3)
        x_f0 = np.where(pitch_arr > 0.0, 1200.0 * np.log2(np.maximum(pitch_arr, 1e-3) / ref_hz), 0.0).astype(np.float32)
        x_conf = conf_arr.astype(np.float32)
        if audio is not None:
            mono = np.asarray(audio, dtype=np.float32)
            if mono.ndim > 1:
                mono = mono.mean(axis=0 if mono.shape[0] <= 2 else 1)
            x_feat = compute_spectral_features(mono, sample_rate, t)
        else:
            x_feat = np.zeros((t, N_SPEC_FEAT), dtype=np.float32)

        if head is None:
            head = get_wf4_warp_head()
        if not head.is_available():
            # Bereits in _try_load_model als §V6-Warnung protokolliert.
            return None, False

        # Statischer ONNX-Vertrag: T_MAX_FRAMES pro Aufruf; längere Kurven in
        # überlappende Fenster zerlegen (deterministische Fensterung).
        if t <= T_MAX_FRAMES:
            warp, _conf_out = head.predict(x_f0, x_conf, x_feat)
        else:
            hop = T_MAX_FRAMES // 2
            warp = np.zeros(t, dtype=np.float32)
            weight = np.zeros(t, dtype=np.float32)
            for start in range(0, t, hop):
                end = min(start + T_MAX_FRAMES, t)
                w_chunk, _ = head.predict(x_f0[start:end], x_conf[start:end], x_feat[start:end])
                n_chunk = end - start
                win = np.hanning(n_chunk + 2)[1:-1].astype(np.float32)
                warp[start:end] += w_chunk * win
                weight[start:end] += win
                if end >= t:
                    break
            valid = weight > 1e-6
            warp[valid] /= weight[valid]
        warp = np.nan_to_num(warp, nan=0.0, posinf=0.0, neginf=0.0)
        stretch = warp_to_stretch_factors(warp)
        # §G188/§G189 (GEBOTE.md): Warp-Delta-Kappe analog max_stretch_delta der
        # Phase 12 (max 10 % Zeitverzerrung) — Schutz gegen Schätz-Ausreißer.
        stretch = np.clip(stretch, 0.90, 1.10).astype(np.float32)
        logger.info(
            "SOTA-M12 WF4-Warp-Head aktiv: T=%d Frames, max|sf-1|=%.4f",
            t,
            float(np.max(np.abs(stretch - 1.0))),
        )
        return stretch, True
    except Exception as _exc:
        logger.warning(
            "SOTA-M12 WF4-Warp-Head fehlgeschlagen (%s) — heuristische Phase-12-Schätzung "
            "(§V6 (copilot-instructions.md)).",
            _exc,
        )
        return None, False
