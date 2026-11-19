"""SOTA-M9 (D-K4-2): Nicht-kausaler TF-GridNet-Klassen-Vollband-Denoiser.

Optionale Quality-Mode-Stufe der Spektral-Entrauschung. Architektur-Klasse:
TF-GridNet (Zhang et al., ICASSP 2023) — grid-basierte TF-Verarbeitung mit
Intra-Frame-Vollband- und Inter-Frame-Cross-Frame-Blöcken, strikt nicht-kausal
(bidirektionale Rekurrenz über Frequenz UND Zeit). DeepFilterNet (DFN) bleibt
Default (§III.9 (copilot-instructions.md): kleine lokale Modelle bevorzugen).

Vertrag:
    - Flag: ``backend.core.music_model_flags.use_tf_gridnet_quality`` (SOTA-M9,
      vorhanden, Default False). False ⇒ keine Verhaltensänderung. Aktivierung
      erst nach Never-worsen-Gate + Hörordnungs-Sign-off (§v10.802
      (copilot-instructions.md)).
    - ONNX-Modell (optional): models/tf_gridnet/tf_gridnet_quality.onnx.
      Input ``log_mag`` float32 (1, F, T) mit F = N_FFT//2+1 und
      T = model_t_frames(); log_mag = log1p(|STFT|). Output ``mask`` float32
      (1, F, T), Werte in [0, 1] (Sigmoid im Modell; der Wrapper clippt
      defensiv). Kanal-unabhängig: Stereo läuft als zwei Modell-Aufrufe.
      Statischer Export über feste CHUNK-Fenster — das Training exportiert mit
      den Fenster-Konstanten dieses Moduls (scripts/train_tf_gridnet_denoiser.py).
    - Stereo-Layout-Invariante (AGENTS.md §3): Eingabe (N,), (2, N) oder (N, 2);
      intern channels-first (C, N); die Rückgabe hat das Eingabe-Layout.
    - Determinismus (§G5 (copilot-instructions.md)): ONNX-CPU-Session mit fixen
      Thread-Zahlen (§III.9 (copilot-instructions.md): ohne Paritäts-Nachweis
      kein GPU-EP), feste Fenster-Rasterung, keine zeitabhängige Logik.
    - Silent-Failure-Verbot (§V6 (copilot-instructions.md)): jeder Fallback
      (kein Modell, Vertrags-Bruch, Inferenzfehler, Never-worsen-Rücksprung)
      mit logger.warning + Begründung — kein stilles Degradieren.
    - NaN/Inf-Schutz (§III.6 (copilot-instructions.md)): Ausgabe endlich.
    - Never-worsen (§v10.26 (.github/specs/v10.26_autonomer_never_worsen_arbiter.md)):
      grober deterministischer DSP-Proxy oder injizierter Scorer;
      Verschlechterung ⇒ bit-identischer Rücksprung auf die Eingabe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from backend.core import audio_layout, music_model_flags

logger = logging.getLogger(__name__)  # §III.7 (copilot-instructions.md)

# ── Signal-/Modell-Konstanten (Vertrag mit scripts/train_tf_gridnet_denoiser.py) ──

TARGET_SR = 48_000
N_FFT = 1024
HOP = 256
CHUNK_SEC = 4.0
OVERLAP_SEC = 0.5
DEFAULT_MODEL_SUBPATH = "models/tf_gridnet/tf_gridnet_quality.onnx"
#: Grobe Proxy-Schwelle in Score-Einheiten (≈ dB) für den Never-worsen-Rücksprung.
NEVER_WORSEN_MARGIN = 0.5

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_WINDOW_CACHE: dict[int, np.ndarray] = {}


# ── Layout-Helfer (Stereo-Layout-Invariante, AGENTS.md §3) ───────────────────


def _to_channels_first(audio: np.ndarray) -> tuple[np.ndarray, str]:
    """Normalisiert nach (C, N); liefert (Array, Layout-Kennung).

    Layout-Entscheidung ausschließlich über `backend/core/audio_layout.py`
    (§G9 (copilot-instructions.md): Layout-EINE-Quelle, Muster D-K3-6):
    `to_channels_first` transponiert bit-identisch nur wenn nötig — dieselbe
    Rückgabe (Identität) markiert das Nicht-Transponieren; (2, 2)-artige
    ambige Shapes bleiben in der Pipeline-invarianten (C, N)-Lesart
    (AGENTS.md §3).
    """
    x = np.asarray(audio)
    if x.ndim == 1:
        return x[np.newaxis, :].astype(np.float32, copy=False), "mono"
    if x.ndim != 2:
        raise ValueError(f"Nicht unterstütztes Audio-Layout: shape={x.shape} (erwartet (N,) oder 2-D)")
    channels = audio_layout.to_channels_first(x)
    layout = "nc" if channels is not x else "cn"
    return np.ascontiguousarray(channels).astype(np.float32, copy=False), layout


def _restore_layout(channels_first: np.ndarray, layout: str) -> np.ndarray:
    res: np.ndarray
    if layout == "mono":
        res = np.ascontiguousarray(channels_first[0])
    elif layout == "nc":
        res = np.ascontiguousarray(channels_first.T)
    else:
        res = np.ascontiguousarray(channels_first)
    return res


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Layout-bewusste Mono-Summe für die Proxy-Bewertung (nie audio[:, 0])."""
    x = np.asarray(audio)
    res: np.ndarray
    if x.ndim == 1:
        res = x.astype(np.float64)
    elif x.shape[0] in (1, 2) or x.shape == (2, 2):
        res = x.astype(np.float64).mean(axis=0)
    else:
        res = x.astype(np.float64).mean(axis=1)
    return res


# ── STFT-Analyse/-Synthese (deterministisch, §G5 (copilot-instructions.md)) ──


def _analysis_window() -> np.ndarray:
    win = _WINDOW_CACHE.get(N_FFT)
    if win is None:
        win = np.hanning(N_FFT).astype(np.float32)
        _WINDOW_CACHE[N_FFT] = win
    return win


def window_samples() -> int:
    """Fensterlänge in Samples — jedes Modell-Fenster ist exakt so lang."""
    return int(round(CHUNK_SEC * TARGET_SR))


def model_t_frames() -> int:
    """Spektrogramm-Zeitfenster je Modell-Aufruf (Vertrag mit dem ONNX-Export)."""
    pad = N_FFT // 2
    tail = N_FFT // 2 + HOP
    l_pad = pad + window_samples() + tail
    return (l_pad - N_FFT) // HOP + 1


def stft_magnitude(x: np.ndarray) -> np.ndarray:
    """Betragsspektrum (F, T) float32 eines Mono-Fensters — Feature-Vertrag."""
    spec, _ = _analysis(np.asarray(x, dtype=np.float32))
    return np.abs(spec).astype(np.float32)  # type: ignore[no-any-return]


def _analysis(x: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """(N,) float32 → complex64 (F, T) plus Synthese-Kontext."""
    n = int(x.shape[-1])
    pad = N_FFT // 2
    tail = N_FFT // 2 + HOP
    xp = np.zeros(pad + n + tail, dtype=np.float32)
    xp[pad : pad + n] = x
    n_frames = (xp.shape[0] - N_FFT) // HOP + 1
    win = _analysis_window()
    spec = np.empty((N_FFT // 2 + 1, n_frames), dtype=np.complex64)
    for t in range(n_frames):
        seg = xp[t * HOP : t * HOP + N_FFT] * win
        spec[:, t] = np.fft.rfft(seg)
    return spec, {"n": n, "n_frames": n_frames}


def _synthesis(spec: np.ndarray, ctx: dict[str, int]) -> np.ndarray:
    """complex64 (F, T) → (N,) float32; Fenster-Quadrat-OLA, exakte Rückfaltung."""
    n = ctx["n"]
    n_frames = ctx["n_frames"]
    pad = N_FFT // 2
    tail = N_FFT // 2 + HOP
    out = np.zeros(pad + n + tail, dtype=np.float64)
    env = np.zeros(pad + n + tail, dtype=np.float64)
    win = _analysis_window().astype(np.float64)
    w2 = win * win
    for t in range(n_frames):
        seg = np.fft.irfft(spec[:, t], n=N_FFT).astype(np.float64) * win
        out[t * HOP : t * HOP + N_FFT] += seg
        env[t * HOP : t * HOP + N_FFT] += w2
    valid = env > 1e-8
    out[valid] /= env[valid]
    out[~valid] = 0.0
    res_ola: np.ndarray = out[pad : pad + n].astype(np.float32)
    return res_ola


# ── Never-worsen-Harness (§v10.26) ───────────────────────────────────────────


def default_quality_scorer(mono: np.ndarray, sample_rate: int) -> float:
    """Deterministischer DSP-Proxy (höher = besser), ohne ML-Load.

    Kombiniert Pegel (RMS), HF-Erhalt (Brillanz) und Crest-Faktor (Transienten).
    Bewusst grob: die finale Abnahme läuft über das Never-worsen-Gate plus
    Hörordnungs-Sign-off (§v10.802 (copilot-instructions.md)).
    """
    x = np.nan_to_num(np.asarray(mono, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x * x)))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    if x.size >= 4:
        spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
        power = spec * spec
        total = float(power.sum()) + 1e-12
        freqs = np.fft.rfftfreq(x.size, d=1.0 / max(int(sample_rate), 1))
        hf_ratio = float(power[freqs > 6000.0].sum()) / total
    else:
        hf_ratio = 0.0
    peak = float(np.max(np.abs(x)))
    crest_db = float(np.clip(20.0 * np.log10((peak + 1e-12) / (rms + 1e-12)), 0.0, 20.0))
    return rms_db + 10.0 * hf_ratio + 0.5 * crest_db  # type: ignore[no-any-return]


def never_worsen_select(
    input_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    *,
    scorer: Callable[[np.ndarray, int], float] | None = None,
    margin: float = NEVER_WORSEN_MARGIN,
) -> np.ndarray:
    """Liefert den Kandidaten — oder bit-identisch die Eingabe (§V6-Warning).

    Args:
        input_audio:  Referenz (Layout beliebig, wie beim Denoiser).
        candidate_audio: verarbeiteter Kandidat, identisches Layout/Shape.
        sample_rate: Abtastrate.
        scorer: optional injizierbarer Proxy (mono, sr) -> float; Default
            ``default_quality_scorer``.
        margin: erlaubte Score-Einbuße, bevor zurückgesprungen wird.
    """
    inp = np.asarray(input_audio)
    cand = np.asarray(candidate_audio)
    if inp.shape != cand.shape:
        raise ValueError(f"Never-worsen: Shape-Mismatch input={inp.shape} candidate={cand.shape}")
    score_fn = scorer if scorer is not None else default_quality_scorer
    score_in = float(score_fn(_to_mono(inp), sample_rate))
    score_out = float(score_fn(_to_mono(cand), sample_rate))
    if score_out < score_in - margin:
        logger.warning(
            "SOTA-M9 Never-worsen (§V6 (copilot-instructions.md)): Kandidat verschlechtert den Proxy "
            "(in=%.2f, out=%.2f, margin=%.2f) — bit-identischer Rücksprung auf die Eingabe.",
            score_in,
            score_out,
            margin,
        )
        res_rev: np.ndarray = np.array(inp, copy=True)
        return res_rev
    res_cand: np.ndarray = np.ascontiguousarray(cand)
    return res_cand


# ── ONNX-Inferenz (TF-GridNet-Klassen-Denoiser) ───────────────────────────────


class TFGridNetDenoiser:
    """Nicht-kausaler Vollband-Denoiser der TF-GridNet-Klasse (ONNX-Inferenz).

    Chunked Overlap-Add mit festen Fenstern (statischer ONNX-Vertrag), pro Kanal
    ein Modell-Aufruf, optionaler Never-worsen-Abschluss.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        never_worsen: bool = True,
        scorer: Callable[[np.ndarray, int], float] | None = None,
    ) -> None:
        self._session: Any | None = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._model_ok: bool = False
        self._never_worsen = never_worsen
        self._scorer = scorer
        if model_path is None:
            self._model_path = _PROJECT_ROOT / DEFAULT_MODEL_SUBPATH
        else:
            self._model_path = Path(model_path)
        self._try_load_model()

    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        if not self._model_path.exists():
            logger.warning(
                "SOTA-M9 TF-GridNet-ONNX nicht gefunden: %s — Quality-Mode-Stufe bleibt inaktiv "
                "(§V6 (copilot-instructions.md): DFN/Bestand als Fallback).",
                self._model_path,
            )
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("TFGridNetDenoiser", size_gb=0.60):
                    logger.warning(
                        "SOTA-M9 TF-GridNet: ML-Memory-Budget erschöpft — Quality-Mode-Stufe inaktiv "
                        "(§V6 (copilot-instructions.md): DFN/Bestand als Fallback)."
                    )
                    return
            except Exception as _exc:
                logger.debug("SOTA-M9 TF-GridNet: Speicherobergrenze nicht verfügbar (unkritisch): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.log_severity_level = 3
            # §III.9 (copilot-instructions.md): ohne Paritäts-Nachweis gegen
            # ONNX-CPU kein GPU-EP — der ONNX-Pfad bleibt CPU (§V6, copilot-instructions.md).
            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            if not self._contract_ok():
                self._session = None
                return
            self._model_ok = True
            logger.info(
                "SOTA-M9 TF-GridNet ONNX geladen: %s | In=%s | Out=%s",
                self._model_path,
                self._input_name,
                self._output_name,
            )
        except Exception as _exc:
            logger.warning(
                "SOTA-M9 TF-GridNet-ONNX nicht ladbar (%s) — Quality-Mode-Stufe inaktiv "
                "(§V6 (copilot-instructions.md): DFN/Bestand als Fallback).",
                _exc,
            )

    def _contract_ok(self) -> bool:
        """Prüft Input-Vertrag (1, F, T): statische Dims müssen passen (§V6, copilot-instructions.md)."""
        sess = self._session
        if sess is None:
            return False
        try:
            shape = sess.get_inputs()[0].shape
        except Exception:
            shape = None
        if not shape or len(shape) != 3:
            logger.warning(
                "SOTA-M9 TF-GridNet: Modell-Input hat %s statt (1, F, T) — Vertrags-Bruch, "
                "Quality-Mode-Stufe inaktiv (§V6 (copilot-instructions.md)).",
                shape,
            )
            return False
        expected = (1, N_FFT // 2 + 1, model_t_frames())
        for axis, dim in enumerate(shape):
            if isinstance(dim, int) and dim != expected[axis]:
                logger.warning(
                    "SOTA-M9 TF-GridNet: Modell-Dim Achse %d = %d ≠ %d (Vertrag (1, F=%d, T=%d)) — "
                    "Vertrags-Bruch, Quality-Mode-Stufe inaktiv (§V6 (copilot-instructions.md)).",
                    axis,
                    dim,
                    expected[axis],
                    expected[1],
                    expected[2],
                )
                return False
        return True

    def is_available(self) -> bool:
        return self._model_ok and self._session is not None

    # ------------------------------------------------------------------

    def _infer_mask(self, log_mag: np.ndarray) -> np.ndarray:
        """(F, T) log_mag → (F, T) Mask in [0, 1]."""
        sess = self._session
        if sess is None:
            # Ehrlicher Abbruch statt stiller Fallback (§V6 (copilot-instructions.md)).
            raise RuntimeError("TF-GridNet: ONNX-Session nicht geladen — Inferenz unmöglich")
        feed = np.ascontiguousarray(log_mag[np.newaxis], dtype=np.float32)
        out = sess.run([self._output_name], {self._input_name: feed})[0]
        mask = np.asarray(out, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[0]
        res_mask: np.ndarray = np.clip(mask, 0.0, 1.0)
        return res_mask

    def _process_window(self, window: np.ndarray) -> np.ndarray:
        """(C, window_samples()) float32 → gedämpftes Fenster (C, N) float32."""
        channels = []
        for c in range(window.shape[0]):
            spec, ctx = _analysis(window[c])
            mag = np.abs(spec).astype(np.float32)
            mask = self._infer_mask(np.log1p(mag))
            channels.append(_synthesis(spec * mask.astype(np.complex64), ctx))
        res_win: np.ndarray = np.stack(channels, axis=0).astype(np.float32)
        return res_win

    def denoise(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        scorer: Callable[[np.ndarray, int], float] | None = None,
    ) -> np.ndarray | None:
        """Nicht-kausale Vollband-Entrauschung; None, wenn nicht verfügbar.

        Args:
            audio: (N,), (2, N) oder (N, 2) float; Layout bleibt erhalten.
            sample_rate: 48 kHz erwartet (Vertrag), sonst None (§V6-Warning).
            scorer: optionaler Never-worsen-Proxy (ersetzt den Konstruktor-Wert).
        """
        if not self.is_available():
            logger.warning(
                "SOTA-M9 TF-GridNet ohne Modell — keine Inferenz (§V6 (copilot-instructions.md): "
                "DFN/Bestand als Fallback)."
            )
            return None
        if int(sample_rate) != TARGET_SR:
            logger.warning(
                "SOTA-M9 TF-GridNet: sr=%s ≠ %d — Vertrags-Abbruch (§V6 (copilot-instructions.md): "
                "DFN/Bestand als Fallback).",
                sample_rate,
                TARGET_SR,
            )
            return None

        x_in, layout = _to_channels_first(audio)
        x = np.nan_to_num(x_in, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        n = x.shape[-1]
        if n < N_FFT:
            logger.info("SOTA-M9 TF-GridNet: Signal zu kurz (%d Samples) — unverändert übernommen.", n)
            return _restore_layout(x, layout)

        win_len = window_samples()
        hop = win_len - int(round(OVERLAP_SEC * TARGET_SR))
        if n <= win_len:
            padded = np.zeros((x.shape[0], win_len), dtype=np.float32)
            padded[:, :n] = x
            out = self._process_window(padded)[:, :n]
        else:
            starts = list(range(0, n - win_len + 1, hop))
            if starts[-1] + win_len < n:
                starts.append(n - win_len)
            blend = np.hanning(win_len).astype(np.float64) + 1e-3
            acc = np.zeros((x.shape[0], n), dtype=np.float64)
            env = np.zeros(n, dtype=np.float64)
            for s in starts:
                rec = self._process_window(x[:, s : s + win_len]).astype(np.float64)
                acc[:, s : s + win_len] += rec * blend
                env[s : s + win_len] += blend
            valid = env > 1e-8
            acc[:, valid] /= env[valid]
            acc[:, ~valid] = 0.0
            out = acc.astype(np.float32)

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)  # §III.6

        if self._never_worsen:
            scorer_fn = scorer if scorer is not None else self._scorer
            out = np.asarray(never_worsen_select(x, out, int(sample_rate), scorer=scorer_fn), dtype=np.float32)
        return _restore_layout(out, layout)


# ── Singleton (§V8 (copilot-instructions.md): Session ohne Song-Zustand) ──────

_tfg_singleton: TFGridNetDenoiser | None = None


def get_tf_gridnet_denoiser() -> TFGridNetDenoiser:
    """Prozess-Singleton der ONNX-Inferenz (zustandslos, pro Song aufrufbar)."""
    global _tfg_singleton  # pylint: disable=global-statement
    if _tfg_singleton is None:
        _tfg_singleton = TFGridNetDenoiser()
    return _tfg_singleton


# ── Quality-Mode-Stufe (Flag-Gate, §V6-Fallback) ──────────────────────────────


def apply_tf_gridnet_quality_stage(
    audio: np.ndarray,
    sample_rate: int,
    *,
    denoiser: TFGridNetDenoiser | None = None,
) -> tuple[np.ndarray, bool]:
    """Optionale Quality-Mode-Stufe NUR hinter ``use_tf_gridnet_quality``.

    Returns:
        (Audio, used). Flag aus ⇒ exakt die Eingabe (identisches Objekt, keine
        Verhaltensänderung). Jeder Fehler hinter dem Flag ⇒ logger.warning
        (§V6 (copilot-instructions.md)) + DFN/Bestand als Fallback.
    """
    if not music_model_flags.use_tf_gridnet_quality:
        return audio, False
    try:
        active = denoiser if denoiser is not None else get_tf_gridnet_denoiser()
        if not active.is_available():
            logger.warning(
                "SOTA-M9 TF-GridNet-Quality-Stage: Modell nicht verfügbar — DFN/Bestand als Fallback "
                "(§V6 (copilot-instructions.md))."
            )
            return audio, False
        result = active.denoise(audio, sample_rate)
        if result is None:
            logger.warning(
                "SOTA-M9 TF-GridNet-Quality-Stage ohne Inferenz-Ergebnis — DFN/Bestand als Fallback "
                "(§V6 (copilot-instructions.md))."
            )
            return audio, False
        return result, True
    except Exception as _exc:
        logger.warning(
            "SOTA-M9 TF-GridNet-Quality-Stage fehlgeschlagen (%s) — DFN/Bestand als Fallback "
            "(§V6 (copilot-instructions.md)).",
            _exc,
        )
        return audio, False


# ── Torch-Architektur (Training/Export; lazy Import) ──────────────────────────


def build_tf_gridnet_model(n_freq: int = N_FFT // 2 + 1, hidden: int = 64, n_blocks: int = 2) -> Any:
    """TF-GridNet-Klassen-Netz (nicht-kausal, Vollband) als Torch-Modell.

    Vertrag: forward(log_mag (B, F, T)) → mask (B, F, T) in (0, 1). Der
    ONNX-Export (scripts/train_tf_gridnet_denoiser.py) ist statisch über die
    CHUNK-Fenster dieses Moduls (torch 2.11: dynamische LSTM-Exporte scheitern
    an Dynamo-Shape-Constraints — statischer Export ist der dokumentierte Weg).

    - Intra-Frame-Block: BiLSTM über die F-Achse (ALLE Bins, „Vollband").
    - Inter-Frame-Block: BiLSTM über die T-Achse („nicht-kausal": Vergangenheit
      UND Zukunft, Zhang et al., ICASSP 2023).
    """
    import torch
    import torch.nn as nn

    class _IntraFrameBlock(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
            self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, h: Any) -> Any:
            b, f, t, _ = h.shape
            seq = h.permute(0, 2, 1, 3).reshape(b * t, f, -1)
            y, _ = self.lstm(seq)
            y = self.proj(y).reshape(b, t, f, -1).permute(0, 2, 1, 3)
            return self.norm(h + y)

    class _InterFrameBlock(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
            self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, h: Any) -> Any:
            b, f, t, _ = h.shape
            seq = h.reshape(b * f, t, -1)
            y, _ = self.lstm(seq)
            y = self.proj(y).reshape(b, f, t, -1)
            return self.norm(h + y)

    class TFGridNetClassModel(nn.Module):
        def __init__(self, freq_bins: int, hidden_dim: int, blocks: int) -> None:
            super().__init__()
            self.freq_emb = nn.Parameter(torch.randn(freq_bins, hidden_dim) * 0.02)
            self.in_proj = nn.Linear(1, hidden_dim)
            layers: list[nn.Module] = []
            for i in range(blocks):
                layers.append(_IntraFrameBlock(hidden_dim))
                layers.append(_InterFrameBlock(hidden_dim))
            self.blocks = nn.ModuleList(layers)
            self.out_proj = nn.Linear(hidden_dim, 1)

        def forward(self, log_mag: Any) -> Any:
            h = self.in_proj(log_mag.unsqueeze(-1)) + self.freq_emb[None, :, None, :]
            for blk in self.blocks:
                h = blk(h)
            return torch.sigmoid(self.out_proj(h).squeeze(-1))

    return TFGridNetClassModel(n_freq, hidden, n_blocks)
