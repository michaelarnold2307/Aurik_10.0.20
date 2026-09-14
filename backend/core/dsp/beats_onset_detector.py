"""§SOTA-TP-V1: BEATs-Onset-Witness — Konsens mit der Spectral-Flux-Detektion.

Der lokale BEATs-ONNX (`models/beats/beats_iter3.onnx`) ist der ITER3-ENCODER
(fbank [B,T,128] → 768-dim-Tokens [B,T',768]) — KEIN 527-Klassen-Tagger. Für
die Onset-Detektion ist genau das der richtige Export: die zeitliche Ableitung
der Token-Sequenz liefert eine neuronale Onset-Stärkekurve.

Rolle (Hörordnung §8a): ZEUGE, nicht Richter. Die Phasen (phase_08 Superflux,
phase_36 Envelope) behalten ihre Detektion; dieser Witness liefert die
Konsens-Statistik (Anteil bestätigter Onsets) als Metadaten — die
adaptive-Schwelle-Stufe bleibt ein dokumentierter Folge-Schritt (Never-worsen:
Onsets wegnehmen würde Transienten-Schutz verringern).

Befund 2026-09-14: plugins/beats_plugin.py füttert den ENCODER-ONNX mit
Roh-Audio statt fbank (Rank-Mismatch) und interpretiert Token-Output als
527-Scores — der Tagger-Pfad läuft nie (stiller DSP-Fallback, §V6 (copilot-instructions.md)).
Korrektur ist ein eigener Task (Tagger-Head-ONNX
oder Head auf Tokens — GPU); dieses Modul umgeht das korrekt mit fbank-Input.

Determinismus (§G5 (GEBOTE.md)): festes Fenster, ONNX-Inferenz fix.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_ONNX_PATH = _ROOT / "models" / "beats" / "beats_iter3.onnx"
_MODEL_SR = 16000
_MAX_WINDOW_S = 10.0
_FRAME_S = 0.010  # 10-ms-Hop der BEATs-fbank

_lock = threading.Lock()
_session: object | None = None
_session_tried: bool = False


def beats_available() -> bool:
    return _ONNX_PATH.is_file()


def _get_session() -> object | None:
    global _session, _session_tried  # pylint: disable=global-statement
    if _session is not None:
        return _session
    with _lock:
        if _session is not None or _session_tried:
            return _session
        _session_tried = True
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            _session = ort.InferenceSession(str(_ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
            logger.info("beats_onset_detector: BEATs-Encoder-ONNX geladen (CPU, §SOTA-TP-V1)")
            return _session
        except Exception as _exc:
            logger.warning("beats_onset_detector: ONNX nicht ladbar (%s) — Witness übersprungen.", _exc)
            return None


def _unload_session() -> None:
    global _session  # pylint: disable=global-statement
    with _lock:
        _session = None


def fbank_16k(mono_16k: np.ndarray) -> np.ndarray:
    """Kaldi-fbank [1, T, 128] — exakt die BEATs-iter3-Konvention (25/10 ms)."""
    import torch
    import torchaudio

    wav = np.nan_to_num(np.asarray(mono_16k, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    fb = torchaudio.compliance.kaldi.fbank(
        wav_t,
        num_mel_bins=128,
        frame_length=25.0,
        frame_shift=10.0,
        sample_frequency=_MODEL_SR,
    )
    _fb_out: np.ndarray = np.asarray(fb.numpy(), dtype=np.float32)[np.newaxis, ...]
    return _fb_out


def _tokens_for_window(mono_16k: np.ndarray) -> np.ndarray | None:
    session = _get_session()
    if session is None:
        return None
    try:
        fb = fbank_16k(mono_16k)
        out = session.run(None, {"fbank": fb})[0]  # type: ignore[attr-defined]
        tokens = np.asarray(out, dtype=np.float32)  # [1, T', 768]
        _tokens_out: np.ndarray = tokens[0]
        return _tokens_out
    except Exception as _exc:
        logger.warning("beats_onset_detector: Inferenz fehlgeschlagen (%s) — Witness übersprungen.", _exc)
        return None


def beats_pooled_embedding(audio: np.ndarray, sr: int, max_window_s: float = _MAX_WINDOW_S) -> np.ndarray | None:
    """BEATs-Encoder-Embedding (768-dim, Mean-Pool über die Token-Zeitachse).

    Nutzt den ENCODER-Export sinnvoll (plugins/beats_plugin.py hat keinen
    Tagger-Head-ONNX — siehe Befund 2026-09-14). Deterministisch (festes
    Fenster); None bei Fehler/fehlendem Modell (§V6 (copilot-instructions.md)).
    """
    if not beats_available():
        return None
    try:
        from math import gcd

        from scipy.signal import resample_poly

        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mono = arr.mean(axis=0) if arr.ndim == 2 else arr
        if mono.size < _MODEL_SR // 2:
            return None
        if sr != _MODEL_SR:
            g = gcd(sr, _MODEL_SR)
            mono16 = resample_poly(mono, _MODEL_SR // g, sr // g).astype(np.float32)
        else:
            mono16 = mono
        n_win = int(max_window_s * _MODEL_SR)
        if len(mono16) > n_win:
            start16 = (len(mono16) - n_win) // 2
            mono16 = mono16[start16 : start16 + n_win]
        tokens = _tokens_for_window(mono16)
        if tokens is None:
            return None
        emb = np.asarray(tokens.mean(axis=0), dtype=np.float32)
        norm = float(np.linalg.norm(emb)) + 1e-12
        _emb_out: np.ndarray = (emb / norm).astype(np.float32)
        return _emb_out
    except Exception as _exc:
        logger.warning("beats_onset_detector: Embedding fehlgeschlagen (%s) — None.", _exc)
        return None


def beats_onset_curve(
    audio: np.ndarray, sr: int, max_window_s: float = _MAX_WINDOW_S
) -> tuple[np.ndarray | None, int, int]:
    """Neurale Onset-Stärkekurve über ein zentriertes Fenster.

    Returns:
        (curve, covered_start, covered_end): curve hat die Länge des Eingangs
        (Sample-Grid bei `sr`); außerhalb [covered_start, covered_end) ist sie 0.
        None-Curve ⇒ Witness nicht verfügbar (§V6 (copilot-instructions.md)).
    """
    if not beats_available():
        return None, 0, 0
    arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mono = arr.mean(axis=0) if arr.ndim == 2 else arr
    if mono.size < _MODEL_SR // 2:
        return None, 0, 0
    try:
        from math import gcd

        from scipy.signal import resample_poly

        if sr != _MODEL_SR:
            g = gcd(sr, _MODEL_SR)
            mono16 = resample_poly(mono, _MODEL_SR // g, sr // g).astype(np.float32)
        else:
            mono16 = mono
        n_win = int(max_window_s * _MODEL_SR)
        if len(mono16) <= n_win:
            seg = mono16
            start16 = 0
        else:
            start16 = (len(mono16) - n_win) // 2
            seg = mono16[start16 : start16 + n_win]
        tokens = _tokens_for_window(seg)
        if tokens is None or tokens.shape[0] < 4:
            return None, 0, 0
        # Frame-Differenz der Token-Sequenz (L2), robust normalisiert.
        diff = np.linalg.norm(np.diff(tokens, axis=0), axis=1)
        denom = np.median(diff[diff > 0]) + 1e-9 if np.any(diff > 0) else 1.0
        curve_frames = diff / (denom * 3.0)
        curve_frames = np.clip(curve_frames, 0.0, 1.0)
        # Frame-Grid: Token-Frames decken das Fenster ab; auf das Eingangs-Grid legen.
        n_in = len(mono)
        curve = np.zeros(n_in, dtype=np.float32)
        covered_start = int(round(start16 * sr / _MODEL_SR))
        win_len_in = int(round(len(seg) * sr / _MODEL_SR))
        covered_end = min(n_in, covered_start + win_len_in)
        xs = np.linspace(covered_start, covered_end - 1, len(curve_frames))
        curve[covered_start:covered_end] = np.interp(np.arange(covered_start, covered_end), xs, curve_frames).astype(
            np.float32
        )
        return curve, covered_start, covered_end
    except Exception as _exc:
        logger.warning("beats_onset_detector: Onset-Kurve fehlgeschlagen (%s) — Witness übersprungen.", _exc)
        return None, 0, 0


def onset_witness_stats(
    onset_times_s: np.ndarray,
    onset_strengths: np.ndarray,
    curve: np.ndarray,
    sr: int,
    covered_start: int,
    covered_end: int,
    tol_ms: float = 60.0,
) -> dict[str, float | int]:
    """Konsens-Statistik: Anteil der Flux-Onsets, die die BEATs-Kurve bestätigt.

    Ein Onset gilt als bestätigt, wenn im ±tol_ms-Fenster um seine Zeit die
    Kurve ein lokales Maximum über 0,15 erreicht.
    """
    tol = int(tol_ms / 1000.0 * sr)
    confirmed = 0
    peaks: list[float] = []
    for t_s in np.asarray(onset_times_s, dtype=np.float64):
        idx = int(round(float(t_s) * sr))
        lo, hi = max(covered_start, idx - tol), min(covered_end, idx + tol)
        if hi <= lo:
            continue
        seg = curve[lo:hi]
        if seg.size == 0:
            continue
        local = float(seg.max())
        peaks.append(local)
        if local > 0.15:
            confirmed += 1
    n = int(len(onset_times_s))
    return {
        "n_onsets": n,
        "n_confirmed": confirmed,
        "agreement_ratio": round(confirmed / n, 3) if n else 1.0,
        "mean_curve_peak_at_onsets": round(float(np.mean(peaks)), 3) if peaks else 0.0,
        "covered_s": round((covered_end - covered_start) / max(1, sr), 2),
    }
