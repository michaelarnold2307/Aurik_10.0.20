"""
Resemblyzer Plugin — Speaker-Embedding (d-vector, GE2E-Loss)
=============================================================

Wrapper um das Resemblyzer-Paket (d-vector, 256-dim) für §2.35c
Singer-Identity-Cosine und §gender_detection Genderklassifikation.

Modell: Resemblyzer VoiceEncoder (GE2E, 256-dim d-vector)
    - Eingabe: 16 kHz Mono float32 (beliebige Länge)
    - Ausgabe: 256-dim L2-normierter Embedding-Vektor

Spec-Referenzen:
    §2.35c:  singer_identity_cosine — VOR und NACH Pipeline messen;
             cos_sim < 0.92 → Phase-Rollback letzte Vokal-Phase
    §0j:     Resemblyzer ist leichtgewichtig → CPU-only (kein GPU-Overhead)
    §3.2:    Singleton + Double-Checked Locking, thread-safe
    §3.1:    NaN/Inf-Guard; Fallback ohne Absturz
    §4.4:    Resemblyzer (dvector, GE2E-Loss) als primäres Speaker-ID-Modell;
             DSP-Fallback: MFCC-Pearson × Centroid-Korrelation

DSP-Fallback:
    Wenn Resemblyzer nicht installiert oder Fehler → embed() gibt None zurück
    → Aufrufer nutzt _compute_singer_identity_dsp() aus vocal_quality_index.py
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import Any, cast

import numpy as np

try:
    import librosa as _librosa
except Exception:
    _librosa = None  # type: ignore[assignment]

try:
    import onnxruntime as _ort
except Exception:
    _ort = None  # type: ignore[assignment]

# ONNX-Fallback-Pfad (§0j, CPU-only): models/resemblyzer/resemblyzer_voice_encoder.onnx
# exportiert 2026-09-13 aus pretrained.pt (opset 17, [B, T, 40] Mels → 256-dim,
# Parität cos=1.0000). Erwartet dieselben Mel-Parameter wie das Package:
# 16 kHz, n_fft=400, hop=160, n_mels=40.
_ONNX_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "resemblyzer",
    "resemblyzer_voice_encoder.onnx",
)
_ONNX_MEL_N_FFT = 400
_ONNX_MEL_HOP = 160
_ONNX_MEL_N_MELS = 40

# Lokales Resemblyzer-Paket aus models/resemblyzer/ einbinden (offline-fähig,
# kein pip install nötig). Pfad wird nur einmalig in sys.path eingetragen.
# Bugfix 2026-09-15: der Pfad zeigte auf "models/rezemblyzer" (Tippfehler mit
# "z") — das Verzeichnis existiert nicht, daher war der Package-Pfad nie
# erreichbar und die Kaskade sprang immer auf ONNX (bzw. None ohne onnxruntime).
_LOCAL_RESEMBLYZER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "resemblyzer",
)
if os.path.isdir(_LOCAL_RESEMBLYZER_DIR) and _LOCAL_RESEMBLYZER_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_RESEMBLYZER_DIR)

logger = logging.getLogger(__name__)


def _ensure_webrtcvad_shim() -> None:
    """webrtcvad-Import-Shim für das vendored Resemblyzer-Paket (§0j offline-fähig).

    Das unveränderte Vendored-Paket (models/resemblyzer, MIT) importiert
    webrtcvad in audio.py auf Modulebene. Ohne das Modul schlägt der
    Package-Pfad der Kaskade (Package→ONNX→None) immer fehl, obwohl torch
    verfügbar ist. Der Shim liefert einen Minimal-Stub: ``trim_long_silences``
    deaktiviert damit die VAD-Trimmung (alles als Sprache gewertet) — für den
    Pre/Post-Witness-Vergleich unkritisch, da beide Seiten identisch
    vorverarbeitet werden. Echtes webrtcvad wird bevorzugt, falls installiert.
    """
    import types

    if "webrtcvad" in sys.modules:
        return
    try:
        import webrtcvad
    except Exception:

        class _Vad:
            """Stub: alle Frames gelten als Sprache (keine VAD-Trimmung)."""

            def __init__(self, mode: int = 3) -> None:
                self._mode = mode

            def set_mode(self, mode: int) -> None:
                self._mode = mode

            def is_speech(self, buf: bytes, sample_rate: int) -> bool:
                return True

        _mod = types.ModuleType("webrtcvad")
        _mod.Vad = _Vad  # type: ignore[attr-defined]
        sys.modules["webrtcvad"] = _mod
        logger.debug("resemblyzer_plugin: webrtcvad nicht installiert — Import-Shim aktiv (keine VAD-Trimmung)")


_ensure_webrtcvad_shim()

try:
    from resemblyzer import VoiceEncoder as _ResemblyzerVoiceEncoder
    from resemblyzer import preprocess_wav as _resemblyzer_preprocess_wav
except Exception:
    _ResemblyzerVoiceEncoder = None
    _resemblyzer_preprocess_wav = None


# ---------------------------------------------------------------------------
# Singleton-Lock (§3.2 — Double-Checked Locking)
# ---------------------------------------------------------------------------
_instance: ResemblyzerPlugin | None = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Plugin-Klasse
# ---------------------------------------------------------------------------
class ResemblyzerPlugin:
    """Resemblyzer VoiceEncoder — Speaker-Embedding für §2.35c Singer-Identity.

    CPU-only (§0j: leichtgewichtiges Modell, kein GPU-Overhead gerechtfertigt).
    Thread-sicherer Singleton via Double-Checked Locking (§3.2).

    Invarianten (§3.1):
        - embed() gibt None zurück wenn Modell nicht verfügbar (kein Absturz)
        - Alle Embedding-Vektoren sind L2-normiert ∈ [-1, 1]^256
        - NaN/Inf in Eingaben werden zu 0.0 bereinigt
    """

    # Resemblyzer erwartet 16 kHz Mono
    MODEL_SR: int = 16_000

    def __init__(self) -> None:
        self._encoder: Any | None = None
        self._preprocess_wav_fn: Callable[..., np.ndarray] | None = None
        self._onnx_session: Any | None = None
        self._load()

    # ------------------------------------------------------------------
    # Laden
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Lädt VoiceEncoder einmalig lazy; warnt bei Fehler, kein Absturz.

        Kaskade: Python-Package → ONNX (models/resemblyzer) → None (DSP-Fallback
        liegt beim Aufrufer). Beide Pfade liefern 256-dim d-vectoren.
        """
        try:
            if _ResemblyzerVoiceEncoder is None or _resemblyzer_preprocess_wav is None:
                raise ImportError("resemblyzer unavailable")

            # CPU-only: Resemblyzer ist leichtgewichtig (§0j)
            self._encoder = _ResemblyzerVoiceEncoder("cpu")
            self._preprocess_wav_fn = _resemblyzer_preprocess_wav
            logger.info("resemblyzer_plugin: VoiceEncoder geladen (256-dim d-vector, CPU, §2.35c)")
        except Exception as exc:
            logger.warning(
                "resemblyzer_plugin: Resemblyzer-Package nicht verfügbar — ONNX-Ersatzpfad wird geprüft: %s", exc
            )
            self._encoder = None
            self._preprocess_wav_fn = None
            self._load_onnx()

    def _load_onnx(self) -> None:
        """Lädt resemblyzer_voice_encoder.onnx via onnxruntime (CPU-only).

        Kein Absturz bei Fehler: _onnx_session bleibt None, embed() gibt None
        zurück und der Aufrufer nutzt den DSP-Fallback (§3.1).
        """
        if _ort is None:
            logger.warning("resemblyzer_plugin: onnxruntime nicht verfügbar — kein ONNX-Pfad")
            return
        if not os.path.exists(_ONNX_MODEL_PATH):
            logger.warning("resemblyzer_plugin: ONNX nicht gefunden (%s) — kein ONNX-Pfad", _ONNX_MODEL_PATH)
            return
        try:
            self._onnx_session = _ort.InferenceSession(_ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
            logger.info("resemblyzer_plugin: VoiceEncoder-ONNX geladen (256-dim d-vector, CPU, §2.35c)")
        except Exception as exc:
            logger.warning("resemblyzer_plugin: ONNX-Laden fehlgeschlagen — DSP-Ersatzpfad aktiv: %s", exc)
            self._onnx_session = None

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True wenn Resemblyzer (Package oder ONNX) geladen und einsatzbereit."""
        return self._encoder is not None or self._onnx_session is not None

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """Berechnet 256-dim d-vector Embedding.

        Args:
            audio: Mono oder Stereo float32 ndarray, beliebige SR.
            sr:    Sample-Rate von audio.

        Returns:
            256-dim L2-normierter Embedding-Vektor (float32) oder None bei Fehler.
        """
        if self._onnx_session is not None and self._encoder is None:
            return self._embed_onnx(audio, sr)
        if self._encoder is None or self._preprocess_wav_fn is None:
            return None
        try:
            # Zu Mono normieren
            mono = _to_mono(audio)
            mono = np.asarray(np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)

            # Auf Resemblyzer-SR resamplen (16 kHz)
            if sr != self.MODEL_SR:
                if _librosa is None:
                    return None
                mono = np.asarray(_librosa.resample(mono, orig_sr=sr, target_sr=self.MODEL_SR), dtype=np.float32)

            # preprocess_wav → normiert + getrimmtes float32
            wav = np.asarray(
                self._preprocess_wav_fn(mono, source_sr=self.MODEL_SR),
                dtype=np.float32,
            )

            # embed_utterance → 256-dim d-vector
            emb = np.asarray(cast(Any, self._encoder).embed_utterance(wav), dtype=np.float32)
            emb = np.asarray(np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
            return emb  # type: ignore[no-any-return]

        except Exception as exc:
            logger.debug("resemblyzer_plugin: embed() Fehler — None zurückgegeben: %s", exc)
            return None

    def _embed_onnx(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """ONNX-Pfad: 16 kHz mono → Volumen-Norm (−30 dBFS, increase-only wie
        Resemblyzer) → Energie-VAD-Trim → Mel (400/160/40) → ONNX → L2-Norm.

        Verwendet dieselben Mel-Parameter wie das Python-Package, damit die
        d-vectoren beider Pfade vergleichbar bleiben.
        """
        try:
            mono = _to_mono(audio)
            mono = np.asarray(np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)

            if sr != self.MODEL_SR:
                if _librosa is None:
                    return None
                mono = np.asarray(_librosa.resample(mono, orig_sr=sr, target_sr=self.MODEL_SR), dtype=np.float32)

            # Volumen-Normierung wie resemblyzer.normalize_volume(increase_only=True)
            rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))) + 1e-12)
            target_rms = 10.0 ** (-30.0 / 20.0)  # −30 dBFS
            if rms < target_rms:
                mono = mono * (target_rms / rms)

            # Energie-basiertes VAD-Trim (Ersatz für webrtcvad im Package):
            # 30-ms-Frames, 10-ms-Hop; behalte Regionen über 1e-5 RMS.
            frame_len = 480
            hop = 160
            if mono.size >= frame_len:
                n_frames = 1 + (mono.size - frame_len) // hop
                idx = np.arange(n_frames) * hop
                fr = np.stack([mono[i : i + frame_len] for i in idx])
                energy = np.mean(np.square(fr), axis=1)
                voice = energy > 1e-5
                if not np.any(voice):
                    return None
                first, last = int(np.argmax(voice)), int(n_frames - 1 - np.argmax(voice[::-1]))
                mono = mono[first * hop : (last + 1) * hop + frame_len]

            if mono.size < frame_len or _librosa is None or self._onnx_session is None:
                return None

            mels = _librosa.feature.melspectrogram(
                y=mono,
                sr=self.MODEL_SR,
                n_fft=_ONNX_MEL_N_FFT,
                hop_length=_ONNX_MEL_HOP,
                n_mels=_ONNX_MEL_N_MELS,
            )
            mel_input = np.ascontiguousarray(mels.T, dtype=np.float32)[np.newaxis, ...]  # [1, T, 40]
            emb = self._onnx_session.run(None, {"mels": mel_input})[0][0]  # [256]
            norm = float(np.linalg.norm(emb)) + 1e-12
            emb = (emb / norm).astype(np.float32)
            _emb_clean: np.ndarray = np.asarray(np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
            return _emb_clean

        except Exception as exc:
            logger.debug("resemblyzer_plugin: _embed_onnx() Fehler — None zurückgegeben: %s", exc)
            return None

    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """L2-normierte Cosinus-Ähnlichkeit ∈ [0, 1] zwischen zwei Embeddings.

        Args:
            emb_a: 256-dim Embedding.
            emb_b: 256-dim Embedding.

        Returns:
            Cosinus-Ähnlichkeit ∈ [0, 1]. NaN-safe.
        """
        a = np.nan_to_num(np.array(emb_a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(np.array(emb_b, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        cos = float(np.dot(a, b) / denom)
        # Resemblyzer-Embeddings sind L2-normiert → cos ∈ [-1, 1]; clippen auf [0, 1]
        return float(np.clip(cos, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Channels-first (2,N) oder samples-first (N,2) → mono (N,)."""
    if audio.ndim == 1:
        mono_audio = np.asarray(audio, dtype=np.float32)
        return mono_audio  # type: ignore[no-any-return]
    if audio.ndim == 2:
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            mono_audio = np.asarray(audio.mean(axis=0), dtype=np.float32)
            return mono_audio  # type: ignore[no-any-return]
        if audio.shape[1] == 2:
            mono_audio = np.asarray(audio.mean(axis=1), dtype=np.float32)
            return mono_audio  # type: ignore[no-any-return]
        if audio.shape[0] == 1:
            mono_audio = np.asarray(audio[0], dtype=np.float32)
            return mono_audio  # type: ignore[no-any-return]
    mono_audio = np.asarray(audio.flatten(), dtype=np.float32)
    return mono_audio  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2)
# ---------------------------------------------------------------------------


def get_resemblyzer_plugin() -> ResemblyzerPlugin:
    """Gibt den thread-sicheren Singleton zurück (Double-Checked Locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ResemblyzerPlugin()
    return _instance
