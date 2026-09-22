#!/usr/bin/env python3
"""
§v10.130: MERT Quality Gate — leverages MERT for music-aware quality assessment.

MERT (117M, 160k+ hours of music training) KNOWS what good music sounds like.
Instead of forcing it to denoise (which requires a huge decoder), we use it as
a QUALITY GATE that scores audio segments and guides the denoising pipeline.

Architecture:
  Audio → MERT features → Quality Score (0-100)
    ↓
  Score < 40: Heavy denoising (DFN strength=1.0)
  Score 40-70: Medium denoising (DFN strength=0.5)
  Score > 70: Light touch-up (DFN strength=0.2)
  Score > 90: Pass-through

Quality dimensions:
  - Harmonic clarity: ratio of harmonic to noise energy
  - Spectral balance: deviation from natural music spectrum
  - Transient preservation: attack clarity
  - Overall naturalness: MERT embedding distance from "clean music" centroid

Memory: ~500 MB (MERT ONNX) + <1 MB (quality classifier)
Latency: ~30ms per 2s chunk (GPU) or ~100ms (CPU)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Optional, cast

import numpy as np
from scipy.signal import resample_poly

from backend.file_import import load_audio_file

log = logging.getLogger(__name__)

_PROJECT = Path(__file__).resolve().parent.parent

MERT_SR = 16000
TARGET_SR = 48000
CHUNK_SEC = 2.0
OVERLAP = 0.5

# Quality thresholds
QUALITY_HEAVY = 40  # Below this: aggressive denoising
QUALITY_MEDIUM = 70  # 40-70: moderate denoising
QUALITY_LIGHT = 90  # 70-90: light touch-up


class MERTQualityGate:
    """Scores audio quality using frozen MERT embeddings."""

    def __init__(
        self,
        mert_onnx: str = "models/mert/mert.onnx",
        device: str = "cuda",
        gpu_id: int = 0,
    ):
        # §V6 (copilot-instructions.md)/§III.9: EP-Auswahl nur nach Verfügbarkeit; provider_options muss
        # dieselbe Länge wie providers haben (ort wirft sonst „EP Error“ beim
        # CPU-Fallback — Produktionsbefund 2026-09-20: Score-Ausfall im Gate).
        try:
            import onnxruntime as ort  # pylint: disable=import-outside-toplevel
        except Exception as _ort_exc:
            raise RuntimeError(f"onnxruntime nicht verfügbar: {_ort_exc}") from _ort_exc

        _available = ort.get_available_providers()
        if device == "cuda" and "ROCMExecutionProvider" in _available:
            providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            provider_options: list[dict] | None = [{"device_id": str(gpu_id)}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = None

        mert_path = Path(mert_onnx)
        if not mert_path.is_absolute():
            mert_path = _PROJECT / mert_path
        if not mert_path.exists():
            raise FileNotFoundError(f"MERT ONNX not found: {mert_path}")

        self.session = ort.InferenceSession(
            str(mert_path),
            providers=providers,
            provider_options=provider_options,
        )

        # Reference "clean music" embedding centroid (pre-computed from clean corpus)
        self._clean_centroid: np.ndarray | None = None  # [768]
        self._load_or_compute_centroid()

        self._warmed = False
        log.info("MERT Quality Gate: initialized")

    def _load_or_compute_centroid(self):
        """Lädt das Clean-Musik-Zentroid; self-heal bei Dimensions-Mismatch.

        mert_330m.onnx liefert 1024-dim Features — ältere Zentroide (768-dim,
        MERT-95m/fairseq) würden in score_chunk einen Dot-Dim-Fehler werfen.
        """
        centroid_path = _PROJECT / "models" / "mert" / "clean_music_centroid.npy"
        expected_dim = int(self.session.get_outputs()[0].shape[-1])
        if centroid_path.exists():
            loaded = np.load(centroid_path)
            if loaded.ndim == 1 and loaded.shape[0] == expected_dim:
                self._clean_centroid = loaded
                log.info("MERT Quality Gate: Clean-Zentroid geladen (%d-dim)", expected_dim)
                return
            log.warning(
                "MERT Quality Gate: Zentroid-Dimensionen passen nicht (%s != %d) — "
                "Neuberechnung aus corpus/*/clean/*.wav",
                getattr(loaded, "shape", None),
                expected_dim,
            )
        self._clean_centroid = self._compute_clean_centroid(expected_dim)

    def _compute_clean_centroid(self, expected_dim: int) -> np.ndarray:
        """Mittelt MERT-Features über alle Clean-Dateien des Echt-Audio-Corpus (§15.2)."""
        clean_files = sorted((_PROJECT / "corpus").rglob("*clean*.wav"))
        if not clean_files:
            log.warning("MERT Quality Gate: keine Corpus-Clean-Dateien — Scores unkalibriert")
            return cast(np.ndarray, np.zeros(expected_dim, dtype=np.float32))
        acc = np.zeros(expected_dim, dtype=np.float64)
        count = 0
        for path in clean_files:
            try:
                _loaded = load_audio_file(str(path))
                if not _loaded or _loaded.get("error"):
                    continue
                y = np.asarray(_loaded.get("audio"), dtype=np.float32)
                sr = int(_loaded.get("sr") or MERT_SR)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != MERT_SR:
                    y = self._resample(y, sr, MERT_SR)
                feat = self._extract_features(y)
                acc += feat.mean(axis=0).astype(np.float64)
                count += 1
            except Exception as exc:  # pylint: disable=broad-except — §V6 (copilot-instructions.md): Datei-übersprung sichtbar
                log.warning("MERT Quality Gate: Clean-Datei %s übersprungen (%s)", path, exc)
        if count == 0:
            log.warning("MERT Quality Gate: keine nutzbare Clean-Datei — Scores unkalibriert")
            return cast(np.ndarray, np.zeros(expected_dim, dtype=np.float32))
        centroid = (acc / count).astype(np.float32)
        try:
            centroid_path = _PROJECT / "models" / "mert" / "clean_music_centroid.npy"
            np.save(centroid_path, centroid)
            log.info(
                "MERT Quality Gate: Clean-Zentroid (%d-dim, %d Dateien) gespeichert: %s",
                expected_dim,
                count,
                centroid_path,
            )
        except OSError as exc:
            log.warning("MERT Quality Gate: Zentroid-Speicherung fehlgeschlagen (%s)", exc)
        return cast(np.ndarray, centroid)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        g = math.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return cast(np.ndarray, (resample_poly(audio, up, down).astype(np.float32)))

    def _extract_features(self, audio_16k: np.ndarray) -> np.ndarray:
        """Extract MERT embeddings for audio chunk. Returns [T, 768]."""
        audio_norm = audio_16k / (np.abs(audio_16k).max() + 1e-10)
        outputs = self.session.run(
            None,
            {"input_values": audio_norm[np.newaxis, :].astype(np.float32)},
        )
        return cast(np.ndarray, outputs[0][0].astype(np.float32))  # [T, 768]

    def score_chunk(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Score a single audio chunk (0-100) based on cosine similarity to clean music centroid.
        Higher = more music-like, lower = noisy/degraded.
        """
        if sample_rate != MERT_SR:
            audio = self._resample(audio, sample_rate, MERT_SR)

        feat = self._extract_features(audio)  # [T, 768]
        feat_mean = feat.mean(axis=0)  # [768]

        if self._clean_centroid is not None and self._clean_centroid.sum() != 0:
            cosine_sim = np.dot(feat_mean, self._clean_centroid) / (
                np.linalg.norm(feat_mean) * np.linalg.norm(self._clean_centroid) + 1e-10
            )
            score = float((cosine_sim + 1) * 50)  # 0-100 from cosine [-1,1]
        else:
            score = 50.0

        return float(np.clip(score, 0.0, 100.0))

    def analyze(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Full analysis returning quality dimensions based on MERT embedding analysis.
        """
        if sample_rate != MERT_SR:
            audio = self._resample(audio, sample_rate, MERT_SR)

        feat = self._extract_features(audio)  # [T, 768]
        feat_mean = feat.mean(axis=0)  # [768]

        # Overall naturalness: cosine similarity to clean music centroid
        if self._clean_centroid is not None and self._clean_centroid.sum() != 0:
            cosine_sim = np.dot(feat_mean, self._clean_centroid) / (
                np.linalg.norm(feat_mean) * np.linalg.norm(self._clean_centroid) + 1e-10
            )
            naturalness = float((cosine_sim + 1) * 50)
        else:
            naturalness = 50.0

        overall = float(np.clip(naturalness, 0.0, 100.0))

        # Frame-level variance: high variance = inconsistent quality
        feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-10)
        centroid_norm = cast(np.ndarray, self._clean_centroid) / (
            np.linalg.norm(cast(np.ndarray, self._clean_centroid)) + 1e-10
        )
        frame_sims = feat_norm @ centroid_norm  # [T]
        frame_scores = (frame_sims + 1) * 50
        consistency = float(100.0 - min(100.0, frame_scores.std() * 3))
        min_frame = float(np.clip(frame_scores.min(), 0, 100))

        # Determine recommended denoising strength
        if overall < QUALITY_HEAVY:
            strength, mode_val = 1.0, "heavy"
        elif overall < QUALITY_MEDIUM:
            strength, mode_val = 0.6, "medium"
        elif overall < QUALITY_LIGHT:
            strength, mode_val = 0.3, "light"
        else:
            strength, mode_val = 0.0, "pass"

        return {
            "overall_score": round(overall, 1),
            "naturalness": round(naturalness, 1),
            "consistency": round(consistency, 1),
            "min_frame_score": round(min_frame, 1),
            "recommended_strength": strength,
            "recommended_mode": mode_val,
            "frame_count": len(feat),
        }

    def warmup(self):
        """Run one inference to warm up GPU/ONNX."""
        if self._warmed:
            return
        dummy = np.random.randn(MERT_SR * 2).astype(np.float32)
        _ = self._extract_features(dummy)
        self._warmed = True
        log.info("MERT Quality Gate: warmup complete")
