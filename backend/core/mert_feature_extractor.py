"""
§v10.126: MERT Feature Extractor — MERT-v1-330M (ONNX, provider policy).

MERT (Music undERstanding Transformer) ist auf 160k+ Stunden Musik vortrainiert.
Produziert 1024-dim Features bei 24 kHz, die musikalische Struktur codieren:
  - Genre, Instrumentierung, Harmonik, Rhythmus

Nutzung:
  extractor = MERTFeatureExtractor()
  features = extractor.extract(audio, sample_rate)
  # features: [frames, 1024] — eine Feature-Matrix pro ~10ms
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import cast

import numpy as np
from scipy.signal import resample_poly

from backend.core.gpu_model_registry import get_onnx_providers

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "mert" / "mert.onnx"
MERT_SAMPLE_RATE = 24000


class MERTFeatureExtractor:
    """MERT-v1-330M-Extraktor mit zentraler ONNX-Provider-Policy."""

    def __init__(self):
        import onnxruntime as ort

        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"MERT model not found: {_MODEL_PATH}")

        self._session = ort.InferenceSession(
            str(_MODEL_PATH),
            providers=get_onnx_providers(_MODEL_PATH.resolve()),
        )
        self._provider = self._session.get_providers()[0]
        logger.info("MERT geladen: %s (330M params, %s)", _MODEL_PATH.name, self._provider)

    def extract(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Extrahiert MERT-Features aus Audio.

        Args:
            audio: float32 [samples]
            sample_rate: Eingangs-Sample-Rate; wird auf die MERT-Rate 24 kHz resampelt.

        Returns:
            np.ndarray [frames, 1024] — MERT-v1-330M-Musik-Features
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate muss positiv sein")

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 2:
            # Pipeline-Standard ist channels-first; samples-first wird ebenfalls
            # akzeptiert, damit Modulgrenzen kein Stereo zu zwei Samples falten.
            if audio.shape[0] <= 8 and audio.shape[0] <= audio.shape[1]:
                audio = audio.mean(axis=0)
            elif audio.shape[1] <= 8 and audio.shape[1] < audio.shape[0]:
                audio = audio.mean(axis=1)
            else:
                raise ValueError(f"Mehrdeutiges Stereo-Layout für MERT: {audio.shape}")
        elif audio.ndim != 1:
            raise ValueError(f"MERT erwartet Mono- oder Stereo-Audio, erhalten: {audio.shape}")

        if audio.size == 0:
            raise ValueError("MERT erwartet nichtleeres Audio")

        audio = np.nan_to_num(audio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if sample_rate != MERT_SAMPLE_RATE:
            divisor = math.gcd(sample_rate, MERT_SAMPLE_RATE)
            audio = cast(
                np.ndarray,
                resample_poly(
                    audio,
                    MERT_SAMPLE_RATE // divisor,
                    sample_rate // divisor,
                ).astype(np.float32),
            )

        # Normalize
        peak = np.abs(audio).max() + 1e-10
        audio = audio / peak

        # Add batch dimension
        audio_batch = audio[np.newaxis, :]

        # Inferenz
        outputs = self._session.run(None, {"input_values": audio_batch})
        features = outputs[0]  # [1, frames, 1024] für MERT-v1-330M
        return cast(np.ndarray, features[0])  # [frames, 1024]

    def extract_mean(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """Extrahiert gemittelte MERT-Features (ein Vektor pro Audiodatei).

        Nützlich für Genre-Erkennung oder globale Audio-Klassifikation.
        """
        features = self.extract(audio, sample_rate)
        return cast(np.ndarray, features.mean(axis=0).astype(np.float32))  # [1024]

    def extract_segments(self, audio: np.ndarray, sample_rate: int = 48000, segment_s: float = 5.0) -> np.ndarray:
        """Extrahiert MERT-Features in Segmenten (für lange Audiodateien).

        Args:
            audio: float32 [samples]
            sample_rate: Sample-Rate
            segment_s: Segment-Länge in Sekunden

        Returns:
            np.ndarray [n_segments, 1024]
        """
        segment_samples = int(segment_s * sample_rate)
        n_segments = (len(audio) + segment_samples - 1) // segment_samples
        features = []

        for i in range(n_segments):
            start = i * segment_samples
            end = min(start + segment_samples, len(audio))
            segment = audio[start:end]
            feat = self.extract_mean(segment, sample_rate)
            features.append(feat)

        return cast(np.ndarray, (np.stack(features, axis=0)))


# ── Context-Analyse ────────────────────────────────────────────────────────


def compute_music_context(audio: np.ndarray, sample_rate: int = 48000) -> dict:
    """Hochrangige Musik-Kontext-Analyse via MERT.

    Returns dict mit:
      - genre_proxy: float [0,1] — "how acoustic" (0=elektronisch, 1=akustisch)
      - density: float [0,1] — spektrale Dichte (0=sparse, 1=dense)
      - brightness: float [0,1] — Helligkeit (0=dunkel, 1=hell)
      - is_vocal: float [0,1] — Vocal-Präsenz
    """
    try:
        extractor = MERTFeatureExtractor()
        features = extractor.extract(audio, sample_rate)
        mean_feat = features.mean(axis=0)

        # Heuristische Mappings (vereinfacht, aber nützlich)
        # MERT-Dimensionen korrelieren mit musikalischen Eigenschaften
        low_dim = mean_feat[:256]  # Untere Dimensionen → Rhythmus, Bass
        mid_dim = mean_feat[256:512]  # Mittlere → Harmonik, Instrumentierung
        high_dim = mean_feat[512:]  # Obere → Textur, Vocals

        return {
            # Rhythmische Dichte (Varianz in unteren Dimensionen)
            "density": float(np.clip(np.std(low_dim) * 3.0, 0.0, 1.0)),
            # Harmonische Helligkeit (Mittelwert der mittleren Dimensionen)
            "brightness": float(np.clip((np.mean(mid_dim) + 0.5), 0.0, 1.0)),
            # Vocal-Präsenz (hohe Dimensionen korrelieren mit Stimme)
            "is_vocal": float(np.clip(np.mean(np.abs(high_dim)) * 2.0, 0.0, 1.0)),
            # Akustik-Proxy (niedrige Varianz = akustisch, hohe = elektronisch)
            "genre_proxy": float(1.0 - np.clip(np.std(high_dim) * 2.0, 0.0, 1.0)),
        }
    except Exception as e:
        logger.debug("MERT-Kontext-Analyse fehlgeschlagen: %s", e)
        return {"density": 0.5, "brightness": 0.5, "is_vocal": 0.5, "genre_proxy": 0.5}
