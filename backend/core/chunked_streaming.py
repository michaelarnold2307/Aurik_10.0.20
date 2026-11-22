"""backend/core/chunked_streaming.py — §v10.350 Chunked-Streaming Architektur.

Grundstein für O(duration) → O(1) RAM-Transformation.
Verarbeitet lange Audio-Dateien in überlappenden Chunks statt als Ganzes.

Status: ✅ Produktion (§v10.450 UnifiedRestorerV3._restore_chunked)

Design:
- Audio wird in Chunks von chunk_duration_s (default 30s) zerlegt
- Jeder Chunk durchläuft die komplette Pipeline
- Crossfade-Overlap (default 2s) für nahtlose Übergänge
- Nur 2 Chunks gleichzeitig im RAM (aktueller + nächster)
- RAM-Bedarf: O(chunk_size) statt O(duration)
  - 224s Audio: von ~28 GB auf ~4 GB

Usage::

    from backend.core.chunked_streaming import ChunkedPipeline

    cp = ChunkedPipeline(restorer, chunk_duration_s=30.0, overlap_s=2.0)
    result = cp.process(audio, sample_rate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

# §v10.350: Default-Parameter aus Benchmarking (224s Kassette, 42 Phasen).
# §P3-1 (2026-09-27): Chunk-Dauer von 30s auf 120s erhöht — weniger Crossfade-Nähte
# = weniger Artefakte + bessere Phasenkohärenz über längere Abschnitte. Per-Chunk-Fixkosten
# (LGE-Saliency, carrier_chain, FCPE-Reload) werden von 8× auf 2× reduziert.
DEFAULT_CHUNK_DURATION_S: float = 120.0
DEFAULT_OVERLAP_S: float = 2.0
DEFAULT_CROSSFADE_S: float = 0.05  # 50ms Crossfade

# §PERF-R16: Grenzen für den AURIK_CHUNK_S-Override (Supervised-Evidenz).
CHUNK_OVERRIDE_MIN_S: float = 10.0
CHUNK_OVERRIDE_MAX_S: float = 600.0


def select_chunk_duration_s(total_s: float, env_override: str = "") -> tuple[float, bool]:
    """Chunk-Dauer aus Heuristik + AURIK_CHUNK_S-Override (§PERF-R16).

    Deterministisch (§G5 (GEBOTE.md)): gleiche Eingaben => gleiche Ausgabe.
    Default unverändert (30s <= 300s Songs, sonst 60s) => bestehende Läufe
    bleiben bit-identisch. Der Override (10..600s) dient Supervised-Evidenz-
    läufen für die Chunk-Vergrößerung; ungültige Werte fallen auf die
    Heuristik zurück und melden applied=False.

    Args:
        total_s:     Song-Länge in Sekunden.
        env_override: Wert von AURIK_CHUNK_S (roh, ggf. leer).

    Returns:
        (chunk_duration_s, applied) — applied=True wenn der Override griff.
    """
    base = 120.0 if total_s > 300.0 else 60.0
    override = env_override.strip()
    if not override:
        return base, False
    try:
        override_f = float(override)
    except ValueError:
        logger.warning(
            "chunked_streaming: Override %r nicht numerisch -> Basis-Chunking %.2fs (§V6, copilot-instructions.md)",
            override,
            base,
        )
        return base, False
    if CHUNK_OVERRIDE_MIN_S <= override_f <= CHUNK_OVERRIDE_MAX_S:
        return override_f, True
    return base, False


@dataclass
class ChunkConfig:
    """Konfiguration für Chunked-Streaming."""

    chunk_duration_s: float = DEFAULT_CHUNK_DURATION_S
    overlap_s: float = DEFAULT_OVERLAP_S
    crossfade_s: float = DEFAULT_CROSSFADE_S
    min_chunk_duration_s: float = 10.0  # Letzter Chunk mindestens so lang


@dataclass
class ChunkResult:
    """Ergebnis eines einzelnen Chunk-Durchlaufs."""

    audio: np.ndarray
    sample_rate: int
    chunk_index: int
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class ChunkDeltaPatch:
    """Sparse phase-off delta for one chunk (sample-first, Spec 26 DRT-13)."""

    chunk_index: int
    sample_indices: np.ndarray
    delta: np.ndarray


class ChunkedPipeline:
    """Chunked-Streaming-Wrapper für die Restaurierungs-Pipeline.

    Zerlegt langes Audio in Chunks, verarbeitet jeden durch die Pipeline,
    und setzt sie mit Crossfade-Overlap wieder zusammen.

    Thread-sicher: verarbeitet Chunks sequentiell (kein Parallelismus).
    """

    def __init__(
        self,
        chunk_duration_s: float = DEFAULT_CHUNK_DURATION_S,
        overlap_s: float = DEFAULT_OVERLAP_S,
        crossfade_s: float = DEFAULT_CROSSFADE_S,
    ) -> None:
        self.config = ChunkConfig(
            chunk_duration_s=chunk_duration_s,
            overlap_s=overlap_s,
            crossfade_s=crossfade_s,
        )
        logger.info(
            "ChunkedPipeline: chunk=%.1fs overlap=%.1fs crossfade=%.0fms",
            chunk_duration_s,
            overlap_s,
            crossfade_s * 1000,
        )

    def compute_chunks(self, audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
        """Berechnet Chunk-Grenzen (start_sample, end_sample) für Audio.

        Returns:
            Liste von (start_sample, end_sample) Tupeln.
        """
        # §v10.451: audio.shape[0] für Sample-Zahl (nicht shape[-1]=Kanäle)
        n_samples = audio.shape[0]
        chunk_samples = int(self.config.chunk_duration_s * sample_rate)
        overlap_samples = int(self.config.overlap_s * sample_rate)
        # §v10.459: Guard gegen negative step (overlap > chunk_size)
        if overlap_samples >= chunk_samples:
            overlap_samples = max(1, chunk_samples // 4)
            logger.warning(
                "ChunkedPipeline: overlap (%.1fs) >= chunk (%.1fs) → clamped to %.1fs",
                self.config.overlap_s,
                self.config.chunk_duration_s,
                overlap_samples / sample_rate,
            )
        step = chunk_samples - overlap_samples

        chunks: list[tuple[int, int]] = []
        start = 0
        while start < n_samples:
            end = min(start + chunk_samples, n_samples)
            # Letzter Chunk: mindestens min_chunk_duration_s
            remaining = n_samples - start
            if remaining < chunk_samples and remaining < int(self.config.min_chunk_duration_s * sample_rate):
                # Zu kurz für eigenen Chunk → an vorherigen anhängen
                if chunks:
                    chunks[-1] = (chunks[-1][0], n_samples)
                else:
                    chunks.append((0, n_samples))
                break
            chunks.append((start, end))
            start += step

        logger.info("ChunkedPipeline: %d Chunks für %.1fs Audio", len(chunks), n_samples / sample_rate)
        return chunks

    def crossfade(self, chunk_a: np.ndarray, chunk_b: np.ndarray, overlap_samples: int) -> np.ndarray:
        """Hann-Fenster-Crossfade zwischen zwei Chunks.

        §v10.401: Hann-Fenster statt linear — null Ableitung an den
        Endpunkten verhindert Klick-Artefakte an Chunk-Grenzen.

        Args:
            chunk_a: Ende des vorherigen Chunks.
            chunk_b: Anfang des nächsten Chunks.
            overlap_samples: Anzahl Samples im Überlappungsbereich.

        Returns:
            np.ndarray mit Crossfade-Ergebnis (nur Überlappungsbereich).
        """
        if overlap_samples <= 0:
            return chunk_b

        # Hann-Fenster: fade-out + fade-in = 1.0 (perfekte Rekonstruktion)
        t = np.linspace(0, np.pi, overlap_samples, dtype=np.float32)
        fade_out = np.cos(t * 0.5) ** 2  # cos² fade-out
        fade_in = np.sin(t * 0.5) ** 2  # sin² fade-in
        # §v10.451: Stereo: (n,1) broadcast mit (n,ch) → (n,ch)
        if chunk_a.ndim == 2:
            fade_out = fade_out[:, np.newaxis]
            fade_in = fade_in[:, np.newaxis]
            a_tail = chunk_a[-overlap_samples:, :]
            b_head = chunk_b[:overlap_samples, :]
        else:
            a_tail = chunk_a[-overlap_samples:]
            b_head = chunk_b[:overlap_samples]

        return a_tail * fade_out + b_head * fade_in  # type: ignore[no-any-return]

    @staticmethod
    def _rms_match_ratio(
        previous_audio: np.ndarray,
        current_audio: np.ndarray,
        match_samples: int,
        channels_first: bool,
    ) -> float:
        if match_samples <= 0:
            return 1.0
        previous = previous_audio.T if channels_first and previous_audio.ndim == 2 else previous_audio
        current = np.asarray(current_audio, dtype=np.float64)
        if channels_first and current.ndim == 2:
            current = current.T
        previous_tail = previous[-min(match_samples, previous.shape[0]) :]
        current_head = current[: min(match_samples, current.shape[0])]
        rms_previous = float(np.sqrt(np.mean(np.square(previous_tail)) + 1e-12))
        rms_current = float(np.sqrt(np.mean(np.square(current_head)) + 1e-12))
        if rms_previous <= 1e-10 or rms_current <= 1e-10:
            return 1.0
        return float(np.clip(rms_previous / rms_current, 0.25, 4.0))

    def collect_results(self, results: list[ChunkResult], sample_rate: int) -> np.ndarray:
        """Setzt Chunk-Ergebnisse mit Crossfade + RMS-Matching zusammen.

        §v10.712: RMS-Matching an Chunk-Grenzen vor Crossfade.
        Jeder Folge-Chunk wird auf den RMS-Pegel des vorherigen Chunks
        normalisiert (letzte 500ms des vorherigen vs erste 500ms des aktuellen).
        Verhindert die massiven Pegelsprünge (8–17 dB) die durch per-Chunk
        OneTakeExport und variable Post-Processing entstehen.

        Returns:
            np.ndarray — vollständiges restauriertes Audio.
        """
        if not results:
            raise ValueError("Keine Chunk-Ergebnisse")

        total_samples = results[-1].end_sample
        # §v10.451: Stereo: (total_samples, kanäle), nicht (chunk_samples, total_samples)
        # §v10.740 (2026-09-09): Kanal-Layout normalisieren — die Pipeline liefert
        # teils (2, N) channels-first; shape[1] ist dann N (Chunklänge), nicht die
        # Kanalzahl (Befund Lauf 4: 56.7-TiB-Allokation (total, chunk_len)).
        _first_audio = results[0].audio
        _channels = 1
        _is_channels_first = False
        if _first_audio.ndim == 2:
            if _first_audio.shape[0] == 2 and _first_audio.shape[1] > 2:
                _channels = 2
                _is_channels_first = True
            else:
                _channels = int(_first_audio.shape[1])
        shape = (total_samples,) if _channels == 1 else (total_samples, _channels)
        output = np.zeros(shape, dtype=np.float32)

        overlap_samples = int(self.config.overlap_s * sample_rate)
        # §v10.712: RMS-Matching-Fenster — letzte 500ms für Pegelvergleich
        match_samples = int(0.5 * sample_rate)

        for i, cr in enumerate(results):
            chunk_audio = cr.audio.astype(np.float64)
            if _is_channels_first and chunk_audio.ndim == 2:
                chunk_audio = chunk_audio.T  # §v10.740: (2, N) → (N, 2)
            out_start = cr.start_sample
            out_end = cr.end_sample

            if i > 0 and match_samples > 0:
                # §v10.712: RMS-Match vor Crossfade
                _prev_chunk = results[i - 1]
                _rms_ratio = self._rms_match_ratio(
                    _prev_chunk.audio,
                    cr.audio,
                    match_samples,
                    _is_channels_first,
                )
                if abs(_rms_ratio - 1.0) > 0.02:  # > 0.17 dB difference
                    chunk_audio = chunk_audio * _rms_ratio
                    _rms_db = 20.0 * np.log10(_rms_ratio)
                    logger.info(
                        "§v10.712 RMS-Match Chunk %d→%d: Verhaeltnis=%.3f (%.1f dB)",
                        i - 1,
                        i,
                        _rms_ratio,
                        _rms_db,
                    )

            chunk_audio = chunk_audio.astype(np.float32)

            if i == 0:
                # Erster Chunk: direkt kopieren
                chunk_len = out_end - out_start
                if output.ndim == 1:
                    output[out_start:out_end] = chunk_audio[:chunk_len]
                else:
                    output[out_start:out_end, :] = chunk_audio[:chunk_len, :]
            else:
                # Crossfade mit vorherigem Chunk
                crossfade_len = min(overlap_samples, out_end - out_start)
                if crossfade_len > 0:
                    if output.ndim == 1:
                        prev_tail = output[out_start : out_start + crossfade_len]
                        curr_head = chunk_audio[:crossfade_len]
                    else:
                        prev_tail = output[out_start : out_start + crossfade_len, :]
                        curr_head = chunk_audio[:crossfade_len, :]
                    faded = self.crossfade(prev_tail, curr_head, crossfade_len)
                    if output.ndim == 1:
                        output[out_start : out_start + crossfade_len] = faded
                    else:
                        output[out_start : out_start + crossfade_len, :] = faded

                # Rest kopieren
                remaining = out_end - out_start - crossfade_len
                if remaining > 0:
                    if output.ndim == 1:
                        output[out_start + crossfade_len : out_end] = chunk_audio[crossfade_len:]
                    else:
                        output[out_start + crossfade_len : out_end, :] = chunk_audio[crossfade_len:, :]

        return cast(np.ndarray, output)

    def apply_chunk_deltas(
        self,
        assembled_audio: np.ndarray,
        results: list[ChunkResult],
        patches: list[ChunkDeltaPatch],
        sample_rate: int,
    ) -> np.ndarray:
        """Wendet DRT-13-Deltas mit den Produktions-Gains und Crossfades an."""
        if not results:
            raise ValueError("Keine Chunk-Ergebnisse für DRT-13")
        base = np.asarray(assembled_audio, dtype=np.float32)
        if not np.isfinite(base).all():
            raise ValueError("DRT-13 Assembly enthält NaN/Inf")
        first_audio = results[0].audio
        channels_first = bool(first_audio.ndim == 2 and first_audio.shape[0] == 2 and first_audio.shape[1] > 2)
        channels = 2 if channels_first else (int(first_audio.shape[1]) if first_audio.ndim == 2 else 1)
        expected_shape = (results[-1].end_sample,) if channels == 1 else (results[-1].end_sample, channels)
        if base.shape != expected_shape:
            raise ValueError(f"DRT-13 Assembly-Shape {base.shape} != {expected_shape}")

        patches_by_chunk: dict[int, ChunkDeltaPatch] = {}
        for patch in patches:
            if patch.chunk_index in patches_by_chunk:
                raise ValueError(f"Mehrere DRT-13-Patches für Chunk {patch.chunk_index}")
            if patch.chunk_index < 0 or patch.chunk_index >= len(results):
                raise ValueError(f"Ungültiger DRT-13-Chunk-Index {patch.chunk_index}")
            indices = np.asarray(patch.sample_indices)
            delta = np.asarray(patch.delta)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("DRT-13-Sample-Indizes müssen ein eindimensionales Integer-Array sein")
            if delta.ndim not in (1, 2) or delta.shape[0] != indices.size:
                raise ValueError("DRT-13-Delta-Länge stimmt nicht mit Sample-Indizes überein")
            if not np.isfinite(delta).all():
                raise ValueError("DRT-13-Delta enthält NaN/Inf")
            if indices.size and (
                int(indices.min()) < 0
                or int(indices.max()) >= results[patch.chunk_index].audio.shape[-1 if channels_first else 0]
            ):
                raise ValueError(f"DRT-13-Sample-Index außerhalb Chunk {patch.chunk_index}")
            if np.unique(indices).size != indices.size:
                raise ValueError(f"Doppelte DRT-13-Sample-Indizes in Chunk {patch.chunk_index}")
            expected_delta_shape = (indices.size,) if channels == 1 else (indices.size, channels)
            if delta.shape != expected_delta_shape:
                raise ValueError(f"DRT-13-Delta-Shape {delta.shape} != {expected_delta_shape}")
            patches_by_chunk[patch.chunk_index] = patch

        delta_output = np.zeros_like(base)
        overlap_samples = int(self.config.overlap_s * sample_rate)
        match_samples = int(0.5 * sample_rate)

        for i, chunk_result in enumerate(results):
            chunk_len = chunk_result.end_sample - chunk_result.start_sample
            chunk_delta_shape = (chunk_len,) if channels == 1 else (chunk_len, channels)
            chunk_delta = np.zeros(chunk_delta_shape, dtype=np.float32)
            chunk_patch = patches_by_chunk.get(chunk_result.chunk_index)
            if chunk_patch is not None and np.asarray(chunk_patch.sample_indices).size:
                chunk_delta[np.asarray(chunk_patch.sample_indices)] = np.asarray(chunk_patch.delta, dtype=np.float32)

            if i > 0:
                ratio = self._rms_match_ratio(
                    results[i - 1].audio,
                    chunk_result.audio,
                    match_samples,
                    channels_first,
                )
                if abs(ratio - 1.0) > 0.02:
                    chunk_delta *= ratio

            start = chunk_result.start_sample
            end = chunk_result.end_sample
            if i == 0:
                delta_output[start:end] = chunk_delta[: end - start]
                continue

            crossfade_len = min(overlap_samples, end - start)
            if crossfade_len > 0:
                delta_output[start : start + crossfade_len] = self.crossfade(
                    delta_output[start : start + crossfade_len],
                    chunk_delta[:crossfade_len],
                    crossfade_len,
                )
            remaining = end - start - crossfade_len
            if remaining > 0:
                delta_output[start + crossfade_len : end] = chunk_delta[crossfade_len : crossfade_len + remaining]

        candidate = base.copy()
        changed = delta_output != 0
        candidate[changed] = base[changed] + delta_output[changed]
        return cast(np.ndarray, candidate)

    @staticmethod
    def verify_crossfade(results: list, sample_rate: int) -> list[str]:
        """§v10.440: Prüft Crossfade-Qualität an Chunk-Grenzen.

        Misst RMS-Sprung zwischen letzten 100ms von Chunk N und
        ersten 100ms von Chunk N+1. Loggt Warnings bei >0.5dB.

        Returns:
            Liste von Warnungen (leer = perfekte Übergänge).
        """
        warnings: list[str] = []
        check_ms = 0.100  # 100ms Fenster
        check_samples = int(check_ms * sample_rate)
        for i in range(1, len(results)):
            _a = results[i - 1]
            _b = results[i]
            if not hasattr(_a, "audio") or not hasattr(_b, "audio"):
                continue
            _a_end = _a.audio[-check_samples:, :] if _a.audio.ndim >= 2 else _a.audio[-check_samples:]
            _b_start = _b.audio[:check_samples, :] if _b.audio.ndim >= 2 else _b.audio[:check_samples]
            _rms_a = float(np.sqrt(np.mean(np.square(_a_end)) + 1e-12))
            _rms_b = float(np.sqrt(np.mean(np.square(_b_start)) + 1e-12))
            _rms_db = 20.0 * np.log10(max(_rms_a, _rms_b) / max(min(_rms_a, _rms_b), 1e-12))
            if _rms_db > 0.5:
                _w = f"Crossfade-Warnung Chunk {i - 1}→{i}: RMS-Sprung {_rms_db:.1f} dB"
                logger.warning(_w)
                warnings.append(_w)
        if not warnings:
            logger.debug(f"Crossfade-Qualität: {len(results) - 1} Übergänge OK")
        return warnings
