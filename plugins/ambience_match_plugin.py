#!/usr/bin/env python3
"""Ambience-Match-Plugin (Spec 25, Hörordnung Ebene 2): additiv, deterministisch.

Additives Raumambiente, das je Bark-Band unter der Maskierungsschwelle des
Signals bleibt (H1). Das Signal wird nie verändert (H2, rein additiv), in
Defekt-Bändern wird das Ambiente zusätzlich abgesenkt (H3), Signal und
Ambiente bleiben getrennt abrufbar (H4). Kein Modell, keine Gewichte, keine
Zeitquelle — gleicher Input + gleiche Version ⇒ bit-identischer Output (§G5 (copilot-instructions.md)).

Zustandslos: alle Zufallszahlen stammen aus per-Aufruf abgeleiteten Seeds
(Song-Isolation §V8 (copilot-instructions.md)/§G1 (copilot-instructions.md) durch Konstruktion). Stärke-Entscheidungen laufen
zentral über `GLOBAL_AMBIENCE_SCALAR` (§V7 (copilot-instructions.md)).

Spec: .github/specs/25_ambience_match_plugin.md

Usage:
    from plugins.ambience_match_plugin import AmbienceMatchPlugin
    plugin = AmbienceMatchPlugin(blend=0.3, safety_margin_db=6.0)
    out = plugin.process(audio, sr)          # Signal + Ambiente (float32)
    amb = plugin.ambience(audio, sr)         # Ambiente allein (H4), float32
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np

from backend.core.dsp.masking_model import (
    _frame_band_energies,
    bark_band_edges,
    compute_masking_threshold_db,
)

logger = logging.getLogger(__name__)

# Zentraler Stärke-Skalar (§V7 (copilot-instructions.md)): wirkt multiplikativ auf die Ambiente-Amplitude.
GLOBAL_AMBIENCE_SCALAR = 1.0

# Session-Saat: deterministisch; pro Song einfach dieselbe Instanz/Konfig nutzen.
DEFAULT_SESSION_SEED = 20260922

_QUIET_QUANTILE = 0.20  # Anteil der leisesten Frames für die Raumprofil-Schätzung
_MAX_PROFILE_DEPTH_DB = 30.0  # Cap der Profil-Dynamik (Band-Kontrast realer Räume)
_MIN_SAFETY_DB = 3.0  # Untergrenze der Sicherheitsmarge (Spec: Standard σ = 6 dB)
_DEFECT_ATTEN_DB = 12.0  # Zusatzdämpfung in Defekt-Bändern (H3)
_CHUNK_SECONDS = 3.0  # Synthese-Segment (Spec: Latenz-Budget je 3-s-Segment)
_ROOM_IR_SECONDS = 0.60  # Länge des internen Raum-IR
_ROOM_IR_DECAY = 0.15  # Abkling-Konstante des internen Raum-IR (s)
_ROOM_IR_LP_ALPHA = 0.08  # Ein-Pol-Tiefpass-Koeffizient des internen Raum-IR


def _mix_seed(seed: int, sr: int, chunk_index: int, ch: int) -> int:
    """Deterministische Seed-Ableitung ohne hash() (PYTHONHASHSEED-neutral)."""
    return (seed * 1000003 + sr * 7919 + chunk_index * 104729 + ch * 1299709) & 0xFFFFFFFF


def _chunk_size(sr: int) -> int:
    return max(4096, int(round(sr * _CHUNK_SECONDS)))


def _bin_band_indices(n_bins: int, sr: int, n_bands: int) -> np.ndarray:
    """Bark-Band-Index je FFT-Bin (0-basiert, geklemmt auf [0, n_bands-1])."""
    edges = bark_band_edges(sr)
    freqs = np.fft.rfftfreq(2 * (n_bins - 1), d=1.0 / sr)
    idx = np.searchsorted(edges, freqs, side="right") - 1
    return cast(np.ndarray, np.clip(idx, 0, n_bands - 1))


def _synthesize_room_ir(sr: int, seed: int) -> np.ndarray:
    """Deterministisches Raum-IR: tiefpassgefiltertes, exponentiell abklingendes
    Rauschen (internes Basis-Set, Spec: optionale IR-Faltung, deterministisch)."""
    n = max(64, int(round(sr * _ROOM_IR_SECONDS)))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    t = np.arange(n, dtype=np.float64) / float(sr)
    filt = np.empty(n, dtype=np.float64)
    acc = 0.0
    for i in range(n):
        acc += _ROOM_IR_LP_ALPHA * (noise[i] - acc)
        filt[i] = acc
    ir = filt * np.exp(-t / _ROOM_IR_DECAY)
    ir = ir - np.mean(ir)  # DC-frei
    rms = np.sqrt(np.mean(ir**2)) + 1e-12
    return cast(np.ndarray, ir / rms)


class AmbienceMatchPlugin:
    """Raum-Ambiente als Politur — Aufruf ausschließlich über backend/api/bridge.py.

    Invarianten (Spec 25):
      H1  Ambiente je Bark-Band ≤ Maskierungsschwelle − safety_margin_db.
      H2  Signal wird nie verändert (rein additiv).
      H3  Defekt-Bänder zusätzlich um 12 dB abgesenkt.
      H4  blend=0 ⇒ bit-identischer Passthrough; Ambiente separat abrufbar.
    """

    def __init__(
        self,
        safety_margin_db: float = 6.0,
        blend: float = 0.0,
        seed: int = DEFAULT_SESSION_SEED,
    ) -> None:
        if safety_margin_db < _MIN_SAFETY_DB:
            raise ValueError(f"safety_margin_db muss >= {_MIN_SAFETY_DB} dB sein (Spec: σ ≥ 3 dB)")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend muss in [0, 1] liegen")
        self._safety_margin_db = float(safety_margin_db)
        self._blend = float(blend)
        self._seed = int(seed)
        self._ir_cache: dict[int, np.ndarray] = {}

    # ── Öffentliche API ────────────────────────────────────────────────────

    def ambience(
        self,
        audio: np.ndarray,
        sr: int,
        defect_bands: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Nur der Ambiente-Anteil (H4), gleiche Form/dtype wie der Input."""
        audio_in = np.asarray(audio)
        if audio_in.dtype != np.float32:
            audio_in = audio_in.astype(np.float32)
        if audio_in.size == 0:
            return cast(np.ndarray, audio_in)
        orig_shape = audio_in.shape
        mono = audio_in.reshape(-1, audio_in.shape[-1])
        n = mono.shape[-1]
        n_chunk = _chunk_size(sr)
        ref_amp = self._probe_ref_amp(sr, n_chunk)
        amb = np.empty_like(mono, dtype=np.float32)
        for ch in range(mono.shape[0]):
            level_db = self._target_levels_db(mono[ch], sr, defect_bands)
            gains = np.clip(10.0 ** (level_db / 20.0) / ref_amp, 1e-6, 1.0)
            amb[ch] = self._ambience_channel(n, sr, gains, ch, n_chunk)
        return cast(np.ndarray, amb.reshape(orig_shape))

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        blend: float | None = None,
        defect_bands: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Signal + blend·Ambiente. blend=0 ⇒ bit-identischer Passthrough (H4)."""
        audio_in = np.asarray(audio)
        if audio_in.dtype != np.float32:
            audio_in = audio_in.astype(np.float32)
        b = self._blend if blend is None else float(blend)
        if b <= 0.0 or audio_in.size == 0:
            return cast(np.ndarray, audio_in)  # unverändert, gleiches Objekt
        amb = self.ambience(audio_in, sr, defect_bands)
        scale = np.float32(b * GLOBAL_AMBIENCE_SCALAR)
        out = (audio_in + scale * amb).astype(np.float32)
        return cast(np.ndarray, out)

    # ── Pegel-/Profil-Schätzung ────────────────────────────────────────────

    def _target_levels_db(
        self,
        audio_1d: np.ndarray,
        sr: int,
        defect_bands: Sequence[int] | None,
    ) -> np.ndarray:
        """Zielpegel je Bark-Band (dB, Amplitude): Maskierungsschwelle − σ,
        geformt durch das Raumprofil der leisesten Segmente (H1 bleibt
        garantiert, da das Profil nur absenkt)."""
        x = np.asarray(audio_1d, dtype=np.float32)
        thr, _z = compute_masking_threshold_db(x, sr)
        thr_mean = np.mean(thr, axis=0)

        e_db, _z = _frame_band_energies(x, sr)
        frame_sum = np.mean(e_db, axis=1)
        quiet = frame_sum <= np.quantile(frame_sum, _QUIET_QUANTILE)
        if not np.any(quiet):
            quiet = np.ones(len(frame_sum), dtype=bool)  # Fallback: alle Frames
        prof = np.mean(e_db[quiet], axis=0)
        # Profil-Dynamik kappen: reale Räume haben begrenzten Band-Kontrast;
        # extreme Täler würden nur die Fenster-Leckage der Messung (Hann,
        # 2048-FFT) dominieren lassen, ohne hörbaren Nutzen.
        prof_rel = np.maximum(prof - np.max(prof), -_MAX_PROFILE_DEPTH_DB)  # ≤ 0 dB

        level = thr_mean - self._safety_margin_db + np.minimum(prof_rel, 0.0)
        if defect_bands is not None:
            idx = np.asarray(defect_bands, dtype=int)
            idx = idx[(idx >= 0) & (idx < len(level))]
            if idx.size:
                level = level.copy()
                level[idx] -= _DEFECT_ATTEN_DB  # H3
        return cast(np.ndarray, level)

    # ── Synthese ───────────────────────────────────────────────────────────

    def _room_ir(self, sr: int) -> np.ndarray:
        if sr not in self._ir_cache:
            self._ir_cache[sr] = _synthesize_room_ir(sr, _mix_seed(self._seed, sr, -2, 0))
        return self._ir_cache[sr]

    def _probe_ref_amp(self, sr: int, n_chunk: int) -> np.ndarray:
        """Kalibrier-Referenz: Band-Amplitude der Kette (Formung + IR) bei
        gain=1, gemessen im stationären Bereich. Zwei Probe-Chunks werden
        synthetisiert und der zweite Chunk gemessen — er enthält den
        Ring-In des ersten Chunks (Overlap-Add) und vermeidet zugleich den
        leisen Ring-Out des Ketten-Endes. Die Kette ist linear, daher
        skaliert der Zielpegel exakt mit dem Gain je Band."""
        n_bands = len(bark_band_edges(sr)) - 1
        ones = np.ones(n_bands)
        n_probe = 2 * n_chunk
        amb = self._ambience_channel(n_probe, sr, ones, ch=0, n_chunk=n_chunk)
        mid = amb[n_chunk : n_chunk + n_chunk]
        e_db, _z = _frame_band_energies(mid.astype(np.float32), sr)
        e_ref_mean = np.mean(e_db, axis=0)
        return cast(np.ndarray, np.sqrt(10.0 ** (e_ref_mean / 10.0) + 1e-30))

    def _ambience_channel(
        self,
        n: int,
        sr: int,
        gains: np.ndarray,
        ch: int,
        n_chunk: int,
    ) -> np.ndarray:
        """Chunkweise Rausch-Synthese (3-s-Segmente) mit Bark-Formung und
        Overlap-Add-Faltung mit dem internen Raum-IR. Deterministisch je
        (seed, sr, chunk_index, ch)."""
        out = np.zeros(n, dtype=np.float64)
        ir = self._room_ir(sr)
        n_ir = len(ir)
        n_bands = len(gains)
        tail = np.zeros(max(n_ir - 1, 0), dtype=np.float64)
        pos = 0
        chunk_index = 0
        while pos < n:
            m = min(n_chunk, n - pos)
            rng = np.random.default_rng(_mix_seed(self._seed, sr, chunk_index, ch))
            noise = rng.standard_normal(m)
            spec = np.fft.rfft(noise)
            spec *= gains[_bin_band_indices(len(spec), sr, n_bands)]
            shaped = np.fft.irfft(spec, n=m)
            n_conv = m + n_ir - 1
            conv = np.fft.irfft(np.fft.rfft(shaped, n_conv) * np.fft.rfft(ir, n_conv), n_conv)
            add = np.zeros(m, dtype=np.float64)
            k = min(m, len(tail))
            add[:k] += tail[:k]
            seg = conv[:m] + add
            out[pos : pos + m] = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
            new_tail = conv[m : m + n_ir - 1].copy()
            if m < n_ir - 1:
                new_tail[: n_ir - 1 - m] += tail[m:]
            tail = new_tail
            pos += m
            chunk_index += 1
        return cast(np.ndarray, out.astype(np.float32))
