"""
backend/core/live_ab_preview.py — Live A/B-Vorschau (§v10.9)
=============================================================

Ermöglicht Vorher/Nachher-Vergleich für jede Phase im GUI.
Nutzt den bestehenden SharedAudioRing für Live-Waveform.

Usage:
    from backend.core.live_ab_preview import LiveABRing
    ring = LiveABRing(max_frames=3 * 48000)  # 3s @ 48kHz
    ring.write_phase_audio(phase_id, audio_pre, audio_post, sr)
    # GUI pollt via ring.get_pair(phase_id) → (pre, post)
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)

_MAX_PHASE_PAIRS: int = 8  # Maximal 8 Phasen-Paare im Ring


class LiveABRing:
    """Ring-Puffer für Phase Pre/Post-Audio-Snapshots.

    Jeder Eintrag: (phase_id, audio_pre, audio_post, sample_rate).
    Maximal _MAX_PHASE_PAIRS Einträge — älteste werden verdrängt.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pairs: OrderedDict[str, tuple[np.ndarray, np.ndarray, int]] = OrderedDict()

    def write_phase_audio(
        self,
        phase_id: str,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Schreibt Pre/Post-Audio für eine Phase in den Ring.

        Args:
            phase_id: Eindeutige Phase-ID (z.B. 'phase_03_denoise').
            audio_pre: Audio VOR der Phase.
            audio_post: Audio NACH der Phase.
            sample_rate: Sample-Rate.
        """
        # Nur ersten 3 Sekunden speichern (reicht für A/B-Vergleich)
        _max_s = int(3 * sample_rate)
        _pre = np.asarray(audio_pre[..., :_max_s], dtype=np.float32)
        _post = np.asarray(audio_post[..., :_max_s], dtype=np.float32)

        with self._lock:
            self._pairs[phase_id] = (_pre, _post, sample_rate)
            # Alte Einträge verdrängen
            while len(self._pairs) > _MAX_PHASE_PAIRS:
                self._pairs.popitem(last=False)

    def get_pair(self, phase_id: str) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Gibt Pre/Post-Audio für eine Phase zurück oder None."""
        with self._lock:
            entry = self._pairs.get(phase_id)
            if entry is not None:
                return entry
        return None

    @property
    def available_phases(self) -> list[str]:
        """Liste aller Phasen mit gespeicherten Paaren."""
        with self._lock:
            return list(self._pairs.keys())


# Singleton für die GUI
_ab_ring: LiveABRing | None = None
_ab_lock = threading.Lock()


def get_ab_ring() -> LiveABRing:
    """Gibt die Singleton-Instanz des LiveABRing zurück."""
    global _ab_ring
    if _ab_ring is None:
        with _ab_lock:
            if _ab_ring is None:
                _ab_ring = LiveABRing()
    return _ab_ring
