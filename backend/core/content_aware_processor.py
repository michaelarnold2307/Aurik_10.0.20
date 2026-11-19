"""
Aurik 10.0.0 — ContentAwareProcessor (§2.36)
========================================
Verfeinert die PerceptualAttentionModel-Salienz-Karte anhand von
Wort-Zeitstempeln aus LyricsTranscriber (§2.36).

Prinzip:
    Betonte Frikative (z. B. /s/ in „süß", „sehnsucht") tragen semantischen
    Inhalt — ihre HF-Energie (Bark-Bänder 17–23, ~4–16 kHz) darf von NR
    kaum angetastet werden.  Unbetonte Silben und Stille-Frames können
    aggressiver behandelt werden.

Salienz-Boost-Faktoren (SALIENCY_BOOST, §2.36):
    fricative_stressed:   2.0   → HF-Bänder 17–23 × 2.0
    fricative_unstressed: 1.4
    vowel_stressed:       1.6
    vowel_unstressed:     1.0
    plosive:              1.5
    silence:              0.5
    mixed:                1.0

G_floor-Override (G_FLOOR_FRICATIVE_STRESSED = 0.90):
    Bei fricative+stressed: saliency_map[k, 17:24] mindestens 0.90,
    bevor NR-Gain angewendet wird.

Invarianten (§2.36, §3.1, §3.2):
    - Salienz-Werte ∈ [0.3, 2.0] (PAM-Invariante §2.22)
    - Kein Absenken unter 0.3 (kein völliges Unterdrücken)
    - NaN/Inf-safe überall
    - Thread-safe: Singleton (Double-Checked Locking §3.2)
    - Laufzeit: ≤ 0.5 s Overhead auf LyricsTranscriber-Dauer
    - Datenschutz: Lyrics-Text niemals geloggt
"""

from __future__ import annotations

import threading

import numpy as np

from backend.core.lyrics_guided_enhancement import (  # canonical (§dedup)
    ContentAwareProcessor,
    LyricsTranscriptionResult,
    WordTimestamp,
)

# Duck-typed Stubs: Der Processor benötigt nur diese Attribute, keine konkrete Plugin-Klasse.

# ---------------------------------------------------------------------------
# Konstanten (§2.36)
# ---------------------------------------------------------------------------

N_BARK_BANDS: int = 24
# Anzahl Bark-Bänder — identisch zu PerceptualAttentionModel (§2.22).

# HF-Bark-Bänder-Index-Bereich: Bänder 17–23 (~4–16 kHz, §2.36)
HF_BARK_START: int = 17
HF_BARK_END: int = 24  # exclusive (= N_BARK_BANDS)

# Clamp-Grenzen (PAM-Invariante §2.22)
_SALIENCY_MIN: float = 0.3
_SALIENCY_MAX: float = 2.0

# G_floor an fricative+stressed HF-Bins (§2.36)
G_FLOOR_FRICATIVE_STRESSED: float = 0.90

# Salienz-Boost-Tabelle (§2.36)
SALIENCY_BOOST: dict[str, float] = {
    "fricative_stressed": 2.0,
    "fricative_unstressed": 1.4,
    "vowel_stressed": 1.6,
    "vowel_unstressed": 1.0,
    "plosive": 1.5,
    "silence": 0.5,
    "mixed": 1.0,
}

# Frame-Konfiguration (identisch zu PAM §2.22: 500 ms, 250 ms Hop)
_FRAME_DURATION_S: float = 0.5
_FRAME_HOP_S: float = 0.25


# ---------------------------------------------------------------------------
# Öffentliche Hilfsfunktionen
# ---------------------------------------------------------------------------


def _resolve_boost_key(phoneme_type: str, is_stressed: bool) -> str:
    """Leitet den SALIENCY_BOOST-Schlüssel aus Phonem-Typ und Betonung ab.

    Args:
        phoneme_type: "vowel" | "fricative" | "plosive" | "silence" | "mixed"
        is_stressed:  True wenn das Wort betont ist

    Returns:
        Schlüssel aus SALIENCY_BOOST; "mixed" als Fallback.
    """
    if phoneme_type == "fricative":
        return "fricative_stressed" if is_stressed else "fricative_unstressed"
    if phoneme_type == "vowel":
        return "vowel_stressed" if is_stressed else "vowel_unstressed"
    if phoneme_type in SALIENCY_BOOST:
        return phoneme_type
    return "mixed"


def _find_word_at(
    words: list,
    t_center_s: float,
) -> object | None:
    """Findet das Wort, das den Zeitpunkt t_center_s enthält.

    Args:
        words:      Liste von WordTimestamp-Objekten (start_s, end_s)
        t_center_s: Frame-Mittelpunkt in Sekunden

    Returns:
        Das erste passende WordTimestamp oder None.
    """
    for w in words:
        if w.start_s <= t_center_s < w.end_s:
            return w  # type: ignore[no-any-return]
    return None


# ---------------------------------------------------------------------------
# ContentAwareProcessor (§2.36)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Singleton + Convenience (§3.2)
# ---------------------------------------------------------------------------

_instance: ContentAwareProcessor | None = None
_lock = threading.Lock()


def get_content_aware_processor() -> ContentAwareProcessor:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ContentAwareProcessor()
    return _instance


def compute_lyrics_saliency(
    base_saliency: np.ndarray,
    transcription: LyricsTranscriptionResult,
    sr: int = 48_000,
) -> np.ndarray:
    """Convenience-Wrapper: verfeinert Salienz-Karte ohne Klassen-Instantiierung."""
    return get_content_aware_processor().compute_lyrics_saliency(base_saliency, transcription, sr)
