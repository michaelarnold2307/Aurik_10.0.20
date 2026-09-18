"""§v10.303.52 — Sprachadaptive Phonem-Schwellen: Unit-Tests.

Die Klassifikation (_classify_phoneme_type) nutzt sprachabhängige
Detektions-Schwellen (Sibilanten-/Plosiv-Inventar). Geprüft: Defaults für
unbekannte Sprachen (keine Regression), messbarer Klassifikations-Unterschied
zwischen Profilen (HF-Dominanz + Plosiv-Kappe), Tabellen-Konsistenz.
"""

from __future__ import annotations

import numpy as np
import pytest

from plugins.lyrics_transcriber_plugin import LyricsTranscriber


def _classifier() -> LyricsTranscriber:
    # Nur die Klassifikation nutzen — kein Modell-Load im Unit-Test.
    return object.__new__(LyricsTranscriber)  # type: ignore[return-value]


def _band_noise(center_hz: float, bw_hz: float, sr: int = 16000, n: int = 4000, seed: int = 0) -> np.ndarray:
    from scipy.signal import butter, lfilter

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    lo = max(100.0, center_hz - bw_hz / 2.0)
    hi = min(center_hz + bw_hz / 2.0, 0.95 * sr / 2.0)
    b, a = butter(4, [lo / (sr / 2.0), hi / (sr / 2.0)], btype="band")
    y = lfilter(b, a, x)
    return y[2000:].astype(np.float32)  # Einschwingen verwerfen (Steady-State)


def test_profile_defaults_for_unknown_language() -> None:
    """en/fr/unbekannt → identische Klassifikation (keine Regression)."""
    t = _classifier()
    seg = _band_noise(6000.0, 4000.0)
    base = t._classify_phoneme_type(seg, 16000, "")
    for lang in ("en", "fr", "unknown", "ja"):
        assert t._classify_phoneme_type(seg, 16000, lang) == base


def test_language_profile_changes_classification() -> None:
    """HF-Dominanz-Verhältnis 0.9 (de) vs 1.3 (es) entscheidet messbar anders."""
    t = _classifier()
    sr = 16000
    lo = _band_noise(3200.0, 2400.0)  # LF-Band (0–4 kHz)
    hi = _band_noise(5600.0, 2400.0)  # HF-Band (4–16 kHz)

    def _ratio(seg: np.ndarray) -> float:
        n_fft = min(len(seg), 1024)
        mag2 = np.abs(np.fft.rfft(seg[:n_fft])) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        e_hi = float(np.sum(mag2[(freqs >= 4000.0) & (freqs < 16000.0)])) + 1e-12
        e_lo = float(np.sum(mag2[freqs < 4000.0])) + 1e-12
        return e_hi / e_lo

    # Amplituden iterativ so kalibrieren, dass energy_hi/energy_lo ≈ 1.15
    # (Bänder überlappen an der 4-kHz-Grenze → Konvergenz-Schleife).
    _k = 1.0
    for _ in range(6):
        seg = (hi * _k + lo).astype(np.float32)
        _r = _ratio(seg)
        _k *= float(np.sqrt(1.15 / max(_r, 1e-9)))
    ratio = _ratio(seg)
    assert 0.9 < ratio < 1.3, f"Test-Signal außerhalb des sensiblen Bereichs (ratio={ratio:.2f})"
    p_de = t._classify_phoneme_type(seg, sr, "de")
    p_es = t._classify_phoneme_type(seg, sr, "es")
    assert p_de == "fricative", f"de erwartet fricative, bekam {p_de}"
    assert p_es != "fricative", "es darf bei schwacher HF-Dominanz kein Frikativ melden"


def test_plosive_cap_flip() -> None:
    """27-ms-Impuls: de (Kappe 30 ms) → plosive; es (Kappe 25 ms) → nicht plosive."""
    t = _classifier()
    sr = 16000
    seg = np.zeros(432, dtype=np.float32)  # 27 ms
    seg[0] = 1.0
    seg = np.convolve(seg, np.hanning(64), mode="same").astype(np.float32)
    p_de = t._classify_phoneme_type(seg, sr, "de")
    p_es = t._classify_phoneme_type(seg, sr, "es")
    assert p_de == "plosive", f"de erwartet plosive, bekam {p_de}"
    assert p_es != "plosive", f"es erwartet kein plosive, bekam {p_es}"


def test_profile_table_values() -> None:
    """Tabellen-Konsistenz: de permissiver, es/it/pt konservativer als Default."""
    t = _classifier()
    assert t._LANG_PHONEME_PROFILES["de"]["zcr_fricative"] < t._ZCR_FRICATIVE_THRESHOLD
    assert t._LANG_PHONEME_PROFILES["de"]["hf_ratio"] < 1.0
    for lang in ("es", "it", "pt"):
        assert t._LANG_PHONEME_PROFILES[lang]["zcr_fricative"] > t._ZCR_FRICATIVE_THRESHOLD
        assert t._LANG_PHONEME_PROFILES[lang]["hf_ratio"] > 1.0
        assert t._LANG_PHONEME_PROFILES[lang]["plosive_max_s"] < t._PLOSIVE_MAX_DURATION_S
