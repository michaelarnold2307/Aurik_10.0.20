from __future__ import annotations

import pytest

"""Tests für core/remaster_detector.py — RemasterDetector (§2.14).

≥ 17 Unit-Tests: alle synthetic, kein Audio-File, np.random.seed(42).
"""


import math
import threading

import numpy as np

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(sr: int = 48_000, duration: float = 2.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.asarray(np.sin(2 * np.pi * freq * t), dtype=np.float32)


def _wideband_noise(sr: int = 48_000, duration: float = 2.0, seed: int = 42) -> np.ndarray:
    """Weißes Rauschen, volle Bandbreite 0–Nyquist."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.05


def _digital_silence_with_hiss(
    sr: int = 48_000,
    duration: float = 3.0,
    hiss_rms: float = 1e-5,
    seed: int = 42,
) -> np.ndarray:
    """Simuliert Remaster-Szenario: sehr niedriger Rauschboden + volle Bandbreite.

    Erzeuge sehr leises Rauschen (< −80 dBFS) plus wideband-Komponente
    damit rolloff > 18 kHz erreichbar wird.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    noise = rng.standard_normal(n).astype(np.float32)
    # Skale so dass RMS ≈ hiss_rms (tief genug für floor < -80 dBFS)
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    noise = noise * (hiss_rms / max(noise_rms, 1e-12))
    return noise


# ---------------------------------------------------------------------------
# Import-Guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    """Modul importierbar ohne Fehler."""
    from backend.core.remaster_detector import analyse_remaster

    assert callable(analyse_remaster)


# ---------------------------------------------------------------------------
# T01: Sinus → kein Remaster
# ---------------------------------------------------------------------------


def test_01_sine_not_remaster():
    """Reiner Sinuston hat hohen Rauschboden und schmale BW → is_remaster=False."""
    from backend.core.remaster_detector import analyse_remaster

    audio = _sine()
    result = analyse_remaster(audio, 48_000)
    assert result.is_remaster is False


# ---------------------------------------------------------------------------
# T02: Sehr tiefer Rauschboden + volle BW → Remaster erkannt
# ---------------------------------------------------------------------------


def test_02_deep_floor_wideband_is_remaster():
    """Extrem leises Signal mit voller Bandbreite → is_remaster=True."""
    from backend.core.remaster_detector import analyse_remaster

    np.random.seed(42)
    # Sehr leises weißes Rauschen: RMS ~1e-6 → floor << -80 dBFS
    sr = 48_000
    n = sr * 3
    audio = np.random.randn(n).astype(np.float32) * 1e-6
    result = analyse_remaster(audio, sr)
    # Rauschboden muss sehr niedrig sein
    assert result.noise_floor_db < -80.0
    assert result.is_remaster is True


# ---------------------------------------------------------------------------
# T03: Schmalbandige Quelle → kein Remaster (BW-Kriterium schlägt fehl)
# ---------------------------------------------------------------------------


def test_03_lowpass_not_remaster():
    """Tiefpassgefiltertes Signal (BW < 18 kHz) mit tiefem Floor → kein Remaster."""
    from scipy.signal import firwin, lfilter

    from backend.core.remaster_detector import analyse_remaster

    np.random.seed(42)
    sr = 48_000
    # Tiefpass bis 8 kHz — simuliert alte Schallplatte
    h = firwin(127, 8_000 / (sr / 2))
    wide = np.random.randn(sr * 3).astype(np.float32) * 1e-6
    audio = lfilter(h, 1.0, wide).astype(np.float32)
    result = analyse_remaster(audio, sr)
    # BW-Rolloff sollte < 18 kHz sein
    assert result.hf_rolloff_khz < 18.0
    assert result.is_remaster is False


# ---------------------------------------------------------------------------
# T04: Confidence > 0 wenn Remaster
# ---------------------------------------------------------------------------


def test_04_confidence_positive_when_remaster():
    """Bei erkanntem Remaster muss confidence > 0 sein."""
    from backend.core.remaster_detector import analyse_remaster

    np.random.seed(42)
    sr = 48_000
    audio = np.random.randn(sr * 2).astype(np.float32) * 1e-6
    result = analyse_remaster(audio, sr)
    if result.is_remaster:
        assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# T05: 2D-Stereo-Array (channels-first)
# ---------------------------------------------------------------------------


def test_05_stereo_channels_first():
    """2D-Array (2, N) wird korrekt zu Mono gemittelt."""
    from backend.core.remaster_detector import analyse_remaster

    np.random.seed(42)
    sr = 48_000
    mono = np.random.randn(sr * 2).astype(np.float32) * 1e-6
    stereo = np.stack([mono, mono], axis=0)  # shape (2, N)
    result_mono = analyse_remaster(mono, sr)
    result_stereo = analyse_remaster(stereo, sr)
    # Beide sollten is_remaster gleich sein (identisches Inhalt)
    assert result_mono.is_remaster == result_stereo.is_remaster


# ---------------------------------------------------------------------------
# T06: 2D-Stereo-Array (samples-first)
# ---------------------------------------------------------------------------


def test_06_stereo_samples_first():
    """2D-Array (N, 2) wird korrekt verarbeitet (axis=1 mean)."""
    from backend.core.remaster_detector import analyse_remaster

    np.random.seed(42)
    sr = 48_000
    mono = np.random.randn(sr * 2).astype(np.float32) * 1e-6
    stereo = np.stack([mono, mono], axis=1)  # shape (N, 2)
    result = analyse_remaster(stereo, sr)
    assert isinstance(result.is_remaster, bool)
    assert math.isfinite(result.confidence)


# ---------------------------------------------------------------------------
# T07: Sehr kurzer Clip < 256 Samples → Early Return
# ---------------------------------------------------------------------------


def test_07_short_clip_early_return():
    """Clip < 256 Samples → is_remaster=False, kein Absturz."""
    from backend.core.remaster_detector import analyse_remaster

    audio = np.zeros(100, dtype=np.float32)
    result = analyse_remaster(audio, 48_000)
    assert result.is_remaster is False
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# T08: All-Zeros → kein Remaster
# ---------------------------------------------------------------------------


def test_08_all_zeros_not_remaster():
    """Stille (all zeros) → is_remaster=False, kein Absturz."""
    from backend.core.remaster_detector import analyse_remaster

    audio = np.zeros(48_000, dtype=np.float32)
    result = analyse_remaster(audio, 48_000)
    assert result.is_remaster is False
    assert math.isfinite(result.noise_floor_db)


# ---------------------------------------------------------------------------
# T09: Grenzbedingung — floor=-80 dBFS → is_remaster=False (strict <)
# ---------------------------------------------------------------------------


def test_09_boundary_floor_exactly_threshold():
    """Rauschboden exakt an der Grenze (-80 dBFS) erfüllt die Bedingung nicht.

    floor < -80 ist die Bedingung, also -80.0 self → False.
    """
    from backend.core.remaster_detector import RemasterDetector

    det = RemasterDetector()
    score = det._floor_score(-80.0)
    assert score == 0.0


# ---------------------------------------------------------------------------
# T10: Grenzbedingung — rolloff=18.0 kHz → bw_score=0
# ---------------------------------------------------------------------------


def test_10_boundary_rolloff_exactly_threshold():
    """HF-Rolloff exakt an der Grenze (18.0 kHz) → bw_score=0."""
    from backend.core.remaster_detector import RemasterDetector

    det = RemasterDetector()
    score = det._bw_score(18.0)
    assert score == 0.0


# ---------------------------------------------------------------------------
# T11: floor_score Skalierung
# ---------------------------------------------------------------------------


def test_11_floor_score_scaling():
    """floor=-100 dBFS → floor_score = (-80-(-100))/40 = 0.5."""
    from backend.core.remaster_detector import RemasterDetector

    det = RemasterDetector()
    score = det._floor_score(-100.0)
    assert abs(score - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# T12: bw_score Skalierung
# ---------------------------------------------------------------------------


def test_12_bw_score_scaling():
    """rolloff=20 kHz → bw_score = (20-18)/4 = 0.5."""
    from backend.core.remaster_detector import RemasterDetector

    det = RemasterDetector()
    score = det._bw_score(20.0)
    assert abs(score - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# T13: Singleton-Identität
# ---------------------------------------------------------------------------


def test_13_singleton_identity():
    """get_remaster_detector() gibt immer dasselbe Objekt zurück."""
    from backend.core.remaster_detector import get_remaster_detector

    a = get_remaster_detector()
    b = get_remaster_detector()
    assert a is b


# ---------------------------------------------------------------------------
# T14: Thread-Sicherheit des Singletons
# ---------------------------------------------------------------------------


def test_14_thread_safety():
    """Parallele Zugriffe liefern identisches Singleton-Objekt."""
    from backend.core.remaster_detector import get_remaster_detector

    instances = []
    lock = threading.Lock()

    def _grab() -> None:
        inst = get_remaster_detector()
        with lock:
            instances.append(inst)

    threads = [threading.Thread(target=_grab) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(inst is instances[0] for inst in instances)


# ---------------------------------------------------------------------------
# T15: analyse_remaster() convenience wrapper
# ---------------------------------------------------------------------------


def test_15_convenience_wrapper_matches_direct():
    """analyse_remaster() liefert dasselbe Ergebnis wie .analyse() direkt."""
    from backend.core.remaster_detector import analyse_remaster, get_remaster_detector

    np.random.seed(42)
    audio = np.random.randn(48_000).astype(np.float32) * 0.01
    sr = 48_000
    r1 = get_remaster_detector().analyse(audio, sr)
    r2 = analyse_remaster(audio, sr)
    assert r1.is_remaster == r2.is_remaster
    assert abs(r1.confidence - r2.confidence) < 1e-9
    assert abs(r1.noise_floor_db - r2.noise_floor_db) < 1e-9
    assert abs(r1.hf_rolloff_khz - r2.hf_rolloff_khz) < 1e-9


# ---------------------------------------------------------------------------
# T16: noise_floor_db ≤ 0 für nicht-stilles Audio
# ---------------------------------------------------------------------------


def test_16_noise_floor_nonpositive():
    """noise_floor_db ist immer ≤ 0 dBFS für nicht-stilles Audio."""
    from backend.core.remaster_detector import analyse_remaster

    np.random.seed(42)
    audio = np.random.randn(48_000).astype(np.float32) * 0.1
    result = analyse_remaster(audio, 48_000)
    assert result.noise_floor_db <= 0.0


# ---------------------------------------------------------------------------
# T17: confidence immer in [0, 1]
# ---------------------------------------------------------------------------


def test_17_confidence_bounded():
    """confidence liegt stets in [0.0, 1.0]."""
    from backend.core.remaster_detector import analyse_remaster

    np.random.seed(42)
    for scale in [1e-8, 1e-4, 1e-2, 0.1, 1.0]:
        audio = np.random.randn(48_000).astype(np.float32) * scale
        result = analyse_remaster(audio, 48_000)
        assert 0.0 <= result.confidence <= 1.0, f"confidence={result.confidence} außerhalb [0,1] bei scale={scale}"


# ---------------------------------------------------------------------------
# T18: evidence-Liste NaN-frei (kein NaN in Strings)
# ---------------------------------------------------------------------------


def test_18_evidence_no_nan_strings():
    """evidence-Einträge enthalten keine NaN-Zeichenketten."""
    from backend.core.remaster_detector import analyse_remaster

    np.random.seed(42)
    audio = np.random.randn(48_000).astype(np.float32) * 1e-6
    result = analyse_remaster(audio, 48_000)
    for entry in result.evidence:
        assert "nan" not in entry.lower(), f"NaN in evidence: {entry}"
