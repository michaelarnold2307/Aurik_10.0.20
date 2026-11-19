from __future__ import annotations

"""Tests für backend/core/dsp/pre_echo_detector.py (§4.11)."""


import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def silence_48k():
    """1 Sekunde Stille @ 48 kHz."""
    return np.zeros(48000, dtype=np.float32), 48000


@pytest.fixture
def sine_48k():
    """1 Sekunde Sinus 440 Hz @ 48 kHz (keine Onsets)."""
    t = np.linspace(0, 1.0, 48000, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t), 48000


@pytest.fixture
def impulse_with_pre_echo():
    """Simuliertes MP3-Pre-Echo: Impuls bei t=0.5s mit Energie-Rauschen VOR dem Impuls."""
    sr = 48000
    n = sr  # 1 Sekunde
    audio = np.zeros(n, dtype=np.float32)
    # Starker Transient
    onset_s = int(0.5 * sr)
    audio[onset_s : onset_s + 200] = 0.8
    # Pre-Echo: 20-30 ms VOR dem Transient (Fastl & Zwicker, Temporal Masking)
    pre_echo_start = onset_s - int(0.025 * sr)
    pre_echo_end = onset_s - int(0.005 * sr)
    audio[pre_echo_start:pre_echo_end] = 0.10  # −20 dB unter Transient ≈ Pre-Echo
    return audio, sr


# ---------------------------------------------------------------------------
# Singleton-Tests
# ---------------------------------------------------------------------------


def test_singleton_returns_same_instance():
    from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

    a = get_pre_echo_detector()
    b = get_pre_echo_detector()
    assert a is b, "Singleton-Pattern muss selbe Instanz zurückgeben"


# ---------------------------------------------------------------------------
# SR-Assertion
# ---------------------------------------------------------------------------


def test_sr_assertion_raises():
    from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

    det = get_pre_echo_detector()
    with pytest.raises(AssertionError):
        det.detect(np.zeros(44100, dtype=np.float32), sr=44100)


# ---------------------------------------------------------------------------
# Analog-Material → kein Pre-Echo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("material", ["shellac", "vinyl", "reel_tape", "cassette"])
def test_analog_material_returns_empty(impulse_with_pre_echo, material):
    """Analoge Materialien haben kein Codec-Pre-Echo → leere Liste."""
    from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

    audio, sr = impulse_with_pre_echo
    events = get_pre_echo_detector().detect(audio, sr, material_key=material)
    assert events == [], f"Analog-Material '{material}' sollte kein Pre-Echo melden"


# ---------------------------------------------------------------------------
# Stille → kein Pre-Echo
# ---------------------------------------------------------------------------


def test_silence_returns_empty(silence_48k):
    from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

    audio, sr = silence_48k
    events = get_pre_echo_detector().detect(audio, sr, material_key="mp3_low")
    assert events == []


# ---------------------------------------------------------------------------
# MP3-Material aktiviert Detektion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("material", ["mp3_low", "mp3_high", "aac"])
def test_codec_material_processes(impulse_with_pre_echo, material):
    """Codec-Materialien durchlaufen Detektions-Logik (kein Crash)."""
    from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

    audio, sr = impulse_with_pre_echo
    events = get_pre_echo_detector().detect(audio, sr, material_key=material)
    # Darf nicht None sein; Liste (leer oder mit Events) ist korrekt
    assert isinstance(events, list)


# ---------------------------------------------------------------------------
# Event-Struktur-Validierung
# ---------------------------------------------------------------------------


def test_event_structure(impulse_with_pre_echo):
    """Jedes erkannte Event MUSS alle Pflichtfelder haben."""
    from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

    audio, sr = impulse_with_pre_echo
    events = get_pre_echo_detector().detect(audio, sr, material_key="mp3_low")
    required_keys = {"onset_sample", "pre_echo_start", "pre_echo_end", "severity_db", "confidence"}
    for ev in events:
        for key in required_keys:
            assert key in ev, f"Event fehlt Pflichtfeld '{key}': {ev}"
        assert ev["pre_echo_start"] < ev["pre_echo_end"], "pre_echo_start muss < pre_echo_end"
        assert ev["pre_echo_end"] <= ev["onset_sample"] + 1, "pre_echo_end sollte <= onset_sample"
        assert 0.0 <= ev["confidence"] <= 1.0, "confidence muss ∈ [0, 1]"
        assert ev["severity_db"] >= 0.0, "severity_db muss ≥ 0"


# ---------------------------------------------------------------------------
# repair_region — G_floor-Invariante (§2.62)
# ---------------------------------------------------------------------------


def test_repair_region_gfloor_invariante(impulse_with_pre_echo):
    """repair_region darf Signal nie auf 0 reduzieren (G_floor ≥ 0.10, §2.62)."""
    from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

    audio, sr = impulse_with_pre_echo
    det = get_pre_echo_detector()

    # Synthetisches Event
    event = {
        "onset_sample": int(0.5 * sr),
        "pre_echo_start": int(0.478 * sr),
        "pre_echo_end": int(0.498 * sr),
        "severity_db": 15.0,
        "confidence": 0.8,
    }

    repaired = det.repair_region(audio.copy(), event, sr)

    # Im reparierten Segment darf keine Sample-Energie auf 0 clampen
    seg_orig = audio[event["pre_echo_start"] : event["pre_echo_end"]]
    seg_rep = repaired[event["pre_echo_start"] : event["pre_echo_end"]]

    if np.any(np.abs(seg_orig) > 1e-4):
        # G_floor = 0.10 → mindestens 10% der Original-Energie erhalten
        e_orig = float(np.sqrt(np.mean(seg_orig**2) + 1e-12))
        e_rep = float(np.sqrt(np.mean(seg_rep**2) + 1e-12))
        assert e_rep >= e_orig * 0.09, (
            f"G_floor-Verletzung: repair_region reduzierte Energie von {e_orig:.5f} auf {e_rep:.5f} "
            f"(< {e_orig * 0.10:.5f})"
        )


# ---------------------------------------------------------------------------
# repair_region — kein Signal ausserhalb Fenster verändert
# ---------------------------------------------------------------------------


def test_repair_region_boundary_isolation(impulse_with_pre_echo):
    """repair_region darf NUR im pre_echo_start:pre_echo_end-Bereich eingreifen."""
    from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

    audio, sr = impulse_with_pre_echo
    det = get_pre_echo_detector()

    event = {
        "onset_sample": int(0.5 * sr),
        "pre_echo_start": int(0.478 * sr),
        "pre_echo_end": int(0.498 * sr),
        "severity_db": 10.0,
        "confidence": 0.7,
    }

    repaired = det.repair_region(audio.copy(), event, sr)

    # Bereich VOR pre_echo_start muss unverändert sein
    np.testing.assert_array_equal(
        audio[: event["pre_echo_start"]],
        repaired[: event["pre_echo_start"]],
        err_msg="repair_region hat Bereich VOR pre_echo_start verändert",
    )
    # Bereich NACH pre_echo_end muss unverändert sein
    np.testing.assert_array_equal(
        audio[event["pre_echo_end"] :],
        repaired[event["pre_echo_end"] :],
        err_msg="repair_region hat Bereich NACH pre_echo_end verändert",
    )


# ---------------------------------------------------------------------------
# Stereo-Kompatibilität
# ---------------------------------------------------------------------------


def test_repair_region_stereo():
    """repair_region muss auch mit Stereo-Audio (2, T) funktionieren."""
    from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

    sr = 48000
    audio = np.random.randn(2, sr).astype(np.float32) * 0.1
    audio[:, int(0.5 * sr) : int(0.5 * sr) + 100] = 0.5  # Transient

    event = {
        "onset_sample": int(0.5 * sr),
        "pre_echo_start": int(0.478 * sr),
        "pre_echo_end": int(0.498 * sr),
        "severity_db": 8.0,
        "confidence": 0.6,
    }

    det = get_pre_echo_detector()
    repaired = det.repair_region(audio.copy(), event, sr)
    assert repaired.shape == audio.shape
    assert not np.any(np.isnan(repaired))


# ---------------------------------------------------------------------------
# Threshold-Tabelle Vollständigkeit
# ---------------------------------------------------------------------------


def test_threshold_table_has_unknown_fallback():
    """_PRE_ECHO_THRESHOLDS_DB muss Eintrag 'unknown' haben."""
    from backend.core.dsp.pre_echo_detector import _PRE_ECHO_THRESHOLDS_DB

    assert "unknown" in _PRE_ECHO_THRESHOLDS_DB
    assert _PRE_ECHO_THRESHOLDS_DB["unknown"] > 0.0
