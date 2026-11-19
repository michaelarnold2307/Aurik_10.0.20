"""
§2.30 MDEM Quiet-Zone Guard — Regressionstest für _morph_internal Retry-Pfad.

Sichert: Kein positiver Gain-Boost in der Quiet-Zone (< -36 dBFS) während des
Retry-Pfades (_morph_internal), analog zum Quiet-Zone-Guard in morph().

Regressionsfall: Vinyl-Song, Beginn/Ende enthält Oberflächen-Rauschen auf
-35 bis -45 dBFS (Originalträger). Nach Denoising: -45 bis -60 dBFS (restauriert).
_morph_internal hat ohne Guard: G[k] = clip(-35 - (-45), -4, +4) = +4 dB → Lautstärke-Burst.
"""

import math

import numpy as np
import pytest


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
    return 20.0 * math.log10(rms + 1e-12)


@pytest.fixture
def mdem():
    from backend.core.micro_dynamics_envelope_morphing import get_mdem

    return get_mdem()


def _make_vinyl_audio(sr: int = 48000) -> tuple[np.ndarray, np.ndarray]:
    """
    Erstellt synthetisches Vinyl-Audio:
    - orig:    Anfang/Ende: Oberflächenrauschen bei ~-35 dBFS; Mitte: Musik bei -18 dBFS
    - restored: Anfang/Ende: gereinigtes Rauschen bei ~-48 dBFS; Mitte: Musik unverändert
    """
    total_samples = sr * 10  # 10 Sekunden
    intro_len = sr * 2  # 2 s Intro-Vinyl-Rauschen
    outro_len = sr * 2  # 2 s Outro-Vinyl-Rauschen
    music_len = total_samples - intro_len - outro_len

    rng = np.random.default_rng(42)

    # Original: Intro-Rauschen ~-35 dBFS
    amplitude_intro = 10.0 ** (-35.0 / 20.0)
    intro = rng.uniform(-amplitude_intro, amplitude_intro, intro_len).astype(np.float32)

    # Original: Musik ~-18 dBFS (Sinuston + Noise)
    t = np.arange(music_len, dtype=np.float32) / sr
    amplitude_music = 10.0 ** (-18.0 / 20.0)
    music = (amplitude_music * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    music += rng.uniform(-0.01, 0.01, music_len).astype(np.float32)

    # Original: Outro-Rauschen ~-35 dBFS
    outro = rng.uniform(-amplitude_intro, amplitude_intro, outro_len).astype(np.float32)

    orig = np.concatenate([intro, music, outro]).astype(np.float32)

    # Restored: Intro/Outro gereinigt zu ~-48 dBFS
    amplitude_cleaned = 10.0 ** (-48.0 / 20.0)
    restored_intro = rng.uniform(-amplitude_cleaned, amplitude_cleaned, intro_len).astype(np.float32)
    restored_outro = rng.uniform(-amplitude_cleaned, amplitude_cleaned, outro_len).astype(np.float32)
    restored = np.concatenate([restored_intro, music, restored_outro]).astype(np.float32)

    return orig, restored


@pytest.mark.unit
def test_mdem_no_boost_in_quiet_zone_main_path(mdem):
    """morph() Hauptpfad: Keine Lautstärkeerhöhung im Quiet-Zone-Bereich."""
    orig, restored = _make_vinyl_audio()

    result = mdem.morph(restored, orig, sr=48000, mode="restoration")

    intro_len = 48000 * 2
    outro_start = len(result) - 48000 * 2

    restored_intro_level = _rms_dbfs(restored[:intro_len])
    result_intro_level = _rms_dbfs(result[:intro_len])

    restored_outro_level = _rms_dbfs(restored[outro_start:])
    result_outro_level = _rms_dbfs(result[outro_start:])

    # Kein Boost erlaubt: Mit §2.30b Per-Sample Guard darf maximal +1 dB Boost auftreten
    # (Toleranz für Frame-Randfälle an der Musik/Stille-Grenze).
    assert result_intro_level <= restored_intro_level + 1.0, (
        f"morph() Intro: result={result_intro_level:.1f} dBFS > restored+1={restored_intro_level + 1:.1f} dBFS"
    )
    assert result_outro_level <= restored_outro_level + 1.0, (
        f"morph() Outro: result={result_outro_level:.1f} dBFS > restored+1={restored_outro_level + 1:.1f} dBFS"
    )


def test_mdem_no_boost_in_quiet_zone_morph_internal(mdem):
    """_morph_internal() Retry-Pfad: Keine Lautstärkeerhöhung im Quiet-Zone-Bereich.

    Dieser Test schlägt ohne den -36 dBFS Guard in _morph_internal fehl.
    """
    orig, restored = _make_vinyl_audio()
    orig_mono = orig.copy()
    res_mono = restored.copy()

    result = mdem._morph_internal(res_mono, orig_mono, max_gain=4.0)

    intro_len = 48000 * 2
    outro_start = len(result) - 48000 * 2

    restored_intro_level = _rms_dbfs(restored[:intro_len])
    result_intro_level = _rms_dbfs(result[:intro_len])

    restored_outro_level = _rms_dbfs(restored[outro_start:])
    result_outro_level = _rms_dbfs(result[outro_start:])

    # Kein positiver Boost erlaubt (max. +1 dB Toleranz für Übergangsglättung)
    assert result_intro_level <= restored_intro_level + 1.0, (
        f"_morph_internal Intro: result={result_intro_level:.1f} dBFS > restored+1={restored_intro_level + 1:.1f} dBFS "
        f"— quiet-zone guard fehlt im Retry-Pfad! Bug §2.30"
    )
    assert result_outro_level <= restored_outro_level + 1.0, (
        f"_morph_internal Outro: result={result_outro_level:.1f} dBFS > restored+1={restored_outro_level + 3:.1f} dBFS "
        f"— quiet-zone guard fehlt im Retry-Pfad! Bug §2.30"
    )


def test_mdem_music_section_not_suppressed(mdem):
    """Sichert: Die Musiksektion in der Mitte wird durch den Quiet-Zone-Guard nicht unterdrückt."""
    orig, restored = _make_vinyl_audio()
    res_mono = restored.copy()
    orig_mono = orig.copy()

    result = mdem._morph_internal(res_mono, orig_mono, max_gain=4.0)

    intro_len = 48000 * 2
    outro_start = len(result) - 48000 * 2
    music_result = result[intro_len:outro_start]
    music_restored = restored[intro_len:outro_start]

    # Musik sollte nicht stark unterdrückt sein (max. -3 dB Toleranz)
    music_result_level = _rms_dbfs(music_result)
    music_restored_level = _rms_dbfs(music_restored)

    assert music_result_level >= music_restored_level - 3.0, (
        f"Musiksektion wurde zu stark gedämpft: result={music_result_level:.1f} dBFS "
        f"vs restored={music_restored_level:.1f} dBFS"
    )


def test_mdem_no_boost_at_music_to_quiet_boundary(mdem):
    """§2.30b Per-Sample Guard: Positiver Gain an der Musik→Quiet-Zone-Grenze.

    Reproduziert die Lücke: Frame k hat Musik (+4 dB Boost), Frame k+1 hat
    Quiet-Zone (G=0 durch Post-Smoothing Guard). linspace interpoliert von 1.585
    auf 1.0 über HOP=9600 Samples — aber das Audio ist bereits im Quiet-Zone
    (<-36 dBFS), während die Interpolations-Ramp noch > 1.0 ist.
    Ohne §2.30b Per-Sample Guard: Pegelexplosion im Übergangsbereich.
    """
    sr = 48000
    FSIZE = 19200  # 400 ms Frame

    total = sr * 3  # 3 Sekunden
    music_end = FSIZE  # = 19200 Samples (= 400 ms)

    rng = np.random.default_rng(99)

    # Original: Musik bei -18 dBFS, dann Rauschen bei -35 dBFS
    orig = np.zeros(total, dtype=np.float32)
    t = np.arange(music_end) / sr
    orig[:music_end] = float(10 ** (-18 / 20)) * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    orig[music_end:] = rng.uniform(-float(10 ** (-35 / 20)), float(10 ** (-35 / 20)), total - music_end).astype(
        np.float32
    )

    # Restauriert: Musik leicht schwächer (-24 dBFS), Quiet-Zone bei -50 dBFS
    rest = np.zeros(total, dtype=np.float32)
    rest[:music_end] = float(10 ** (-24 / 20)) * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rest[music_end:] = rng.uniform(-float(10 ** (-50 / 20)), float(10 ** (-50 / 20)), total - music_end).astype(
        np.float32
    )

    result = mdem.morph(rest, orig, sr=sr, mode="restoration")

    # Quiet-Zone-Bereich (ab music_end) darf keinen positiven Boost haben
    rest_peak = float(np.max(np.abs(rest[music_end:])) + 1e-12)
    result_peak = float(np.max(np.abs(result[music_end:])) + 1e-12)
    boost_db = 20.0 * math.log10(result_peak / rest_peak)

    assert boost_db <= 1.0, (
        f"§2.30b Per-Sample Guard verletzt: Pegelexplosion im Quiet-Zone-Übergang: "
        f"+{boost_db:.1f} dB (Grenzwert: +1 dB). "
        f"Interpolations-Ramp aus Musik-Frame noch aktiv obwohl Audio bereits -50 dBFS."
    )
