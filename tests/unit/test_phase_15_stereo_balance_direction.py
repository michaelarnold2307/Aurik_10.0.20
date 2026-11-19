#!/usr/bin/env python3
"""Richtungs-Pins der Stereo-Balance-Phase (Befund 2026-11-18, Stereo-Sinn-Beweis).

Befund: phase_15 korrigierte bei rechts-lauterer Imbalance in die FALSCHE
Richtung (−4,09 → −7,67 dB) — die Gains im else-Zweig von
``_apply_smooth_correction`` waren vertauscht. Diese Tests pinnen die
Richtungs-Korrektheit beider Vorzeichen dauerhaft (§III Stereo-Layout-Invariante
(copilot-instructions.md), Paritäts-Register „Stereo-Imaging/Azimuth“, §G5 (GEBOTE.md)).
"""

from __future__ import annotations

import numpy as np

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_15_stereo_balance import StereoBalancePhaseV2


def _balance_db(audio: np.ndarray) -> float:
    left_rms = float(np.sqrt(np.mean(audio[:, 0] ** 2)))
    right_rms = float(np.sqrt(np.mean(audio[:, 1] ** 2)))
    return 20.0 * np.log10((left_rms + 1e-12) / (right_rms + 1e-12))


def _run(audio: np.ndarray) -> np.ndarray:
    ph = StereoBalancePhaseV2()
    res = ph.process(
        np.asarray(audio, dtype=np.float32),
        sample_rate=48000,
        material_type=MaterialType.VINYL,
        strength=1.0,
    )
    return np.asarray(res.audio, dtype=np.float64)


def test_rechts_lauter_wird_ausgeglichen() -> None:
    """Rechts-lautere Imbalance (−6 dB) MUSS Richtung 0 korrigiert werden."""
    rng = np.random.default_rng(7)
    n = 48000 * 2
    noise = rng.standard_normal((n, 2)) * 0.05
    noise[:, 0] *= 10 ** (-6.0 / 20)  # links leiser
    vorher = _balance_db(noise)
    nachher = _balance_db(_run(noise))
    assert vorher < -5.0, f"Test-Setup falsch: vorher={vorher:+.2f} dB"
    assert nachher > vorher, f"Richtung falsch: {vorher:+.2f} -> {nachher:+.2f} dB"
    assert abs(nachher) < abs(vorher), f"Imbalance nicht verkleinert: {vorher:+.2f} -> {nachher:+.2f} dB"


def test_links_lauter_wird_ausgeglichen() -> None:
    """Links-lautere Imbalance (+6 dB) MUSS Richtung 0 korrigiert werden."""
    rng = np.random.default_rng(11)
    n = 48000 * 2
    noise = rng.standard_normal((n, 2)) * 0.05
    noise[:, 1] *= 10 ** (-6.0 / 20)  # rechts leiser
    vorher = _balance_db(noise)
    nachher = _balance_db(_run(noise))
    assert vorher > 5.0, f"Test-Setup falsch: vorher={vorher:+.2f} dB"
    assert nachher < vorher, f"Richtung falsch: {vorher:+.2f} -> {nachher:+.2f} dB"
    assert abs(nachher) < abs(vorher), f"Imbalance nicht verkleinert: {vorher:+.2f} -> {nachher:+.2f} dB"


def test_korrektur_ueberschiesst_nicht() -> None:
    """Kein Überschießen: die Korrektur darf das Vorzeichen nicht umkehren."""
    rng = np.random.default_rng(23)
    n = 48000 * 2
    noise = rng.standard_normal((n, 2)) * 0.05
    noise[:, 0] *= 10 ** (-4.0 / 20)
    nachher = _balance_db(_run(noise))
    assert nachher <= 0.0, f"Vorzeichen umgekehrt: nachher={nachher:+.2f} dB"


def test_ausgeglichen_bleibt_stabil() -> None:
    """Ausgeglichenes Signal (unterhalb der Vinyl-Schwelle 1,0 dB) bleibt stabil."""
    rng = np.random.default_rng(31)
    n = 48000 * 2
    noise = rng.standard_normal((n, 2)) * 0.05
    vorher = _balance_db(noise)
    nachher = _balance_db(_run(noise))
    assert abs(nachher - vorher) < 0.4, f"zu viel Bewegung: {vorher:+.3f} -> {nachher:+.3f} dB"


def test_mono_passthrough() -> None:
    """Mono-Eingabe wird unverändert durchgereicht (Stereo-Layout-Invariante)."""
    ph = StereoBalancePhaseV2()
    x = np.zeros((48000,), dtype=np.float32)
    res = ph.process(x, sample_rate=48000, material_type=MaterialType.VINYL, strength=1.0)
    assert res.metadata.get("stereo") is False
    assert res.metadata.get("correction_applied") is False
