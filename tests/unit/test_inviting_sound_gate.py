"""Tests für das Einladungs-Gate (Hörordnung Ebene 4, §6)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.inviting_sound_gate import (
    InvitingGateResult,
    check_inviting_gate,
    compute_sharpness_acum,
    get_inviting_gate,
)


def _sine(sr: int = 48000, dur_s: float = 12.0, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * dur_s)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_smooth_sine_passes() -> None:
    sr = 48000
    res = check_inviting_gate(_sine(sr), sr, fatigue_index=0.1)
    assert res.passed is True
    assert res.max_asper_in_voice <= 0.5
    assert res.n_windows >= 2


def test_am_roughness_detected_vs_sine() -> None:
    sr = 48000
    t = np.arange(sr * 12) / sr
    am = (0.3 * (1 + 0.9 * np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    res_am = check_inviting_gate(am, sr, fatigue_index=0.0)
    res_sine = check_inviting_gate(_sine(sr), sr, fatigue_index=0.0)
    # AM (70 Hz Modulation = Rauigkeit) muss deutlich rauer sein als Sinus
    assert res_am.max_asper > res_sine.max_asper + 0.05


def test_fatigue_abort_flag() -> None:
    sr = 48000
    res = check_inviting_gate(_sine(sr, dur_s=8.0), sr, fatigue_index=0.55)
    assert res.fatigue_abort is True
    res_ok = check_inviting_gate(_sine(sr, dur_s=8.0), sr, fatigue_index=0.1)
    assert res_ok.fatigue_abort is False


def test_sharpness_plausibility() -> None:
    sr = 48000
    rng = np.random.default_rng(7)
    bright = np.diff(np.cumsum(rng.standard_normal(sr * 6 + 1))).astype(np.float32) * 0.2
    sine_sharp = compute_sharpness_acum(_sine(sr, dur_s=6.0), sr)
    bright_sharp = compute_sharpness_acum(bright, sr)
    assert 0.0 <= sine_sharp <= 4.0
    assert bright_sharp > sine_sharp  # helles Rauschen ist schärfer als ein 440-Hz-Sinus


def test_short_audio_skipped_gracefully() -> None:
    sr = 48000
    res = check_inviting_gate(np.zeros(sr // 4, dtype=np.float32), sr)
    assert res.details.get("skipped") is not None


def test_singleton_and_defaults() -> None:
    gate = get_inviting_gate()
    assert gate is get_inviting_gate()
    r = InvitingGateResult()
    assert r.passed is True
    assert r.fatigue_abort is False


def test_fluctuation_pumping_detected_delta_basiert() -> None:
    """Hörordnung §6 (2026-11-08): Pumpen ist ein DELTA-Befund — gepumptes
    Material gegenüber ungepumptem Input schlägt an; ohne Referenz ist der
    vacil-Wert rein diagnostisch (musikalischer Puls treibt echte Musik bis
    zum Mess-Clip — kein Absolut-Kriterium).
    """
    sr = 48000
    rng = np.random.default_rng(7)
    noise = (rng.standard_normal(sr * 30) * 0.05).astype(np.float32)
    t = np.arange(sr * 30) / sr
    pump = (noise * (1.0 + 0.9 * np.sin(2 * np.pi * 4.0 * t)).astype(np.float32)).astype(np.float32)
    # Mit Referenz (delta-basiert): ungepumpt → keine Regression; gepumpt → Regression.
    res_plain = check_inviting_gate(noise, sr, fatigue_index=0.0, reference_audio=noise)
    res_pump = check_inviting_gate(pump, sr, fatigue_index=0.0, reference_audio=noise)
    assert res_plain.fluctuation_regression is False
    assert res_pump.max_vacil > 0.25, f"gepumpt {res_pump.max_vacil:.4f} vacil nicht gemessen"
    assert res_pump.fluctuation_regression is True
    assert res_pump.passed is False
    assert any("fluctuation_regression" in f for f in res_pump.details["failures"])
    assert res_pump.max_vacil_input == pytest.approx(res_plain.max_vacil, abs=0.01)
    # Ohne Referenz: diagnostisch — auch ein hoher Absolutwert darf NICHT failed
    # setzen (musikalischer Puls ist kein Pumpen-Befund).
    res_no_ref = check_inviting_gate(pump, sr, fatigue_index=0.0)
    assert res_no_ref.fluctuation_regression is False
    assert res_no_ref.passed is True


def test_fluctuation_puls_kein_false_positive_mit_referenz() -> None:
    """Guard-Kalibrierungs-Grundsatz: identischer musikalischer Puls in Input
    und Output ist KEIN Pumpen-Befund (beide Seiten am Mess-Clip).
    """
    sr = 48000
    rng = np.random.default_rng(11)
    pulse = (rng.standard_normal(sr * 30) * 0.05).astype(np.float32)
    t = np.arange(sr * 30) / sr
    beat = 0.4 * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2.0 * t))).astype(np.float32)  # 120 BPM-Puls
    music = (pulse + beat).astype(np.float32)
    res = check_inviting_gate(music, sr, fatigue_index=0.0, reference_audio=music)
    assert res.fluctuation_regression is False  # identische Referenz ⇒ kein Delta


def test_loudness_jump_delta_basiert() -> None:
    """Hörordnung §6: Lautheits-Sprünge sind nur als REGRESSION gegenüber der
    Referenz ein Befund — natürliche Dynamik (Sprünge, die schon im Input
    waren) ist kein Artefakt.
    """
    sr = 48000
    rng = np.random.default_rng(23)
    quiet = (rng.standard_normal(sr * 6) * 0.01).astype(np.float32)
    loud = (rng.standard_normal(sr * 6) * 0.04).astype(np.float32)  # +12 dB
    const = (rng.standard_normal(sr * 12) * 0.02).astype(np.float32)
    stepped = np.concatenate([quiet, loud]).astype(np.float32)
    # Referenz konstant, Output gestuft ⇒ Sprung-Regression.
    res = check_inviting_gate(stepped, sr, fatigue_index=0.0, reference_audio=const)
    assert res.loudness_jump_max > res.loudness_jump_max_input + 1.0
    assert res.loudness_jump_regression is True
    assert res.passed is False
    assert any("loudness_jump_regression" in f for f in res.details["failures"])
    # Identische Referenz ⇒ natürliche Dynamik, kein Befund.
    res_same = check_inviting_gate(stepped, sr, fatigue_index=0.0, reference_audio=stepped)
    assert res_same.loudness_jump_regression is False
    # Ohne Referenz: rein diagnostisch.
    res_noref = check_inviting_gate(stepped, sr, fatigue_index=0.0)
    assert res_noref.loudness_jump_regression is False
