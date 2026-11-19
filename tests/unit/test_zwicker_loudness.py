"""Unit-Tests für backend/core/dsp/zwicker_loudness.py — ISO 532-1 stationär.

Abdeckung:
  - Referenz-Kalibrierung: 1 kHz Sinus bei 40 dB SPL ⇒ N ≈ 1 sone (±15 %),
    L_N ≈ 40 phon.
  - Pegel-Monotonie: lauteres Signal ⇒ größeres N (sone).
  - Determinismus: zwei identische Aufrufe bit-identisch (§G5 (GEBOTE.md)).
  - NaN/Inf-Freiheit bei Stille und Impulsen (§0a).
  - Maskierungsschwelle: unter Erregung, (mindestens) auf Ruhehörschwelle,
    monoton im Masker-Pegel.
  - Gate-Integration: model="zwicker" läuft deterministisch, liefert 4 Keys;
    fail-open ImportError ⇒ Fallback MPEG-1 (Ergebnis == model="mpeg1").

Hinweis zur Maskierungsschwelle: die untere Flanke (~27 dB/Bark) senkt die
Schwelle über ~4,6 Bark Abstand (1 kHz → 2 kHz) um >120 dB, sodass die
Schwelle dort physikalisch korrekt auf die Ruhehörschwelle zurückfällt. Für
den strengen „über Ruhehörschwelle"- und Monotonie-Check wird daher eine
nähere Frequenz (~1,3 kHz) verwendet, wo Maskierung real stattfindet — die
2-kHz-Prüfung beschränkt sich auf „unter Masker-Erregung und ≥ Ruhehörschwelle".

Keine brittlen Goldwerte; großzügige physikalische Plausibilitäts-Grenzen.
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.core.dsp.audibility_gate as ag
import backend.core.dsp.zwicker_loudness as zl

SR = 48000


def _sine(freq_hz: float, amp: float, duration_s: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(duration_s * SR), endpoint=False)
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float64)


def test_reference_1khz_40db_is_1_sone() -> None:
    """1 kHz Sinus bei 40 dB SPL (Ref-Amplitude) ⇒ N ≈ 1 sone, L_N ≈ 40 phon."""
    x = _sine(1000.0, zl._REF_AMPLITUDE)
    r = zl.compute_zwicker_loudness(x, SR)
    assert r.loudness_sone == pytest.approx(1.0, rel=0.15)
    assert r.loudness_level_phon == pytest.approx(40.0, abs=3.0)
    # Kurzformen konsistent
    assert zl.compute_loudness_sone(x, SR) == pytest.approx(r.loudness_sone, abs=1e-12)
    assert zl.compute_loudness_phon(x, SR) == pytest.approx(r.loudness_level_phon, abs=1e-12)


def test_level_monotonicity() -> None:
    """Lauteres Signal ⇒ größeres N (sone) und größerer L_N (phon)."""
    base = _sine(1000.0, zl._REF_AMPLITUDE)
    louder = _sine(1000.0, zl._REF_AMPLITUDE * 10.0)  # +20 dB
    n_base = zl.compute_loudness_sone(base, SR)
    n_louder = zl.compute_loudness_sone(louder, SR)
    assert n_louder > n_base
    assert zl.compute_loudness_phon(louder, SR) > zl.compute_loudness_phon(base, SR)


def test_determinism() -> None:
    """Zwei identische Aufrufe bit-identisch (§G5 (GEBOTE.md))."""
    x = _sine(440.0, zl._REF_AMPLITUDE) + 0.5 * _sine(880.0, zl._REF_AMPLITUDE * 0.5)
    r1 = zl.compute_zwicker_loudness(x, SR)
    r2 = zl.compute_zwicker_loudness(x, SR)
    assert r1.loudness_sone == r2.loudness_sone
    assert r1.loudness_level_phon == r2.loudness_level_phon
    t1, f1 = zl.zwicker_masking_threshold_db(x, SR)
    t2, f2 = zl.zwicker_masking_threshold_db(x, SR)
    assert np.array_equal(t1, t2)
    assert np.array_equal(f1, f2)


def test_nan_inf_free_silence_and_impulse() -> None:
    """Keine NaN/Inf bei Stille und Impulsen (§0a)."""
    silence = np.zeros(SR, dtype=np.float32)
    impulse = np.zeros(SR, dtype=np.float32)
    impulse[SR // 2] = 1.0
    nan_sig = np.full(SR, np.nan, dtype=np.float32)
    for sig in (silence, impulse, nan_sig):
        r = zl.compute_zwicker_loudness(sig, SR)
        assert np.isfinite(r.loudness_sone)
        assert np.isfinite(r.loudness_level_phon)
        thr, freq = zl.zwicker_masking_threshold_db(sig, SR)
        assert np.all(np.isfinite(thr))
        assert np.all(np.isfinite(freq))
        assert r.loudness_sone >= 0.0


def test_masking_threshold_sanity_and_monotonicity() -> None:
    """Schwelle unter Masker-Erregung, (≥) Ruhehörschwelle, monoton im Pegel."""
    masker = _sine(1000.0, zl._REF_AMPLITUDE * 10.0)  # 60 dB SPL
    thr, freq = zl.zwicker_masking_threshold_db(masker, SR)
    exc, bark = zl.zwicker_excitation_pattern(masker, SR)

    i1k = int(np.argmin(np.abs(freq - 1000.0)))
    i2k = int(np.argmin(np.abs(freq - 2000.0)))
    i13 = int(np.argmin(np.abs(freq - 1300.0)))

    # 1. Schwelle ist am Masker-Ort nicht über der Erregung (Schwelle ≤ Erregung).
    assert thr[i1k] <= exc[i1k] + 0.1

    # 2. Schwelle bei 2 kHz: deutlich UNTER der Masker-Erregung (bei 1 kHz) und
    #    auf/über der Ruhehörschwelle. Mit den EXAKTEN DIN-45631-L_TQ-Tabellen
    #    (Update 2026-09-15) liegt der Tiefstwert bei −4,1 dB (3,15 kHz) und
    #    2 kHz bei −3,0 dB — der Floor ist also negativ und physikalisch korrekt.
    assert thr[i2k] < exc[i1k] - 25.0
    assert thr[i2k] >= float(np.min(zl._TQ)) - 1e-6  # nie unter L_TQ-Floor

    # 3. Monotonie an einer Frequenz, wo Maskierung real ist (~1,3 kHz):
    #    lauterer Masker ⇒ höhere Schwelle.
    louder = _sine(1000.0, zl._REF_AMPLITUDE * 50.0)  # ~74 dB SPL
    thr_louder, _ = zl.zwicker_masking_threshold_db(louder, SR)
    assert thr_louder[i13] > thr[i13]


def test_excitation_pattern_shape() -> None:
    """Erregungsmuster deckt 0–24 Bark in 0,5-Schritten ab (49 Punkte)."""
    x = _sine(440.0, zl._REF_AMPLITUDE)
    exc, bark = zl.zwicker_excitation_pattern(x, SR)
    assert exc.shape == bark.shape
    assert bark[0] == 0.0
    assert bark[-1] <= zl._BARK_MAX + 1e-9
    assert np.all(np.diff(bark) > 0.0)


def test_gate_zwicker_path_runs_and_returns_4_keys() -> None:
    """defect_audibility(..., model='zwicker') läuft, deterministisch, 4 Keys."""
    sr = SR
    rng = np.random.RandomState(0)
    x = (rng.randn(sr) * 0.02).astype(np.float32)
    d0, d1 = sr // 2 - 64, sr // 2 + 64
    x[d0:d1] += 0.6
    r1 = ag.defect_audibility(x, sr, d0, d1, model="zwicker")
    r2 = ag.defect_audibility(x, sr, d0, d1, model="zwicker")
    assert set(r1.keys()) == {"audible", "delta_db", "threshold_db", "skippable"}
    assert r1 == r2  # deterministisch (§G5 (GEBOTE.md))


def test_gate_zwicker_fail_open_falls_back_to_mpeg1(monkeypatch: pytest.MonkeyPatch) -> None:
    """ImportError im Zwicker-Pfad ⇒ Fallback MPEG-1 (Ergebnis == model='mpeg1')."""
    sr = SR
    rng = np.random.RandomState(1)
    x = (rng.randn(sr) * 0.03).astype(np.float32)
    d0, d1 = sr // 2 - 32, sr // 2 + 32
    x[d0:d1] += 0.5

    def _boom(*args, **kwargs):
        raise ImportError("zwicker_loudness nicht verfügbar")

    monkeypatch.setattr(ag, "_defect_audibility_zwicker", _boom)
    res_zwicker = ag.defect_audibility(x, sr, d0, d1, model="zwicker")
    res_mpeg1 = ag.defect_audibility(x, sr, d0, d1, model="mpeg1")
    assert res_zwicker == res_mpeg1
