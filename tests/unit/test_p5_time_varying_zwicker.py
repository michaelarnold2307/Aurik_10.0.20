"""§SOTA-P5 (Zeitvariantes Zwicker + exakte Tabellen) — Regressionsschutz.

Deckt:
- EXAKTE a0-/L_TQ-Tabellen nach DIN 45631 (Zwicker-Verfahren) / ISO 532-1:2017
  (a0[1 kHz] = 0 dB, L_TQ[1 kHz] = 0 dB, Tiefstwert −4,1 dB @ 3,15 kHz)
- ``compute_time_varying_loudness`` (DIN-45631/A1-Struktur): Kurzzeit-Lautheit
  + Angriffs-/Abkling-Zeitbewertung + N5/N10-Perzentile
- Deterministisch, NaN-sicher, Stereo→Mittel, kurze Eingaben ohne Fehler

Autor: Aurik Testing Team
"""

import numpy as np
import pytest

import backend.core.dsp.zwicker_loudness as zl

SR = 48000


def _sine(freq_hz: float, amp: float, duration_s: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


class TestExactTables:
    def test_a0_exact_din45631(self):
        i1k = int(np.argmin(np.abs(zl._CENTERS - 1000.0)))
        i25 = int(np.argmin(np.abs(zl._CENTERS - 25.0)))
        i12k = int(np.argmin(np.abs(zl._CENTERS - 12500.0)))
        assert zl._A0[i1k] == pytest.approx(0.0, abs=1e-9)
        assert zl._A0[i25] == pytest.approx(-32.0, abs=1e-9)
        assert zl._A0[i12k] == pytest.approx(-10.8, abs=1e-9)

    def test_ltq_exact_din45631(self):
        i1k = int(np.argmin(np.abs(zl._CENTERS - 1000.0)))
        i25 = int(np.argmin(np.abs(zl._CENTERS - 25.0)))
        i3150 = int(np.argmin(np.abs(zl._CENTERS - 3150.0)))
        assert zl._TQ[i1k] == pytest.approx(0.0, abs=1e-9)
        assert zl._TQ[i25] == pytest.approx(65.0, abs=1e-9)
        assert zl._TQ[i3150] == pytest.approx(-4.1, abs=1e-9)
        # 28 Bänder, 25 Hz – 12,5 kHz
        assert len(zl._A0) == 28 == len(zl._TQ) == len(zl._CENTERS)

    def test_reference_calibration_intact(self):
        # 1 kHz / 40 dB ⇒ ≈ 1 sone bleibt mit exakten Tabellen erhalten
        x = _sine(1000.0, zl._REF_AMPLITUDE)
        r = zl.compute_zwicker_loudness(x, SR)
        assert r.loudness_sone == pytest.approx(1.0, rel=0.15)


class TestTimeVaryingLoudness:
    def test_steady_tone_flat_curve(self):
        x = _sine(1000.0, zl._REF_AMPLITUDE, duration_s=1.0)
        r = zl.compute_time_varying_loudness(x, SR)
        assert len(r.time_seconds) > 10
        assert len(r.loudness_sone_t) == len(r.time_seconds)
        # Nach dem Einschwingen (~erste Frames) konstant
        tail = r.loudness_sone_t[10:]
        assert float(np.std(tail)) < 1e-3 * (float(np.mean(tail)) + 1e-9) + 1e-6
        # Stationäre Referenz: N ≈ 1 sone
        assert float(np.mean(tail)) == pytest.approx(1.0, rel=0.15)

    def test_louder_signal_larger_n5(self):
        base = _sine(1000.0, zl._REF_AMPLITUDE, duration_s=0.5)
        louder = _sine(1000.0, zl._REF_AMPLITUDE * 10.0, duration_s=0.5)
        n5_base = zl.compute_time_varying_loudness(base, SR).n5_sone
        n5_louder = zl.compute_time_varying_loudness(louder, SR).n5_sone
        assert n5_louder > n5_base

    def test_modulated_signal_varies_over_time(self):
        t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
        am = (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)).astype(np.float32)
        x = (zl._REF_AMPLITUDE * 10.0 * np.sin(2 * np.pi * 1000.0 * t) * am).astype(np.float32)
        r = zl.compute_time_varying_loudness(x, SR)
        assert float(np.std(r.loudness_sone_t)) > 0.01  # Modulation sichtbar
        assert r.n5_sone >= r.n10_sone >= 0.0
        assert r.peak_sone >= r.n5_sone

    def test_attack_faster_than_release(self):
        # 300 ms Stille, dann 400 ms Ton, dann 300 ms Stille — Angriff (τ≈5 ms)
        # deutlich schneller als Abklingen (τ≈100 ms). Gemessen über die
        # 63 %-/37 %-Zeitpunkte (Zeitkonstanten-Definition), robust gegen den
        # spektralen Leakage-Artefakt an den Fenstergrenzen (Befund 2026-09-15:
        # teilgefüllte Hann-Fenster verteilen Energie über Bänder und heben die
        # Kurzzeit-Lautheit am Ton-Ende kurz an — kein physikalischer Transient).
        period = np.sin(2 * np.pi * np.arange(48, dtype=np.float32) / 48.0)
        seg = np.tile(period, 400)  # 400 Perioden = 0,4 s
        seg = np.concatenate([seg, np.zeros(1, dtype=np.float32)])
        seg = (zl._REF_AMPLITUDE * 5.0 * seg).astype(np.float32)
        x = np.zeros(SR, dtype=np.float32)
        start = SR // 2 - len(seg) // 2
        x[start : start + len(seg)] = seg
        t_on = start / SR
        t_off = (start + len(seg)) / SR

        r = zl.compute_time_varying_loudness(x, SR, window_ms=10.0, hop_ms=2.0)
        n_t = r.loudness_sone_t
        t = r.time_seconds
        # Plateau: mittlere Lautheit in der Ton-Mitte
        mid = n_t[(t > t_on + 0.15) & (t < t_off - 0.15)]
        plateau = float(np.median(mid))
        assert plateau > 0.0

        # 63 %-Zeitpunkt (Angriff) und 37 %-Zeitpunkt (Abklingen)
        after_on = t > t_on + 0.005  # Fenster-Anlauf ausblenden
        rise_idx = int(np.flatnonzero(after_on & (n_t >= 0.63 * plateau))[0])
        fall_mask = (t > t_off) & (n_t <= 0.37 * plateau)
        fall_idx = int(np.flatnonzero(fall_mask)[0])
        rise_ms = (t[rise_idx] - t_on) * 1000.0
        fall_ms = (t[fall_idx] - t_off) * 1000.0

        assert rise_ms < 25.0  # Angriff: τ 5 ms + Fenster 10 ms
        assert fall_ms > 60.0  # Abklingen: τ 100 ms
        assert rise_ms < fall_ms

    def test_determinism(self):
        x = _sine(440.0, zl._REF_AMPLITUDE * 3.0, duration_s=0.5)
        r1 = zl.compute_time_varying_loudness(x, SR)
        r2 = zl.compute_time_varying_loudness(x, SR)
        assert np.array_equal(r1.loudness_sone_t, r2.loudness_sone_t)
        assert r1.n5_sone == r2.n5_sone

    def test_short_input_empty_result_no_error(self):
        r = zl.compute_time_varying_loudness(np.zeros(100, dtype=np.float32), SR)
        assert len(r.time_seconds) == 0
        assert r.n5_sone == 0.0

    def test_nan_guarded_and_stereo_mean(self):
        x = _sine(1000.0, zl._REF_AMPLITUDE, duration_s=0.5)
        x[100] = np.nan
        r = zl.compute_time_varying_loudness(x, SR)
        assert np.isfinite(r.loudness_sone_t).all()
        stereo = np.stack([x, x], axis=0)
        r_st = zl.compute_time_varying_loudness(stereo, SR)
        mono = np.nan_to_num(x, nan=0.0)
        r_mo = zl.compute_time_varying_loudness(mono, SR)
        assert np.allclose(r_st.loudness_sone_t, r_mo.loudness_sone_t, atol=1e-9)
