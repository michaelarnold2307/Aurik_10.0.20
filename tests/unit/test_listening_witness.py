"""Unit-Tests für backend/core/listening_witness.py (👂 Reinhör-Witness, 2026-09-11).

Der Witness hört deterministisch in das Phase-Delta hinein und meldet
Pitch-Instabilität, Stimm-Verzerrung und Lautstärke-Pumpen — REPORT-ONLY.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from backend.core.listening_witness import ListeningWitnessResult, evaluate_listening_witness

SR = 48000
T = 6.0
N = int(SR * T)


def _tone_vibrato(f0: float = 220.0, vibrato_hz: float = 0.0, depth_cents: float = 0.0) -> np.ndarray:
    """Synthetischer stimmhafter Ton (mit optionalem Vibrato)."""
    t = np.arange(N) / SR
    depth = depth_cents / 1200.0
    if vibrato_hz > 0:
        f_inst = f0 * (2.0 ** (depth * np.sin(2 * np.pi * vibrato_hz * t)))
        phase = 2 * np.pi * np.cumsum(f_inst) / SR
    else:
        phase = 2 * np.pi * f0 * t
    x = 0.6 * np.sin(phase) + 0.2 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    return (x / np.max(np.abs(x))).astype(np.float32)


def test_no_findings_on_identical_audio() -> None:
    x = _tone_vibrato()
    res = evaluate_listening_witness(x, x.copy(), SR, "phase_x")
    assert isinstance(res, ListeningWitnessResult)
    assert res.findings == []
    assert res.pitch_drift_cents < 1.0


def test_deterministic() -> None:
    x = _tone_vibrato()
    y = _tone_vibrato(f0=221.0)
    r1 = evaluate_listening_witness(x, y, SR, "phase_x")
    r2 = evaluate_listening_witness(x, y, SR, "phase_x")
    assert r1.as_dict() == r2.as_dict()


def test_air_gain_db_reports_air_band_delta() -> None:
    """§Witness↔Goals-Harmonisierung: Brillianz-Zeuge (Luftband-Delta, signiert)."""
    x = _tone_vibrato()
    t = np.arange(N) / SR
    air = 0.02 * np.sin(2 * np.pi * 12000.0 * t)  # 12-kHz-Komponente = Luftband
    res_boost = evaluate_listening_witness(x, (x + air).astype(np.float32), SR, "phase_air")
    assert res_boost.air_gain_db > 0.0, f"Luftband-Boost muss positiv gemeldet werden: {res_boost.air_gain_db}"
    assert "air_loss" not in res_boost.findings

    # Identität: Delta ≈ 0, kein Befund.
    res_id = evaluate_listening_witness(x, x.copy(), SR, "phase_air_id")
    assert abs(res_id.air_gain_db) < 0.5, f"Identität darf kein Luftband-Delta melden: {res_id.air_gain_db}"
    assert "air_loss" not in res_id.findings


def test_air_loss_finding_on_band_suppression() -> None:
    """Starke Luftband-Dämpfung (z. B. muffige Restauration) wird als air_loss gemeldet."""
    x = _tone_vibrato()
    t = np.arange(N) / SR
    air = 0.02 * np.sin(2 * np.pi * 12000.0 * t)
    src = (x + air).astype(np.float32)
    from scipy.signal import butter, sosfiltfilt

    _sos = butter(4, 6000.0, btype="lowpass", fs=SR, output="sos")
    muffled = sosfiltfilt(_sos, src).astype(np.float32)
    res = evaluate_listening_witness(src, muffled, SR, "phase_muffle")
    assert res.air_gain_db < -1.0, f"Luftband-Dämpfung muss negativ gemeldet werden: {res.air_gain_db}"
    assert "air_loss" in res.findings


def test_pitch_drift_detected() -> None:
    """+50 Cent Versatz ⇒ pitch_instability (JND ~5–10 Cent, Schwelle 15 Cent)."""
    x = _tone_vibrato(f0=220.0)
    y = _tone_vibrato(f0=220.0 * (2.0 ** (50.0 / 1200.0)))
    res = evaluate_listening_witness(x, y, SR, "phase_x")
    assert res.pitch_drift_cents > 15.0
    assert "pitch_instability" in res.findings


def test_pitch_modulation_detected() -> None:
    """Neu eingeführtes 5-Hz-Vibrato ⇒ pitch_modulation."""
    x = _tone_vibrato(f0=220.0)
    y = _tone_vibrato(f0=220.0, vibrato_hz=5.0, depth_cents=60.0)
    res = evaluate_listening_witness(x, y, SR, "phase_x")
    assert "pitch_modulation" in res.findings or "pitch_instability" in res.findings


def test_loudness_pumping_detected() -> None:
    """2-Hz-Amplitudenmodulation nach der Phase ⇒ loudness_pumping."""
    t = np.arange(N) / SR
    x = _tone_vibrato(f0=220.0)
    y = x * (1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t))
    res = evaluate_listening_witness(x, y, SR, "phase_x")
    assert res.loud_mod_rise_db > 0.5
    assert "loudness_pumping" in res.findings


def test_vocal_distortion_detected() -> None:
    """Hartes Clipping nach der Phase ⇒ Flat-Top-Anstieg ⇒ vocal_distortion."""
    x = _tone_vibrato(f0=220.0)
    y = np.clip(x * 6.0, -1.0, 1.0).astype(np.float32)
    res = evaluate_listening_witness(x, y, SR, "phase_x")
    assert res.flat_top_rise > 0.01
    assert "vocal_distortion" in res.findings


def test_silence_safe() -> None:
    x = np.zeros(N, dtype=np.float32)
    res = evaluate_listening_witness(x, x.copy(), SR, "phase_x")
    assert res.findings == []


def test_runtime_budget() -> None:
    """≤ 5 s pro 30-s-Paar (Ziel ≤ 2 s für 30 s)."""
    rng = np.random.RandomState(0)
    x = (rng.randn(N) * 0.1).astype(np.float32)
    y = (rng.randn(N) * 0.1).astype(np.float32)
    t0 = time.perf_counter()
    evaluate_listening_witness(x, y, SR, "phase_x")
    dt = time.perf_counter() - t0
    assert dt < 5.0, f"Witness zu langsam: {dt:.1f}s"


# ─── Echte Musik (Testkünstlerin (Schlager), Projekt-Testaudio) ──────────────────────────────


def _load_real(path: Path) -> tuple[np.ndarray, int]:
    import librosa

    data, sr = librosa.load(str(path), sr=None, mono=True)
    return data.astype(np.float32), int(sr)


_REAL_30 = Path(__file__).resolve().parent.parent.parent / "test_audio" / "Testkünstlerin (Schlager) - 30 Sekunden.mp3"
_REAL_224 = (
    Path(__file__).resolve().parent.parent.parent
    / "test_audio"
    / "Testkünstlerin (Schlager) - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"
)


@pytest.mark.skipif(not _REAL_30.exists(), reason="Test-Track-Testaudio (30 s) fehlt")
def test_real_music_no_findings_on_identity() -> None:
    """Echte Musik, identisches Audio ⇒ keine Befunde (keine False-Positives)."""
    x, sr = _load_real(_REAL_30)
    res = evaluate_listening_witness(x, x.copy(), sr, "phase_identity")
    assert res.findings == []


@pytest.mark.skipif(not _REAL_30.exists(), reason="Test-Track-Testaudio (30 s) fehlt")
def test_real_music_deterministic() -> None:
    x, sr = _load_real(_REAL_30)
    y = np.roll(x, 137)
    r1 = evaluate_listening_witness(x, y, sr, "phase_x")
    r2 = evaluate_listening_witness(x, y, sr, "phase_x")
    assert r1.as_dict() == r2.as_dict()


@pytest.mark.skipif(not _REAL_30.exists(), reason="Test-Track-Testaudio (30 s) fehlt")
def test_real_music_pitch_shift_detected() -> None:
    """Echte Musik +100 Cent (Längen-erhaltend) ⇒ pitch_instability."""
    from scipy.signal import resample_poly

    x, sr = _load_real(_REAL_30)
    ratio = 2.0 ** (100.0 / 1200.0)
    # Resample um ratio, dann auf Originallänge stutzen → +100 Cent bei gleicher Länge.
    shifted = resample_poly(x.astype(np.float64), int(round(ratio * 1000)), 1000).astype(np.float32)
    y = shifted[: len(x)]
    res = evaluate_listening_witness(x, y, sr, "phase_pitch")
    assert res.pitch_drift_cents > 15.0
    assert "pitch_instability" in res.findings


@pytest.mark.skipif(not _REAL_30.exists(), reason="Test-Track-Testaudio (30 s) fehlt")
def test_real_music_pumping_detected() -> None:
    """Echte Musik mit 2-Hz-AM nach der Phase ⇒ loudness_pumping."""
    x, sr = _load_real(_REAL_30)
    t = np.arange(len(x)) / sr
    y = (x * (1.0 + 0.4 * np.sin(2 * np.pi * 2.0 * t))).astype(np.float32)
    res = evaluate_listening_witness(x, y, sr, "phase_pump")
    assert res.loud_mod_rise_db > 1.0
    assert "loudness_pumping" in res.findings


@pytest.mark.skipif(not _REAL_224.exists(), reason="Test-Track-Testaudio (224 s) fehlt")
def test_real_music_224s_runtime() -> None:
    """224-s-Song: Witness bleibt im Budget (FFT-Autokorrelation, §V08/§10a)."""
    x, sr = _load_real(_REAL_224)
    t0 = time.perf_counter()
    res = evaluate_listening_witness(x, x.copy(), sr, "phase_identity")
    dt = time.perf_counter() - t0
    assert res.findings == []
    assert dt < 30.0, f"Witness auf 224 s zu langsam: {dt:.1f}s"


@pytest.mark.skipif(not _REAL_30.exists(), reason="Test-Track-Testaudio (30 s) fehlt")
def test_silence_pad_no_false_positive() -> None:
    """§Witness-Fix 2026-09-11: 100-ms-Silence-Pad am Anfang darf kein
    loudness_pumping auslösen — der Silence→Musik-Schritt am Rand war die
    Leakage-Quelle der 37-dB-False-Positives (Produktionsbefund phase_04)."""
    x, sr = _load_real(_REAL_30)
    pad = np.zeros((sr // 10, x.shape[1]), dtype=np.float32) if x.ndim == 2 else np.zeros(sr // 10, dtype=np.float32)
    y = np.concatenate([pad, x], axis=0).astype(np.float32)
    res = evaluate_listening_witness(x, y[: len(x)], sr, "phase_pad")
    assert res.loud_mod_rise_db < 1.0, f"Rand-Leakage nicht unterdrückt: {res.loud_mod_rise_db:.2f} dB"
    assert "loudness_pumping" not in res.findings


def test_phrase_gap_robustness() -> None:
    """§Witness-Fix 2026-09-11: Phrasen-Lücken im F0-Verlauf dürfen keine
    Pitch-Modulation vortäuschen (Konkatenations-Sprünge → 3–8-Hz-Leakage).
    Gleiche Phrasen mit verschobener Lücke ⇒ pitch_mod_depth bleibt klein."""
    sr = 48000
    t = np.arange(sr * 4) / sr

    def _voiced_phrases(gap_at: int) -> np.ndarray:
        seg1 = (0.4 * np.sin(2 * np.pi * 220 * t[:sr])).astype(np.float32)
        gap = np.zeros(gap_at, dtype=np.float32)
        seg2 = (0.4 * np.sin(2 * np.pi * 220 * t[:sr])).astype(np.float32)
        return np.concatenate([seg1, gap, seg2])

    x = _voiced_phrases(int(sr * 0.35))
    y = _voiced_phrases(int(sr * 0.55))
    res = evaluate_listening_witness(x, y, sr, "phrase_gap")
    assert res.pitch_mod_depth_cents < 15.0, f"Phrasen-Lücken-Leakage: {res.pitch_mod_depth_cents:.1f} Cent"


def test_bass_loss_detected() -> None:
    """Hochpass-gefiltertes Signal (Bass weg) → bass_loss-Finding."""
    sr = 48000
    rng = np.random.default_rng(11)
    t = np.arange(sr * 4) / sr
    x = np.clip(
        0.4 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * rng.normal(0, 1, len(t)), -1, 1
    )
    x = x.astype(np.float32)
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 400 / (sr / 2), btype="high", output="sos")
    y = sosfiltfilt(sos, x).astype(np.float32)
    res = evaluate_listening_witness(x, y, sr, "bass_test")
    assert res.bass_drop_db > 1.5, f"Bass-Verlust nicht erkannt: {res.bass_drop_db:.2f} dB"
    assert "bass_loss" in res.findings


def test_transient_smearing_detected() -> None:
    """Glattgebügeltes Signal (Transienten weg) → transient_smearing-Finding."""
    sr = 48000
    rng = np.random.default_rng(13)
    t = np.arange(sr * 4) / sr
    x = (0.4 * np.sin(2 * np.pi * 220 * t) + 0.25 * rng.normal(0, 1, len(t))).astype(np.float32)
    for k in range(4, len(t), sr // 2):
        x[k : k + 8] += 0.9
    x = np.clip(x, -1, 1).astype(np.float32)
    from scipy.ndimage import median_filter

    y = x * (median_filter(np.abs(x) + 1e-4, size=961) / (np.abs(x) + 1e-4)).astype(np.float32)
    y = np.clip(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), -1, 1).astype(np.float32)
    res = evaluate_listening_witness(x, y, sr, "transient_test")
    assert res.transient_smear_ratio > 0.35, f"Verschmierung nicht erkannt: {res.transient_smear_ratio:.3f}"
    assert "transient_smearing" in res.findings


def test_vocal_muffled_detection():
    """§Residual-Defekt-Zeuge (2026-09-13): Dämpfung des Klarheitsbands
    (2–6 kHz) ohne Defekt-Reduktion → vocal_muffled."""
    from scipy.signal import butter, filtfilt

    sr = 48000
    rng = np.random.default_rng(21)
    t = np.arange(sr * 4) / sr
    x = (0.35 * np.sin(2 * np.pi * 220 * t) + 0.15 * rng.standard_normal(len(t))).astype(np.float32)
    x = x + (0.1 * np.sin(2 * np.pi * 3200 * t)).astype(np.float32)
    b, a = butter(2, [2000 / (sr / 2), 6000 / (sr / 2)], btype="bandstop")
    y = filtfilt(b, a, x.astype(np.float64)).astype(np.float32)
    res = evaluate_listening_witness(x, y, sr, "phase_04_eq_correction")
    assert "vocal_muffled" in res.findings
    assert res.vocal_muffled_db > 2.5


def test_vocal_distorted_residual_for_defect_owner():
    """§Residual-Defekt-Zeuge: Rest-Clipping nach der zuständigen Phase → Warnung."""
    sr = 48000
    t = np.arange(sr * 2) / sr
    x = (0.35 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y = np.clip(x * 3.0, -1.0, 1.0).astype(np.float32)  # stark geclippt
    res = evaluate_listening_witness(x, y, sr, "phase_09_crackle_removal")
    assert "vocal_distorted_residual" in res.findings


def test_no_residual_warning_when_defect_healed():
    """§Residual-Defekt-Zeuge: Defekt behoben → KEINE Rest-Warnung."""
    sr = 48000
    t = np.arange(sr * 2) / sr
    x_clean = (0.35 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    clipped = np.clip(x_clean * 3.0, -1.0, 1.0).astype(np.float32)
    res = evaluate_listening_witness(clipped, x_clean, sr, "phase_09_crackle_removal")
    assert "vocal_distorted_residual" not in res.findings


def test_non_owner_phase_no_residual_warning():
    """Nur die zuständige Phase warnt über Rest-Defekte."""
    sr = 48000
    t = np.arange(sr * 2) / sr
    x = (0.35 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y = np.clip(x * 3.0, -1.0, 1.0).astype(np.float32)
    res = evaluate_listening_witness(x, y, sr, "phase_12_wow_flutter_fix")
    assert "vocal_distorted_residual" not in res.findings


def _faded_carrier_am(am_depth: float) -> np.ndarray:
    """Reiner Träger mit kurzen Fades (gegen Hilbert-Kantenartefakte) und
    steuerbarer 30-Hz-AM-Tiefe im Rauigkeitsband."""
    t = np.arange(N) / SR
    carrier = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    fade = int(0.05 * SR)
    w = np.ones(N)
    w[:fade] = np.hanning(2 * fade)[:fade]
    w[-fade:] = np.hanning(2 * fade)[fade:]
    carrier = carrier * w
    am = 1.0 + am_depth * np.sin(2 * np.pi * 30.0 * t)
    x = carrier * am
    return (x / np.max(np.abs(x))).astype(np.float32)


def test_roughness_rise_below_relative_jnd_is_clamped() -> None:
    """SUP-F6 (2026-09-16): Der Rauigkeits-Schätzer ist eine relative Skala
    (~10⁴ auf realer Musik) — kleine, unhörbare Hüllkurven-Änderungen müssen
    KEIN roughness_increase mehr melden (JND-Gate: < 35 % relativer Anstieg
    wird auf 0 geklemmt; Produktionsbefund: +1,16 bei harmlosem 30-Hz-Hochpass
    auf dem Test-Track-Export). 12,5 % AM-Tiefen-Zunahme liegt unter dem
    Vassilakis-JND (~17 %) → kein Befund."""
    x = _faded_carrier_am(0.4)
    y = _faded_carrier_am(0.45)  # +12,5 % relative Tiefe — unter JND
    res = evaluate_listening_witness(x, y, SR, "phase_x")
    assert "roughness_increase" not in res.findings
    assert res.roughness_rise_asper == 0.0


def test_roughness_rise_above_relative_jnd_still_fires() -> None:
    """SUP-F6-Gegenprobe: Ein echter Rauigkeits-Regress (30-Hz-AM-Tiefe von
    0 auf 80 %) muss weiterhin gemeldet werden — das JND-Gate darf
    True-Positive nicht schlucken."""
    x = _faded_carrier_am(0.0)
    y = _faded_carrier_am(0.8)  # hörbare Rauigkeit: volle Modulationstiefe
    res = evaluate_listening_witness(x, y, SR, "phase_x")
    assert "roughness_increase" in res.findings
    assert res.roughness_rise_asper > 0.0


class TestPerfR11BatchedPaths:
    """§PERF-R11 (2026-09-19): Batched Witness-Pfade — numerisch äquivalent
    zum alten Frame-Loop (f0/voiced bit-identisch, Derivate ≤ 1e-4),
    deterministisch nach §G5 (GEBOTE.md), Spektrum-Cache gedeckelt."""

    @staticmethod
    def _carrier(dur: float, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        t = np.arange(int(SR * dur)) / SR
        return (
            0.4 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t) + 0.02 * rng.randn(len(t))
        ).astype(np.float32)

    def test_frame_f0_hnr_batched_equivalent(self) -> None:
        import backend.core.listening_witness as lw

        x = self._carrier(5.0, 3)
        lo = lw._frame_f0_hnr_loop(x, SR)
        ba = lw._frame_f0_hnr_batched(x, SR)
        assert np.array_equal(lo[0], ba[0])  # f0 bit-identisch
        assert np.array_equal(lo[1], ba[1])  # voiced bit-identisch
        assert np.abs(lo[2] - ba[2]).max() < 1e-4  # hnr (Float-Rauschen)
        assert np.abs(lo[3] - ba[3]).max() < 1e-4  # hf_flatness

    def test_transient_sharpness_rms_equivalent(self) -> None:
        import backend.core.listening_witness as lw

        x = self._carrier(3.0, 5)
        _win = max(64, int(lw._TRANSIENT_WIN * SR))
        _hop = _win // 2
        rms_old = np.asarray(
            [np.sqrt(np.mean(x[i : i + _win] ** 2) + 1e-12) for i in range(0, len(x) - _win + 1, _hop)],
            dtype=np.float64,
        )
        _sw = np.lib.stride_tricks.sliding_window_view(x, _win)
        rms_new = np.sqrt(np.mean(_sw[::_hop] ** 2, axis=1) + 1e-12)
        assert np.abs(rms_old - rms_new).max() < 1e-6

    def test_witness_end_to_end_identical_fields(self) -> None:
        import backend.core.listening_witness as lw

        x = self._carrier(4.0, 7)
        y = (x * 0.999 + 1e-6).astype(np.float32)
        _orig = lw._frame_f0_hnr
        try:
            lw._frame_f0_hnr = lw._frame_f0_hnr_loop
            r_old = evaluate_listening_witness(x, y, SR, "phase_x")
        finally:
            lw._frame_f0_hnr = _orig
        r_new = evaluate_listening_witness(x, y, SR, "phase_x")
        d_old, d_new = r_old.as_dict(), r_new.as_dict()
        for k in d_old:
            if k == "findings":
                continue
            assert d_old[k] == d_new[k], f"Feld {k}: {d_old[k]} != {d_new[k]}"

    def test_band_spec_cache_bounded_and_deterministic(self) -> None:
        import backend.core.listening_witness as lw

        lw._BAND_SPEC_CACHE.clear()
        x = self._carrier(2.0, 11)
        v1 = lw._band_energy_ratio_db(x, SR, 20.0, 250.0)
        v2 = lw._band_energy_ratio_db(x, SR, 20.0, 250.0)
        assert v1 == v2
        for i in range(8):
            lw._band_energy_ratio_db((x * (1.0 + i * 1e-4)).astype(np.float32), SR, 20.0, 250.0)
        assert len(lw._BAND_SPEC_CACHE) <= lw._BAND_SPEC_CACHE_MAX
        assert len(lw._BAND_SPEC_CACHE) > 0

    def test_pre_echo_rolling_percentile_equivalent(self) -> None:
        from backend.core.dsp import pre_echo_model as pm

        rng = np.random.RandomState(13)
        env = rng.rand(400).astype(np.float64) + 0.01
        _loc_win = max(4, int(pm._LOCAL_WIN_S / pm._ENV_WIN_S))
        _floor = float(np.max(env)) * 10.0 ** (-60.0 / 20.0)
        old = np.zeros(400, dtype=np.float64)
        for f in range(400):
            _lo = max(0, f - _loc_win)
            _hi = min(400, f + _loc_win + 1)
            old[f] = max(float(np.percentile(env[_lo:_hi], pm._LOCAL_PERC)), _floor)
        new = np.zeros(400, dtype=np.float64)
        _ilo, _ihi = _loc_win, 400 - _loc_win
        if _ihi > _ilo:
            _w_len = 2 * _loc_win + 1
            _w = np.lib.stride_tricks.sliding_window_view(env, _w_len)
            new[_ilo:_ihi] = np.percentile(_w[: _ihi - _ilo], pm._LOCAL_PERC, axis=1)
        for f in list(range(0, _ilo)) + list(range(_ihi, 400)):
            _lo = max(0, f - _loc_win)
            _hi = min(400, f + _loc_win + 1)
            new[f] = max(float(np.percentile(env[_lo:_hi], pm._LOCAL_PERC)), _floor)
        np.maximum(new, _floor, out=new)
        assert np.array_equal(old, new)

    def test_pre_echo_masked_level_no_false_positive(self):
        """SUP-F7: Hinzugefügte Vor-Fenster-Energie bei −15 dB unter dem Onset
        (kein Delta am Onset selbst) darf KEINEN Befund erzeugen. Die alte
        Ratio pre-Δ ÷ post-Δ explodierte hier (post-Δ ≈ 0) — Produktionsbefund
        „Vogel der Nacht": 12 Pre-Echo-Befunde bei loud=0.0dB/hf=0.000."""
        from backend.core.dsp.pre_echo_model import pre_echo_ratio_db

        rng = np.random.RandomState(11)
        n = SR * 3
        x = 0.05 * rng.standard_normal(n)
        onset = SR * 3 // 2
        x[onset : onset + int(0.05 * SR)] *= 8.0  # kräftiger Onset (Attack)
        y = x.copy()
        # Hinzugefügte Energie NUR im Vor-Fenster, −15 dB unter Onset-Pegel
        _target_e = (0.05 * 8.0) ** 2 * 10.0 ** (-15.0 / 10.0)
        y[onset - 6000 : onset - 3000] += np.sqrt(_target_e) * rng.standard_normal(3000)
        db = pre_echo_ratio_db(x, y, SR)
        assert db < -12.0, f"−15 dB unter Onset ist maskiert, kein Befund: {db} dB"

    def test_pre_echo_strong_level_still_detected(self):
        """SUP-F7: −6 dB unter dem Onset (nahe am Pegel) bleibt ein Befund."""
        from backend.core.dsp.pre_echo_model import pre_echo_ratio_db

        rng = np.random.RandomState(12)
        n = SR * 3
        x = 0.05 * rng.standard_normal(n)
        onset = SR * 3 // 2
        x[onset : onset + int(0.05 * SR)] *= 8.0
        y = x.copy()
        _target_e = (0.05 * 8.0) ** 2 * 10.0 ** (-6.0 / 10.0)
        y[onset - 6000 : onset - 3000] += np.sqrt(_target_e) * rng.standard_normal(3000)
        db = pre_echo_ratio_db(x, y, SR)
        assert db >= -12.0, f"−6 dB unter Onset muss als Pre-Echo sichtbar bleiben: {db} dB"


# ---------------------------------------------------------------------------
# Audio-Bundle-Zwischenspeicher: exakte Werte, Ketten-Bit-Identität, Reset
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWitnessBundleCache:
    """Der Bundle-Cache liefert exakt die direkten Metrikwerte (bit-identisch)
    und wird per reset_witness_bundle_cache() geleert (§V8 (copilot-instructions.md))."""

    def test_bundle_values_match_direct_functions(self):
        from backend.core.listening_witness import (
            _AIR_HI_HZ,
            _AIR_LO_HZ,
            _BASS_HI_HZ,
            _BASS_LO_HZ,
            _CLARITY_HI_HZ,
            _CLARITY_LO_HZ,
            _band_energy_ratio_db,
            _frame_f0_hnr,
            _loudness_mod_depth_db,
            _transient_sharpness,
            _witness_audio_bundle,
            reset_witness_bundle_cache,
        )

        reset_witness_bundle_cache()
        x = _tone_vibrato(f0=220.0, vibrato_hz=5.0, depth_cents=30.0)
        f0, voiced, hnr, flat = _frame_f0_hnr(x, SR)
        bun = _witness_audio_bundle(x, SR)
        assert np.array_equal(bun["f0"], f0)
        assert np.array_equal(bun["voiced"], voiced)
        assert np.array_equal(bun["hnr"], hnr)
        assert np.array_equal(bun["flat"], flat)
        assert bun["loud_mod"] == _loudness_mod_depth_db(x, SR)
        assert bun["flat_top"] == float(np.mean(np.abs(x) > 0.98))
        assert bun["bass"] == _band_energy_ratio_db(x, SR, _BASS_LO_HZ, _BASS_HI_HZ)
        assert bun["air"] == _band_energy_ratio_db(x, SR, _AIR_LO_HZ, _AIR_HI_HZ)
        assert bun["clarity"] == _band_energy_ratio_db(x, SR, _CLARITY_LO_HZ, _CLARITY_HI_HZ)
        assert bun["tr"] == _transient_sharpness(x, SR)

    def test_chain_reuse_bit_identical(self):
        """before(k+1) == Kopie von after(k) → warme Kette liefert identisches Ergebnis."""
        from backend.core.listening_witness import (
            _WITNESS_BUNDLE_CACHE,
            evaluate_listening_witness,
            reset_witness_bundle_cache,
        )

        a = _tone_vibrato(f0=220.0, vibrato_hz=5.0, depth_cents=30.0)
        b = (a * 1.01).astype(np.float32)
        c = (b * 1.01).astype(np.float32)

        reset_witness_bundle_cache()
        r_cold = evaluate_listening_witness(b.copy(), c, SR, "phase_03_denoise")
        reset_witness_bundle_cache()
        evaluate_listening_witness(a, b, SR, "phase_03_denoise")  # füllt Bundle für b
        r_warm = evaluate_listening_witness(b.copy(), c, SR, "phase_03_denoise")

        assert r_cold.as_dict() == r_warm.as_dict()
        assert len(_WITNESS_BUNDLE_CACHE) >= 2

    def test_reset_clears_cache(self):
        from backend.core.listening_witness import (
            _WITNESS_BUNDLE_CACHE,
            _witness_audio_bundle,
            reset_witness_bundle_cache,
        )

        reset_witness_bundle_cache()
        _witness_audio_bundle(_tone_vibrato(), SR)
        assert len(_WITNESS_BUNDLE_CACHE) == 1
        reset_witness_bundle_cache()
        assert len(_WITNESS_BUNDLE_CACHE) == 0
