"""§SOTA-PSY-A4 (Equal-Loudness-Rollout, ISO 226) — Regressionsschutz.

Deckt den PSY-A4-Ausbau (2026-09-15):
- ``equal_loudness_strength_factor()``-Helfer (Kontur-SPLs; Bugfix: die
  Vorgänger-Verdrahtung in phase_37 nutzte die Korrekturkurve mit negativen
  dB-Werten und degenerierte über den ``max(..., 1e-9)``-Guard immer auf das
  0,5-Floor — der Faktor war damit NICHT frequenzsensitiv).
- phase_37: Temperierung über den Helfer (Bass-Mix in Phon-Hörbarkeit).
- phase_16: Band-Gain-Temperierung + ``equal_loudness_factors``-Metadaten.
- phase_04: EQ-Kurven-Temperierung + ``equal_loudness_factors``-Metadaten.
- phase_17: Mastering-EQ-Band-Gain-Temperierung (pipeline_metrics).
- phase_38/39: Faktor-Metadaten, nie über Design-Pegel (Faktor ≤ 1).

Autor: Aurik Testing Team
"""

import numpy as np
import pytest

from backend.core.fletcher_munson_curves import (
    FletcherMunsonProcessor,
    equal_loudness_strength_factor,
)

# ─── Helfer ───────────────────────────────────────────────────────────────────


def _contour_factor(freq_hz: float, target_phon: int = 60) -> float:
    """Referenz-Rechnung direkt aus den Kontur-SPLs (Test-Orakel)."""
    proc = FletcherMunsonProcessor()
    contour = proc.get_contour(target_phon)
    spl_f = float(np.asarray(contour.get_spl_at_frequency(np.array([float(freq_hz)])))[0])
    spl_ref = float(np.asarray(contour.get_spl_at_frequency(np.array([1000.0])))[0])
    denom = spl_f if abs(spl_f) >= 1e-9 else 1.0
    return float(np.clip(spl_ref / denom, 0.5, 1.0))


def _factor_one(freqs, target_phon=60, reference_freq_hz=1000.0, min_factor=0.5):
    """Monkeypatch-Ersatz: keine Temperierung (Faktor 1.0)."""
    arr = np.asarray(freqs)
    if arr.ndim == 0:
        return 1.0
    return np.ones(arr.shape, dtype=np.float64)


def _make_audio(n: int, seed: int, sr: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, n).astype(np.float32)


# ─── Helfer-Semantik ──────────────────────────────────────────────────────────


class TestEqualLoudnessStrengthFactor:
    def test_bass_tempered_below_reference(self):
        f80 = equal_loudness_strength_factor(80.0)
        assert 0.5 <= f80 < 1.0
        assert f80 == pytest.approx(_contour_factor(80.0), abs=1e-6)

    def test_reference_frequency_is_neutral(self):
        assert equal_loudness_strength_factor(1000.0) == pytest.approx(1.0)

    def test_more_sensitive_band_never_above_design(self):
        # Presence (3 kHz) ist empfindlicher als 1 kHz → Deckel bei 1.0
        # („nie über Design-Pegel", dokumentierter Konservativ-Deckel).
        assert equal_loudness_strength_factor(3000.0) == pytest.approx(1.0)

    def test_floor_at_deep_bass(self):
        # 20 Hz: rohes Verhältnis < 0.5 → konservatives Floor 0.5.
        assert equal_loudness_strength_factor(20.0) == pytest.approx(0.5)

    def test_vectorized_matches_scalar(self):
        freqs = np.array([80.0, 250.0, 1000.0, 12000.0])
        vec = equal_loudness_strength_factor(freqs)
        assert vec.shape == (4,)
        for f, v in zip(freqs, vec):
            assert v == pytest.approx(equal_loudness_strength_factor(float(f)), abs=1e-12)

    def test_garbage_input_is_neutral(self):
        # §0a NaN-Schutz: Müll-Daten dürfen nie zu einem Faktor > 1 oder < 0.5 werden.
        assert equal_loudness_strength_factor(float("nan")) == 1.0
        assert equal_loudness_strength_factor(0.0) == 1.0
        assert equal_loudness_strength_factor(-5.0) == 1.0

    def test_monotonic_between_bass_and_reference(self):
        vals = [equal_loudness_strength_factor(f) for f in (40.0, 80.0, 200.0, 500.0, 1000.0)]
        assert vals == sorted(vals)


# ─── phase_37: Bass-Mix-Temperierung ──────────────────────────────────────────


class TestPhase37EqualLoudness:
    @pytest.fixture
    def bass_audio(self) -> np.ndarray:
        sr = 48000
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
        return (0.5 * np.sin(2 * np.pi * 80 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_tempered_mix_weaker_than_untempered(self, bass_audio, monkeypatch):
        import backend.core.fletcher_munson_curves as fm
        from backend.core.phases.phase_37_bass_enhancement import BassEnhancement

        res_real = BassEnhancement().process(bass_audio, sample_rate=48000, material_type="vinyl")
        delta_real = float(np.sqrt(np.mean((res_real.audio - bass_audio) ** 2)))
        assert delta_real > 0.0

        monkeypatch.setattr(fm, "equal_loudness_strength_factor", _factor_one)
        res_one = BassEnhancement().process(bass_audio, sample_rate=48000, material_type="vinyl")
        delta_one = float(np.sqrt(np.mean((res_one.audio - bass_audio) ** 2)))

        # Temperierung (≈0,556 bei 80 Hz) reduziert den Eingriff, nie umgekehrt.
        assert delta_one > 0.0
        assert delta_real < delta_one

    def test_faktor_ist_frequenzsensitiv(self):
        # Bugfix-Nachweis: der Faktor folgt den Kontur-SPLs und ist NICHT das
        # konstante 0,5-Floor der Vorgänger-Verdrahtung.
        f80 = equal_loudness_strength_factor(80.0)
        assert f80 == pytest.approx(_contour_factor(80.0), abs=1e-6)
        assert 0.5 < f80 < 1.0


# ─── phase_16: Band-Gain-Temperierung ─────────────────────────────────────────


class TestPhase16EqualLoudness:
    def test_band_factors_recorded_and_bounded(self):
        from backend.core.phases.phase_16_final_eq import FinalEQ

        audio = _make_audio(48000, 7, 48000)
        res = FinalEQ().process(audio, sample_rate=48000, material_type="vinyl")
        factors = res.metadata.get("equal_loudness_factors", {})
        assert set(factors.keys()) == {"low", "low_mid", "high_mid", "high"}
        assert 0.5 <= factors["low"] < 1.0  # 60 Hz: Bass wird temperiert
        assert 0.5 <= factors["low_mid"] < 1.0  # 250 Hz
        assert factors["high_mid"] == pytest.approx(1.0)  # 4 kHz: empfindlicher → Deckel
        for f in factors.values():
            assert 0.5 - 1e-9 <= f <= 1.0 + 1e-9

    def test_bass_gain_reduced_vs_untempered(self, monkeypatch):
        import backend.core.fletcher_munson_curves as fm
        from backend.core.phases.phase_16_final_eq import FinalEQ

        audio = _make_audio(48000, 11, 48000)
        res_real = FinalEQ().process(audio, sample_rate=48000, material_type="vinyl")
        f_low_real = res_real.metadata["equal_loudness_factors"]["low"]
        assert f_low_real < 1.0

        monkeypatch.setattr(fm, "equal_loudness_strength_factor", _factor_one)
        res_one = FinalEQ().process(audio, sample_rate=48000, material_type="vinyl")
        f_low_one = res_one.metadata["equal_loudness_factors"]["low"]
        assert f_low_one == pytest.approx(1.0)

        # Temperierter Bass-Eingriff ist schwächer als untemperierter.
        delta_real = float(np.sqrt(np.mean((res_real.audio - audio) ** 2)))
        delta_one = float(np.sqrt(np.mean((res_one.audio - audio) ** 2)))
        assert delta_real < delta_one

    def test_zero_strength_passthrough(self):
        from backend.core.phases.phase_16_final_eq import FinalEQ

        audio = _make_audio(48000, 13, 48000)
        res = FinalEQ().process(audio, sample_rate=48000, material_type="vinyl", strength=0.0)
        assert np.allclose(res.audio, audio, atol=1e-6)


# ─── phase_04: EQ-Kurven-Temperierung ─────────────────────────────────────────


class TestPhase04EqualLoudness:
    def test_curve_factors_recorded_and_bounded(self):
        from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase

        audio = _make_audio(48000, 17, 48000)
        res = EQCorrectionPhase().process(audio, sample_rate=48000, material_type="vinyl")
        factors = res.modifications.get("equal_loudness_factors", {})
        assert factors, "equal_loudness_factors fehlen in modifications"
        for f in factors.values():
            assert 0.5 - 1e-9 <= f <= 1.0 + 1e-9
        # Bass-Punkte der Korrekturkurve müssen temperiert sein (< 1.0).
        assert any(f < 1.0 for freq, f in factors.items() if float(freq) < 300.0)

    def test_zero_strength_passthrough(self):
        from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase

        audio = _make_audio(48000, 19, 48000)
        res = EQCorrectionPhase().process(audio, sample_rate=48000, material_type="vinyl", strength=0.0)
        assert np.allclose(res.audio, audio, atol=1e-6)


# ─── phase_17: Mastering-EQ-Temperierung ──────────────────────────────────────


class TestPhase17EqualLoudness:
    def test_bass_band_gain_tempered(self, monkeypatch):
        import backend.core.fletcher_munson_curves as fm
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_17_mastering_polish import MasteringPolishPhase

        sr = 48000
        n = int(sr * 0.5)
        rng = np.random.default_rng(23)
        audio = rng.normal(0, 0.1, (n, 2)).astype(np.float32)

        res_real = MasteringPolishPhase().process(audio, sr, MaterialType.VINYL, strength=1.0)
        eq_real = res_real.metadata.get("pipeline_metrics", {}).get("eq", {})
        gains_real = eq_real.get("band_gains_db", {})
        assert gains_real, "band_gains_db fehlen (Temperierung nicht greifbar)"

        monkeypatch.setattr(fm, "equal_loudness_strength_factor", _factor_one)
        res_one = MasteringPolishPhase().process(audio, sr, MaterialType.VINYL, strength=1.0)
        eq_one = res_one.metadata.get("pipeline_metrics", {}).get("eq", {})
        gains_one = eq_one.get("band_gains_db", {})

        assert set(gains_real.keys()) == set(gains_one.keys())
        for band in gains_real:
            g_real, g_one = abs(float(gains_real[band])), abs(float(gains_one[band]))
            # Temperierung reduziert Bass-/Luftband-Gains, niemals umgekehrt.
            assert g_real <= g_one + 1e-9, f"Band {band}: {g_real} > {g_one}"
        # Mindestens ein Band ist real temperiert (Faktor < 1.0).
        assert any(abs(float(gains_real[b])) < abs(float(gains_one[b])) - 1e-9 for b in gains_real)


# ─── phase_38/39: Faktor-Metadaten ────────────────────────────────────────────


class TestPhase38EqualLoudness:
    def test_factors_recorded_never_above_design(self):
        from backend.core.phases.phase_38_presence_boost import PresenceBoost

        audio = _make_audio(48000, 29, 48000)
        res = PresenceBoost().process(audio, sample_rate=48000, material_type="vinyl")
        factors = res.metadata.get("equal_loudness_factors", {})
        assert set(factors.keys()) == {"lower", "upper"}
        # Presence-Band (2,75/4,75 kHz) ist empfindlicher als 1 kHz →
        # Deckel [0,5, 1,0] ⇒ 1,0 (nie über Design-Pegel, aber gemessen).
        assert factors["lower"] == pytest.approx(1.0)
        assert factors["upper"] == pytest.approx(1.0)


class TestPhase39EqualLoudness:
    def test_air_shelf_factor_recorded_and_tempered(self):
        from backend.core.phases.phase_39_air_band_enhancement import AirBandEnhancement

        audio = _make_audio(48000, 31, 48000)
        res = AirBandEnhancement().process(audio, sample_rate=48000, material_type="cd_digital")
        factor = float(res.metadata.get("equal_loudness_factor", 1.0))
        assert 0.5 - 1e-9 <= factor <= 1.0 + 1e-9
        # Luftband-Shelf (~12 kHz) liegt oberhalb des Empfindlichkeits-Maximums
        # → Faktor < 1.0 (Temperierung aktiv).
        assert factor < 1.0
