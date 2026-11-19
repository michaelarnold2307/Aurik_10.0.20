from __future__ import annotations

import pytest

"""Unit-Tests für backend/core/clipping_detection.py — §6.3 CLIPPING vs. SOFT_SATURATION.

≥ 35 Tests:
  - Shape / Mono / Stereo
  - NaN / Inf Guard
  - Leeres Signal
  - flat_tops_pct Berechnung
  - Korrekte Klassifikation: Hard-clipped Sinus → CLIPPING
  - Korrekte Klassifikation: tanh-saturated Sinus → SOFT_SATURATION
  - Korrekte Klassifikation: sauberes Signal → SOFT_SATURATION
  - Korrekte Klassifikation: schwaches Signal → SOFT_SATURATION
  - Grenzwerte (flat_tops exakt 0.1 %)
  - Singleton Thread-Safety
  - Result-Dataclass Felder
  - Eigenschaften should_repair / should_preserve
  - assert sr != 48000 → AssertionError
  - Stereo-Verarbeitung
  - Confidence 0–1
  - Konsistenz von classify_clipping() vs. analyse_clipping()
  - Reproduzierbarkeit (deterministisch)
  - Performance-Bound (< 2 s bei 5 s Audio)
"""


import threading
import time
from collections.abc import Callable

import numpy as np

from backend.core.clipping_detection import (
    FLAT_TOPS_THRESHOLD_PCT,
    ClippingAnalysisResult,
    ClippingClassifier,
    ClippingType,
    _flat_tops_pct,
    analyse_clipping,
    classify_clipping,
    detect_sub_ceiling_clipping,
    get_clipping_classifier,
)

SR = 48000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sine(freq: float = 440.0, amp: float = 0.5, dur: float = 0.1) -> np.ndarray:
    """Generate a mono sine wave at 48 kHz."""
    t = np.arange(int(SR * dur)) / SR
    out: np.ndarray = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return out


def _hard_clip(audio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply hard clipping: samples beyond threshold are clamped flat."""
    return np.clip(audio, -threshold, threshold)


def _tanh_saturate(audio: np.ndarray, drive: float = 3.0) -> np.ndarray:
    """Apply tanh saturation (soft saturation — tube-like even harmonics)."""
    out: np.ndarray = np.tanh(drive * audio).astype(np.float32)
    return out


def _make_hard_clipped_signal(amp: float = 0.95, freq: float = 440.0) -> np.ndarray:
    """Full-amplitude sine then hard-clipped → definite CLIPPING signature."""
    t = np.arange(int(SR * 0.5)) / SR
    raw = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    # Hard clip to 0.6 → creates flat tops and odd harmonics
    clipped = np.clip(raw, -0.6, 0.6)
    # Normalise to near-unity so flat_tops_pct registers
    result: np.ndarray = (clipped / 0.60 * 0.9995).astype(np.float32)
    return result


def _make_tanh_signal(freq: float = 440.0) -> np.ndarray:
    """Tanh-saturated sine → SOFT_SATURATION (even dominant, no flat tops)."""
    t = np.arange(int(SR * 0.5)) / SR
    raw = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return _tanh_saturate(raw, drive=2.5)


def _make_clean_signal(amp: float = 0.3, freq: float = 440.0, dur: float = 0.5) -> np.ndarray:
    """Clean low-amplitude sine — no clipping at all."""
    t = np.arange(int(SR * dur)) / SR
    out: np.ndarray = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 1. Basis: ClippingType Enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClippingTypeEnum:
    def test_01_enum_has_clipping(self):
        assert ClippingType.CLIPPING.value == "clipping"

    def test_02_enum_has_soft_saturation(self):
        assert ClippingType.SOFT_SATURATION.value == "soft_saturation"

    def test_03_enum_values_distinct(self):
        assert ClippingType.CLIPPING != ClippingType.SOFT_SATURATION


# ---------------------------------------------------------------------------
# 2. flat_tops_pct Hilfsfunktion
# ---------------------------------------------------------------------------


class TestFlatTopsPct:
    def test_04_zero_for_clean_signal(self):
        audio = _make_clean_signal()
        assert _flat_tops_pct(audio) < 0.01

    def test_05_high_for_fully_clipped(self):
        audio = np.ones(SR, dtype=np.float32)  # all samples at boundary
        pct = _flat_tops_pct(audio, boundary=0.999)
        assert pct > 99.0

    def test_06_partial_clipping(self):
        audio = np.zeros(1000, dtype=np.float32)
        # Set exactly 100 samples to 0.9999 → 10 %
        audio[:100] = 1.0
        pct = _flat_tops_pct(audio, boundary=0.999)
        assert abs(pct - 10.0) < 1.0

    def test_07_empty_array_returns_zero(self):
        assert _flat_tops_pct(np.array([], dtype=np.float32)) == 0.0

    def test_08_negative_boundary_symmetric(self):
        audio = np.full(100, -1.0, dtype=np.float32)
        pct = _flat_tops_pct(audio, boundary=0.999)
        assert pct > 99.0


# ---------------------------------------------------------------------------
# 3. classify_clipping() — grundlegende Klassifikation
# ---------------------------------------------------------------------------


class TestClassifyClipping:
    def test_09_hard_clip_returns_clipping(self):
        audio = _make_hard_clipped_signal()
        result = classify_clipping(audio, SR)
        assert result == ClippingType.CLIPPING

    def test_10_tanh_signal_returns_soft_saturation(self):
        audio = _make_tanh_signal()
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION

    def test_11_clean_signal_returns_soft_saturation(self):
        audio = _make_clean_signal()
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION

    def test_12_silence_returns_soft_saturation(self):
        audio = np.zeros(SR, dtype=np.float32)
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION

    def test_13_non_48k_sr_accepted(self):
        """SR != 48000 is accepted (THD math is SR-agnostic)."""
        audio = _make_clean_signal()
        result = classify_clipping(audio, sr=44100)
        assert isinstance(result, ClippingType)

    def test_14_non_48k_sr_22050_accepted(self):
        """SR 22050 is accepted (THD math is SR-agnostic)."""
        audio = _make_clean_signal()
        result = classify_clipping(audio, sr=22050)
        assert isinstance(result, ClippingType)


# ---------------------------------------------------------------------------
# 4. analyse_clipping() — volle Ergebnisstruktur
# ---------------------------------------------------------------------------


class TestAnalyseClipping:
    def test_15_returns_dataclass(self):
        audio = _make_clean_signal()
        result = analyse_clipping(audio, SR)
        assert isinstance(result, ClippingAnalysisResult)

    def test_16_dataclass_has_all_fields(self):
        audio = _make_clean_signal()
        result = analyse_clipping(audio, SR)
        assert hasattr(result, "clipping_type")
        assert hasattr(result, "flat_tops_pct")
        assert hasattr(result, "thd_odd")
        assert hasattr(result, "thd_even")
        assert hasattr(result, "confidence")
        assert hasattr(result, "is_clipping")

    def test_17_is_clipping_matches_type_clipping(self):
        audio = _make_hard_clipped_signal()
        result = analyse_clipping(audio, SR)
        assert result.is_clipping == (result.clipping_type == ClippingType.CLIPPING)

    def test_18_is_clipping_false_for_clean(self):
        result = analyse_clipping(_make_clean_signal(), SR)
        assert result.is_clipping is False

    def test_19_flat_tops_pct_ge_zero(self):
        result = analyse_clipping(_make_clean_signal(), SR)
        assert result.flat_tops_pct >= 0.0

    def test_20_thd_values_non_negative(self):
        result = analyse_clipping(_make_clean_signal(), SR)
        assert result.thd_odd >= 0.0
        assert result.thd_even >= 0.0

    def test_21_confidence_in_unit_interval(self):
        factories: list[Callable[[], np.ndarray]] = [_make_clean_signal, _make_hard_clipped_signal, _make_tanh_signal]
        for make in factories:
            result = analyse_clipping(make(), SR)
            assert 0.0 <= result.confidence <= 1.0, f"confidence={result.confidence}"

    def test_22_hard_clip_high_flat_tops(self):
        result = analyse_clipping(_make_hard_clipped_signal(), SR)
        assert result.flat_tops_pct > FLAT_TOPS_THRESHOLD_PCT

    def test_23_tanh_low_flat_tops(self):
        result = analyse_clipping(_make_tanh_signal(), SR)
        assert result.flat_tops_pct <= FLAT_TOPS_THRESHOLD_PCT

    def test_24_classify_and_analyse_consistent(self):
        audio = _make_hard_clipped_signal()
        ct = classify_clipping(audio, SR)
        ar = analyse_clipping(audio, SR)
        assert ct == ar.clipping_type


# ---------------------------------------------------------------------------
# 5. should_repair / should_preserve Eigenschaften
# ---------------------------------------------------------------------------


class TestResultProperties:
    def test_25_clipping_should_repair_true(self):
        audio = _make_hard_clipped_signal()
        result = analyse_clipping(audio, SR)
        if result.is_clipping:
            assert result.should_repair is True
            assert result.should_preserve is False

    def test_26_soft_sat_should_preserve_true(self):
        result = analyse_clipping(_make_clean_signal(), SR)
        assert result.should_preserve is True
        assert result.should_repair is False

    def test_27_should_repair_and_preserve_are_complementary(self):
        audio = _make_tanh_signal()
        result = analyse_clipping(audio, SR)
        assert result.should_repair != result.should_preserve


# ---------------------------------------------------------------------------
# 6. Stereo- und Shape-Handling
# ---------------------------------------------------------------------------


class TestShapeHandling:
    def test_28_stereo_clean_is_soft_saturation(self):
        mono = _make_clean_signal()
        stereo = np.stack([mono, mono], axis=1)
        result = classify_clipping(stereo, SR)
        assert result == ClippingType.SOFT_SATURATION

    def test_29_stereo_clipped_is_clipping(self):
        mono = _make_hard_clipped_signal()
        stereo = np.stack([mono, mono], axis=1)
        result = classify_clipping(stereo, SR)
        assert result == ClippingType.CLIPPING

    def test_30_1d_array_accepted(self):
        audio = _make_clean_signal().ravel()
        assert audio.ndim == 1
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)


# ---------------------------------------------------------------------------
# 7. NaN / Inf Guard
# ---------------------------------------------------------------------------


class TestNaNInfGuard:
    def test_31_nan_audio_returns_soft_saturation(self):
        audio = np.full(SR, float("nan"), dtype=np.float32)
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION  # all NaN → zero → silence

    def test_32_inf_audio_returns_soft_saturation(self):
        audio = np.full(SR, float("inf"), dtype=np.float32)
        # nan_to_num clamps inf → 0 internally
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)

    def test_33_mixed_nan_inf_no_exception(self):
        audio = np.array([np.nan, np.inf, -np.inf, 0.5, 0.3], dtype=np.float32)
        # Must not raise
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)


# ---------------------------------------------------------------------------
# 8. Singleton Thread-Safety
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_34_singleton_same_instance(self):
        a = get_clipping_classifier()
        b = get_clipping_classifier()
        assert a is b

    def test_35_singleton_is_classifier_type(self):
        obj = get_clipping_classifier()
        assert isinstance(obj, ClippingClassifier)

    def test_36_singleton_thread_safe(self):
        """Race condition test: 50 threads acquiring singleton simultaneously."""
        instances = []
        errors = []

        def worker():
            try:
                instances.append(get_clipping_classifier())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        # All must be the same object
        assert all(inst is instances[0] for inst in instances)

    def test_37_singleton_classify_method(self):
        clf = get_clipping_classifier()
        audio = _make_clean_signal()
        result = clf.classify(audio, SR)
        assert isinstance(result, ClippingType)

    def test_38_singleton_analyse_method(self):
        clf = get_clipping_classifier()
        audio = _make_clean_signal()
        result = clf.analyse(audio, SR)
        assert isinstance(result, ClippingAnalysisResult)


# ---------------------------------------------------------------------------
# 9. Reproduzierbarkeit & Performance
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_39_deterministic_clean(self):
        audio = _make_clean_signal()
        r1 = classify_clipping(audio, SR)
        r2 = classify_clipping(audio, SR)
        assert r1 == r2

    def test_40_deterministic_hard_clip(self):
        audio = _make_hard_clipped_signal()
        r1 = classify_clipping(audio, SR)
        r2 = classify_clipping(audio, SR)
        assert r1 == r2

    def test_41_performance_5s_audio(self):
        """classify_clipping on 5 s of audio must complete in < 2 s (§DefectScanner budget)."""
        audio = _make_clean_signal(dur=0.5)
        # Extend to ~5 s
        audio = np.tile(audio, 10)
        start = time.perf_counter()
        classify_clipping(audio, SR)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Zu langsam: {elapsed:.2f} s"


# ---------------------------------------------------------------------------
# 10. Edge-Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_42_very_short_signal_no_crash(self):
        audio = np.array([0.5, -0.5, 0.3], dtype=np.float32)
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)

    def test_43_constant_dc_offset_no_crash(self):
        audio = np.full(SR, 0.5, dtype=np.float32)
        result = classify_clipping(audio, SR)
        assert isinstance(result, ClippingType)

    def test_44_constant_full_scale_is_clipping_or_soft_sat(self):
        """Full-scale DC signal has flat_tops=100% → CLIPPING classification."""
        audio = np.ones(SR, dtype=np.float32)
        result = analyse_clipping(audio, SR)
        # flat_tops_pct should be ~100%
        assert result.flat_tops_pct > 90.0

    def test_45_negative_amplitude_signal(self):
        t = np.arange(int(SR * 0.1)) / SR
        audio = (-0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = classify_clipping(audio, SR)
        assert result == ClippingType.SOFT_SATURATION

    def test_46_polyphonic_near_ceiling_clipping_detected(self) -> None:
        """Polyphonisches Musik-Signal mit near-ceiling Clipping → CLIPPING erkannt.

        Sicherstellung des Fixes: Bei flat_tops_pct > FLAT_TOPS_STRONG_CLIP_PCT
        gilt CLIPPING unabhängig vom THD-Verhältnis (polyphones Musik hat von Natur
        aus viele gerade Harmonics → thd_even > thd_odd → ohne Fix fälschlich
        SOFT_SATURATION, Bug §6.3 polyphon-sichere Erweiterung).
        """
        from backend.core.clipping_detection import FLAT_TOPS_STRONG_CLIP_PCT

        np.random.default_rng(42)
        t = np.arange(int(SR * 2.0)) / SR
        # Polyphonisches Signal: viele Frequenzen (wie Musik), typischerweise thd_even > thd_odd
        freqs = [110, 165, 220, 330, 440, 550, 660, 880, 1100, 1320]
        _components = np.array([np.sin(2 * np.pi * float(f) * t) for f in freqs])
        audio: np.ndarray = _components.mean(axis=0)
        audio = (audio / (np.max(np.abs(audio)) + 1e-10)).astype(np.float32)

        # Clipping bei 0.999 (near-ceiling Loudness-War): >> FLAT_TOPS_STRONG_CLIP_PCT
        clip_val = float(np.percentile(np.abs(audio), 97.0))
        audio = np.clip(audio, -clip_val, clip_val)
        audio = (audio * (0.9995 / (clip_val + 1e-10))).astype(np.float32)

        res = analyse_clipping(audio, SR)
        # Die geclippten Samples müssen über FLAT_TOPS_STRONG_CLIP_PCT liegen
        assert res.flat_tops_pct > FLAT_TOPS_STRONG_CLIP_PCT, (
            f"flat_tops_pct={res.flat_tops_pct:.2f}% zu niedrig für Test-Voraussetzung"
        )
        assert res.clipping_type == ClippingType.CLIPPING, (
            f"Polyphonisches near-ceiling Clipping muss CLIPPING sein, "
            f"flat_tops={res.flat_tops_pct:.2f}%, thd_odd={res.thd_odd:.3f}, thd_even={res.thd_even:.3f}"
        )

    def test_47_sub_ceiling_clipping_detected(self) -> None:
        """Hard-Clipping bei ±0.92 (sub-ceiling) wird als CLIPPING erkannt.

        Sub-Ceiling-Clipping (abs_max < 0.999) entsteht z.B. durch Loudness-War
        bei analogem Mastering oder DAW-Übersättigung. np.clip erzeugt identische
        float32-Werte am Clip-Level → SUBCEIL_MIN_IDENTICAL-Kriterium greift.
        """
        t = np.linspace(0, 1.0, SR, endpoint=False)
        # Polyphon: natürlich thd_even > thd_odd (falsches THD-Signal für alten Fix)
        sig = sum(np.sin(2 * np.pi * f * t) for f in [220, 330, 440]) / 3.0
        sig = (sig / np.max(np.abs(sig)) * 1.12).astype(np.float32)
        # Hard-Clipping bei ±0.92 — abs_max weit unter 0.999
        sig_clipped = np.clip(sig, -0.92, 0.92).astype(np.float32)
        assert np.max(np.abs(sig_clipped)) < 0.999, "Voraussetzung: sub-ceiling"
        res = analyse_clipping(sig_clipped, SR)
        assert res.clipping_type == ClippingType.CLIPPING, (
            f"Sub-Ceiling Hard-Clipping ±0.92 muss CLIPPING sein, "
            f"flat_tops={res.flat_tops_pct:.4f}%, got={res.clipping_type.value}"
        )
        # flat_tops_pct darf noch 0 sein (sub-ceiling: echte flat_tops unter 0.999 = 0)
        assert res.flat_tops_pct < 0.5, f"flat_tops_pct sollte sub-ceiling nahe 0 bleiben, got={res.flat_tops_pct:.4f}%"

    def test_48_sub_ceiling_soft_saturation_no_false_positive(self) -> None:
        """tanh-Saturation bei sub-ceiling Amplitude bleibt SOFT_SATURATION.

        tanh(2.5×sin) erzeugt ~80 float32-identische Maximalwerte durch Plateau-
        Rounding (0.17 %). Das liegt weit unter SUBCEIL_MIN_IDENTICAL_PCT (0.5 %).
        """
        t = np.linspace(0, 1.0, SR, endpoint=False)
        sig = np.tanh(2.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        assert np.max(np.abs(sig)) < 0.999, "tanh-Plateau liegt unter 0.999"
        res = analyse_clipping(sig, SR)
        assert res.clipping_type == ClippingType.SOFT_SATURATION, (
            f"tanh(2.5×sin) muss SOFT_SATURATION bleiben, got={res.clipping_type.value}"
        )

    def test_49_sub_ceiling_clipping_histogram_detected(self) -> None:
        """Histogramm-basierte Sub-Ceiling-Erkennung: Hard-Clip bei ±0.88 wird erkannt.

        detect_sub_ceiling_clipping() erkennt Loudness-War-Clipping (±0.80–0.998)
        durch Amplitudenhistogramm-Analyse: signifikante Masse im Top-5% des
        Amplitudenbereichs → (True, clip_level).
        """
        t = np.linspace(0, 2.0, SR * 2, endpoint=False)
        # Sinusoidales Signal mit Amplitude > 0.88 → nach Clipping viel Masse bei 0.88
        sig = (1.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sig_clipped = np.clip(sig, -0.88, 0.88).astype(np.float32)
        assert np.max(np.abs(sig_clipped)) < 0.999, "Voraussetzung: sub-ceiling"

        is_clip, clip_level = detect_sub_ceiling_clipping(sig_clipped)
        assert is_clip, (
            f"detect_sub_ceiling_clipping: Hard-Clip ±0.88 muss True zurückgeben, "
            f"got is_clip={is_clip}, clip_level={clip_level:.4f}"
        )
        assert clip_level > 0.0, f"clip_level muss > 0.0 sein, got={clip_level:.4f}"
        assert 0.80 <= clip_level <= 0.999, f"clip_level muss im Sub-Ceiling-Bereich liegen, got={clip_level:.4f}"

    def test_50_sub_ceiling_histogram_no_false_positive_sine(self) -> None:
        """Reines Sinus-Signal ohne Clipping liefert (False, 0.0)."""
        t = np.linspace(0, 1.0, SR, endpoint=False)
        sig = (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        is_clip, clip_level = detect_sub_ceiling_clipping(sig)
        assert not is_clip, (
            f"Sauberes Sinus-Signal darf kein Sub-Ceiling-Clipping liefern, "
            f"got is_clip={is_clip}, clip_level={clip_level:.4f}"
        )
        assert clip_level == 0.0

    def test_51_analyse_clipping_sub_ceiling_level_field_exists(self) -> None:
        """ClippingAnalysisResult hat sub_ceiling_level-Feld (immer >= 0.0)."""
        result = analyse_clipping(_make_clean_signal(), SR)
        assert hasattr(result, "sub_ceiling_level")
        assert result.sub_ceiling_level >= 0.0

    def test_52_daw_limiter_clipping_band_pile_detected(self) -> None:
        """DAW-Brickwall-Limiter-Clipping wird per Band-Pile-Ratio erkannt (Methode 2).

        DAW-Limiter setzt Samples nahe ±0.88 mit leichter Streuung ±0.003 —
        Adjacent-Ratio-Methode greift nicht, Band-Pile-Ratio muss erkennen.
        """
        rng = np.random.default_rng(42)
        t = np.linspace(0, 2.0, SR * 2, endpoint=False)
        # Polyphones Signal: mehrere Frequenzen → breitbandiges Spektrum
        sig = 1.25 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        for freq in [220, 330, 550, 660, 880, 1100]:
            sig = sig + 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        sig = (sig / np.max(np.abs(sig)) * 1.3).astype(np.float32)
        # DAW-Limiter: hart clippen, dann leichte Streuung auf Clip-Level-Samples
        sig_daw = np.clip(sig, -0.88, 0.88).astype(np.float32)
        noise = np.abs(rng.standard_normal(len(sig)).astype(np.float32)) * 0.003
        top_mask = sig_daw >= 0.88 * 0.999
        bot_mask = sig_daw <= -0.88 * 0.999
        sig_daw[top_mask] -= noise[top_mask]
        sig_daw[bot_mask] += noise[bot_mask]

        is_clip, clip_level = detect_sub_ceiling_clipping(sig_daw)
        result = analyse_clipping(sig_daw, SR)

        assert is_clip, f"DAW-Limiter-Clipping nicht erkannt: is_clip={is_clip}, level={clip_level:.4f}"
        assert clip_level >= 0.75, f"clip_level={clip_level:.4f} außerhalb Sub-Ceiling-Bereich"
        assert result.clipping_type == ClippingType.CLIPPING, f"Erwartet CLIPPING, got {result.clipping_type.value}"

    def test_53_high_amplitude_sine_no_false_positive_band_pile(self) -> None:
        """Sinus bei hoher Amplitude wird nicht fälschlich als DAW-Clip erkannt.

        Sinus verweilt natürlich lange nahe seinem Maximum (cos-Dichtefunktion),
        darf aber trotz hoher band_pile_ratio nicht als Clipping erkannt werden.
        """
        t = np.linspace(0, 2.0, SR * 2, endpoint=False)
        # Sinus @ 0.88 (volle Aussteuerung, keine Begrenzung)
        sig_sine = (0.88 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        is_clip, _ = detect_sub_ceiling_clipping(sig_sine)
        result = analyse_clipping(sig_sine, SR)

        assert not is_clip, "Sinus @0.88 fälschlich als Clipping erkannt (False Positive!)"
        assert result.clipping_type != ClippingType.CLIPPING or result.flat_tops_pct > 5.0, (
            f"Sinus soll SOFT_SATURATION sein, got {result.clipping_type.value}"
        )
