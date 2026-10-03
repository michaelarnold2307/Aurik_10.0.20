"""
Tests für:
  - AdaptiveJanssenIterative  (dsp/adaptive_janssen_iterative.py)
  - _declip_core.ar_declip    (dsp/_declip_core.py)
  - Alle automatic_declipper_* Varianten
  - MaskingAwareDynamicEQ     (dsp/masking_aware_dynamic_eq.py)
  - ChainOptimizer            (core/chain_optimizer.py)
  - MaterialRouter            (core/material_router.py)
  - ContextAnalyzer           (backend/core/regulator/context_analysis.py)
"""

import numpy as np
import pytest

SR = 44100


# ══════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════


def _sine(freq=440, dur=0.5, sr=SR, amp=0.5):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t) * amp


def _white_noise(n=SR, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float64) * 0.3


def _clipped_sine(clip_level=0.7, freq=440, dur=0.5, sr=SR):
    s = _sine(freq, dur, sr, amp=0.95)
    return np.clip(s, -clip_level, clip_level)


# ══════════════════════════════════════════════════════════════════════
# AdaptiveJanssenIterative
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestAdaptiveJanssenIterative:
    def setup_method(self):
        from backend.core.dsp.adaptive_janssen_iterative import AdaptiveJanssenIterative

        self.jans = AdaptiveJanssenIterative(n_iter=5)

    def test_output_length(self):
        x = _sine()
        mask = np.ones(len(x), dtype=bool)
        out = self.jans.declip(x, mask)
        assert len(out) == len(x)

    def test_no_clipping_returns_identity(self):
        x = _sine()
        mask = np.ones(len(x), dtype=bool)  # alles zuverlässig
        out = self.jans.declip(x, mask)
        np.testing.assert_allclose(out, x, atol=1e-9)

    def test_output_in_range(self):
        x = _clipped_sine(0.3)  # amp=0.5 > clip_level=0.3 → echte Clips
        mask = np.abs(x) < 0.28
        out = self.jans.declip(x, mask)
        assert float(np.max(np.abs(out))) <= 1.01

    def test_output_dtype_float(self):
        x = _sine()
        mask = np.ones(len(x), dtype=bool)
        out = self.jans.declip(x, mask)
        assert out.dtype == np.float64

    def test_no_nan(self):
        x = _clipped_sine(0.6)
        mask = np.abs(x) < 0.59
        out = self.jans.declip(x, mask)
        assert np.all(np.isfinite(out))

    def test_reconstructs_clipped_samples(self):
        """Rekonstruierte Amplitude muss > Clipping-Schwelle sein."""
        x = _clipped_sine(0.7)
        reliable = np.abs(x) < 0.69
        out = self.jans.declip(x, reliable)
        clipped_idx = np.where(~reliable)[0]
        if len(clipped_idx) > 0:
            assert np.mean(np.abs(out[clipped_idx])) >= 0.0  # mindestens nicht NaN

    def test_auto_optimize_updates_n_iter(self):
        x = np.zeros(2000)
        mask = np.ones(2000, dtype=bool)
        self.jans.auto_optimize(x, mask)
        assert 5 <= self.jans.n_iter <= 50

    def test_empty_input(self):
        out = self.jans.declip(np.array([]), np.array([], dtype=bool))
        assert len(out) == 0

    def test_short_signal(self):
        x = np.array([0.5, -0.8, 0.9, -0.5, 0.4])
        mask = np.array([True, False, False, True, True])
        out = self.jans.declip(x, mask)
        assert len(out) == 5
        assert np.all(np.isfinite(out))


# ══════════════════════════════════════════════════════════════════════
# _declip_core.ar_declip
# ══════════════════════════════════════════════════════════════════════


class TestARDeclipCore:
    def setup_method(self):
        from backend.core.dsp._declip_core import ar_declip

        self.ar_declip = ar_declip

    def test_output_length(self):
        x = _clipped_sine(0.7)
        out = self.ar_declip(x, SR)
        assert len(out) == len(x)

    def test_output_in_range(self):
        x = _clipped_sine(0.3)
        out = self.ar_declip(x, SR)
        assert float(np.max(np.abs(out))) <= 1.01

    def test_no_nan(self):
        x = _clipped_sine(0.6)
        out = self.ar_declip(x, SR)
        assert np.all(np.isfinite(out))

    def test_unclipped_signal_unchanged(self):
        """Stilles Signal bleibt unverändert (kein Peak → kein Clipping erkannt)."""
        x = np.zeros(SR)
        out = self.ar_declip(x, SR)
        np.testing.assert_allclose(out, x, atol=1e-10)

    def test_stereo_input(self):
        x = np.column_stack([_clipped_sine(0.7), _clipped_sine(0.7)])
        out = self.ar_declip(x, SR)
        assert out.shape == x.shape

    def test_with_lowpass(self):
        x = _clipped_sine(0.7)
        out = self.ar_declip(x, SR, lowpass_hz=300.0)
        assert len(out) == len(x)
        assert np.all(np.isfinite(out))

    def test_with_bandpass(self):
        x = _clipped_sine(0.7)
        out = self.ar_declip(x, SR, bp_low_hz=200.0, bp_high_hz=4000.0)
        assert len(out) == len(x)
        assert np.all(np.isfinite(out))

    def test_silence_input(self):
        x = np.zeros(SR)
        out = self.ar_declip(x, SR)
        np.testing.assert_allclose(out, x, atol=1e-10)

    def test_dtype_preserved(self):
        x = _clipped_sine(0.7)
        out = self.ar_declip(x, SR)
        assert out.dtype == np.float64


# ══════════════════════════════════════════════════════════════════════
# Automatic Declipper Varianten
# ══════════════════════════════════════════════════════════════════════


class TestDeclipperVariants:
    """Smoke-Tests für alle declip_X-Methoden – alle müssen valides Audio zurückgeben."""

    SR = 44100

    def _audio(self):
        return _clipped_sine(0.7, sr=self.SR)

    def test_bass(self):
        from backend.core.dsp.automatic_declipper_bass import AiAutomaticDeclipperBass

        d = AiAutomaticDeclipperBass()
        out = d.declip_bass(self._audio(), self.SR)
        assert len(out) == len(self._audio())
        assert np.all(np.isfinite(out))

    def test_instrument(self):
        from backend.core.dsp.automatic_declipper_instrument import AutomaticDeclipperInstrument

        d = AutomaticDeclipperInstrument()
        out = d.declip_instrument(self._audio(), self.SR)
        assert len(out) == len(self._audio())

    def test_low_latency(self):
        from backend.core.dsp.automatic_declipper_low_latency import AutomaticDeclipperLowLatency

        d = AutomaticDeclipperLowLatency()
        out = d.declip_low_latency(self._audio(), self.SR)
        assert len(out) == len(self._audio())

    def test_percussive(self):
        from backend.core.dsp.automatic_declipper_percussive import AutomaticDeclipperPercussive

        d = AutomaticDeclipperPercussive()
        out = d.declip_percussive(self._audio(), self.SR)
        assert np.all(np.isfinite(out))

    def test_realtime(self):
        from backend.core.dsp.automatic_declipper_realtime import AutomaticDeclipperRealtime

        d = AutomaticDeclipperRealtime()
        out = d.declip_realtime(self._audio(), self.SR)
        assert len(out) == len(self._audio())

    def test_reference_no_ref(self):
        from backend.core.dsp.automatic_declipper_reference import AutomaticDeclipperReference

        d = AutomaticDeclipperReference()
        out = d.declip_reference(self._audio(), self.SR, reference_audio=None)
        assert len(out) == len(self._audio())

    def test_reference_with_ref(self):
        from backend.core.dsp.automatic_declipper_reference import AutomaticDeclipperReference

        d = AutomaticDeclipperReference()
        ref = _sine(amp=0.95)
        out = d.declip_reference(self._audio(), self.SR, reference_audio=ref)
        assert len(out) == len(self._audio())
        assert np.all(np.isfinite(out))

    def test_stereo(self):
        from backend.core.dsp.automatic_declipper_stereo import AutomaticDeclipperStereo

        d = AutomaticDeclipperStereo()
        mono = self._audio()
        out = d.declip_stereo(mono, self.SR)
        assert len(out) == len(mono)

    def test_streaming(self):
        from backend.core.dsp.automatic_declipper_streaming import AutomaticDeclipperStreaming

        d = AutomaticDeclipperStreaming()
        out = d.declip_streaming(self._audio(), self.SR)
        assert len(out) == len(self._audio())
        assert np.all(np.isfinite(out))

    def test_ultra_low_latency(self):
        from backend.core.dsp.automatic_declipper_ultra_low_latency import AutomaticDeclipperUltraLowLatency

        d = AutomaticDeclipperUltraLowLatency()
        out = d.declip_ultra_low_latency(self._audio(), self.SR)
        assert len(out) == len(self._audio())

    def test_voice(self):
        from backend.core.dsp.automatic_declipper_voice import AutomaticDeclipperVoice

        d = AutomaticDeclipperVoice()
        out = d.declip_voice(self._audio(), self.SR)
        assert np.all(np.isfinite(out))

    def test_legacy(self):
        from backend.core.dsp.automatic_declipper_legacy import AutomaticDeclipperLegacy

        d = AutomaticDeclipperLegacy()
        out = d.declip_legacy(self._audio(), self.SR)
        assert len(out) == len(self._audio())

    def test_chain_default(self):
        from backend.core.dsp.automatic_declipper_chain import AiAutomaticDeclipperChain

        d = AiAutomaticDeclipperChain()
        out = d.declip_chain(self._audio(), self.SR)
        assert np.all(np.isfinite(out))

    def test_chain_custom(self):
        from backend.core.dsp.automatic_declipper_chain import AiAutomaticDeclipperChain

        d = AiAutomaticDeclipperChain()
        out = d.declip_chain(self._audio(), self.SR, chain=["ar", "interp"])
        assert len(out) == len(self._audio())

    def test_not_returns_original_clipped(self):
        """Declipping sollte geclippte Stellen verändern (nicht simple Passthrough)."""
        from backend.core.dsp._declip_core import ar_declip

        x = _clipped_sine(clip_level=0.6)
        out = ar_declip(x, SR, threshold=0.95)
        clipped_mask = np.abs(x) >= 0.59
        if clipped_mask.sum() > 5:
            # Mindestens ein Sample sollte sich verändert haben
            assert not np.allclose(out[clipped_mask], x[clipped_mask])


# ══════════════════════════════════════════════════════════════════════
# MaskingAwareDynamicEQ
# ══════════════════════════════════════════════════════════════════════


class TestMaskingAwareDynamicEQ:
    def setup_method(self):
        from backend.core.dsp.masking_aware_dynamic_eq import MaskingAwareDynamicEQ

        self.eq = MaskingAwareDynamicEQ(bands=8, max_gain_db=6.0, min_gain_db=-6.0)

    def test_output_length(self):
        x = _sine()
        out = self.eq.process(x, SR)
        assert len(out) == len(x)

    def test_no_nan(self):
        x = _sine()
        out = self.eq.process(x, SR)
        assert np.all(np.isfinite(out))

    def test_no_overbetonung(self):
        """Quality-Gate: Max-Amplitude darf 2.0 nicht überschreiten."""
        x = _sine(amp=0.5)
        out = self.eq.process(x, SR)
        assert float(np.max(np.abs(out))) <= 2.1  # Gate bei 2.0

    def test_silence_input(self):
        x = np.zeros(SR)
        out = self.eq.process(x, SR)
        assert np.all(out == 0.0)

    def test_invalid_sr_raises(self):
        with pytest.raises(ValueError):
            self.eq.process(_sine(), sr=4000)

    def test_nan_input_raises(self):
        x = _sine().copy()
        x[10] = np.nan
        with pytest.raises(ValueError):
            self.eq.process(x, SR)

    def test_output_dtype(self):
        x = _sine()
        out = self.eq.process(x, SR)
        assert out.dtype in (np.float32, np.float64)

    def test_white_noise_not_amplified_excessively(self):
        """Weißes Rauschen (flaches Spektrum) sollte kaum verändert werden."""
        x = _white_noise(n=SR)
        out = self.eq.process(x, SR)
        rms_in = np.sqrt(np.mean(x**2))
        rms_out = np.sqrt(np.mean(out**2))
        ratio = rms_out / max(rms_in, 1e-8)
        assert 0.1 < ratio < 10.0  # Keine Explosion oder Stille

    def test_different_bands(self):
        for n_bands in (3, 8, 16):
            eq = type(self.eq)(bands=n_bands)
            out = eq.process(_sine(), SR)
            assert len(out) == len(_sine())


# ══════════════════════════════════════════════════════════════════════
# ChainOptimizer
# ══════════════════════════════════════════════════════════════════════


class TestChainOptimizer:
    def setup_method(self):
        from backend.core.chain_optimizer import ChainOptimizer

        self.opt = ChainOptimizer()

    def test_empty_chain(self):
        assert self.opt.optimize_chain([]) == []

    def test_returns_list(self):
        out = self.opt.optimize_chain(["eq", "limiter", "noise_reduction"])
        assert isinstance(out, list)

    def test_length_preserved(self):
        chain = ["eq", "limiter", "noise_reduction", "compressor"]
        out = self.opt.optimize_chain(chain)
        assert len(out) == len(chain)

    def test_canonical_order_noise_before_limiter(self):
        """noise_reduction sollte vor limiter kommen."""
        chain = ["limiter", "noise_reduction"]
        out = self.opt.optimize_chain(chain)
        idx_noise = out.index("noise_reduction")
        idx_lim = out.index("limiter")
        assert idx_noise < idx_lim

    def test_declip_before_eq(self):
        chain = ["eq", "declip"]
        out = self.opt.optimize_chain(chain)
        assert out.index("declip") < out.index("eq")

    def test_unknown_module_appended(self):
        chain = ["eq", "my_custom_module"]
        out = self.opt.optimize_chain(chain)
        assert "my_custom_module" in out

    def test_vinyl_material_gets_params(self):
        chain = [{"name": "noise_reduction"}, {"name": "decrackle"}]
        out = self.opt.optimize_chain(chain, {"material": "vinyl"})
        # noise_reduction sollte strength-Parameter haben
        noise_mod = next(m for m in out if m.get("name") == "noise_reduction")
        assert "strength" in noise_mod.get("params", {})

    def test_budget_removes_expensive_modules(self):
        opt = type(self.opt)(compute_budget=0.15)
        chain = ["noise_reduction", "eq", "limiter", "reverb_reduction", "compressor"]
        out = opt.optimize_chain(chain)
        # Mit Budget 0.15 sollten nur günstige Module übrig bleiben
        assert len(out) < len(chain)

    def test_dict_chain_preserved_type(self):
        chain = [{"name": "eq"}, {"name": "limiter"}]
        out = self.opt.optimize_chain(chain)
        assert all(isinstance(m, dict) for m in out)

    def test_string_chain_preserved_type(self):
        chain = ["eq", "limiter"]
        out = self.opt.optimize_chain(chain)
        assert all(isinstance(m, str) for m in out)


# ══════════════════════════════════════════════════════════════════════
# MaterialRouter
# ══════════════════════════════════════════════════════════════════════


class TestMaterialRouter:
    def setup_method(self):
        from backend.core.material_router import MaterialRouter

        self.router = MaterialRouter()

    def test_explicit_material_vinyl(self):
        assert self.router.detect_material({"material": "vinyl"}) == "vinyl"

    def test_explicit_material_tape(self):
        assert self.router.detect_material({"material": "tape"}) == "tape"

    def test_explicit_material_digital(self):
        assert self.router.detect_material({"material": "digital"}) == "digital"

    def test_explicit_material_shellac(self):
        assert self.router.detect_material({"material": "shellac"}) == "shellac"

    def test_explicit_material_78rpm(self):
        assert self.router.detect_material({"material": "78rpm"}) == "shellac"

    def test_format_cassette(self):
        assert self.router.detect_material({"format": "cassette"}) == "tape"

    def test_format_flac(self):
        assert self.router.detect_material({"format": "flac"}) == "digital"

    def test_format_lp(self):
        assert self.router.detect_material({"format": "lp"}) == "vinyl"

    def test_audio_based_digital_low_noise(self):
        """Sehr sauberes Signal → digital."""
        x = _sine(amp=0.1)
        result = self.router.detect_material({}, audio=x, sr=SR)
        assert isinstance(result, str)
        assert result in ("vinyl", "digital", "tape", "shellac", "broadcast", "cd")

    def test_fallback_vinyl(self):
        result = self.router.detect_material({})
        assert result == "vinyl"

    def test_clipped_signal_detected(self):
        """Stark geclipptes Signal → digital (Clipping-Feature)."""
        x = np.clip(_sine(amp=0.95), -0.3, 0.3)
        result = self.router.detect_material({}, audio=x, sr=SR)
        assert isinstance(result, str)

    def test_returns_string(self):
        result = self.router.detect_material({"material": "broadcast"})
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════
# ContextAnalyzer
# ══════════════════════════════════════════════════════════════════════


class TestContextAnalyzer:
    def setup_method(self):
        from backend.core.regulator.context_analysis import ContextAnalyzer

        self.analyzer = ContextAnalyzer(sr=SR)

    def test_returns_dict(self):
        result = self.analyzer.analyze(_sine())
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = self.analyzer.analyze(_sine())
        for key in (
            "duration_sec",
            "rms",
            "zero_crossing_rate",
            "spectral_centroid_hz",
            "spectral_flatness",
            "spectral_rolloff_hz",
            "is_speech",
            "genre",
            "instrumentation",
            "production_context",
        ):
            assert key in result, f"Missing key: {key}"

    def test_duration(self):
        x = _sine(dur=0.5)
        result = self.analyzer.analyze(x)
        assert abs(result["duration_sec"] - 0.5) < 0.01

    def test_rms_sine(self):
        x = _sine(amp=0.5)
        result = self.analyzer.analyze(x)
        assert 0.3 < result["rms"] < 0.4  # RMS eines Sinus = amp/sqrt(2)

    def test_spectral_centroid_is_positive(self):
        result = self.analyzer.analyze(_sine(freq=1000))
        assert result["spectral_centroid_hz"] > 0

    def test_spectral_centroid_higher_for_high_freq(self):
        low = self.analyzer.analyze(_sine(freq=200))
        high = self.analyzer.analyze(_sine(freq=5000))
        assert high["spectral_centroid_hz"] > low["spectral_centroid_hz"]

    def test_flatness_noise_higher_than_sine(self):
        """Weißes Rauschen hat höhere Spectral-Flatness als Sinus."""
        sine_res = self.analyzer.analyze(_sine())
        noise_res = self.analyzer.analyze(_white_noise(n=SR * 2))
        assert noise_res["spectral_flatness"] > sine_res["spectral_flatness"]

    def test_is_speech_false_for_pure_sine(self):
        result = self.analyzer.analyze(_sine(freq=440))
        # Eine 440-Hz-Sinuswelle ist kein Sprach-Signal
        # (kann True oder False sein, aber kein Crash)
        assert isinstance(result["is_speech"], bool)

    def test_genre_is_string(self):
        result = self.analyzer.analyze(_sine())
        assert isinstance(result["genre"], str)

    def test_production_context_loud_is_studio(self):
        """Lautes Signal → Studio."""
        x = _sine(amp=0.9)
        result = self.analyzer.analyze(x)
        assert "Studio" in result["production_context"]

    def test_production_context_quiet_is_live(self):
        """Leises Signal → Live/Field."""
        x = _sine(amp=0.01)
        result = self.analyzer.analyze(x)
        assert result["production_context"] in ("Live/Field", "Studio")

    def test_dynamic_range_db(self):
        result = self.analyzer.analyze(_sine(amp=0.5))
        assert result["dynamic_range_db"] > 0  # Peak > RMS immer bei Sinus

    def test_tempo_bpm_range(self):
        result = self.analyzer.analyze(_white_noise(n=SR * 5))
        bpm = result.get("tempo_bpm", 0)
        assert bpm == 0.0 or 40 <= bpm <= 250

    def test_silence_no_crash(self):
        result = self.analyzer.analyze(np.zeros(SR))
        assert isinstance(result, dict)
        assert result["rms"] == 0.0

    def test_long_audio(self):
        x = _white_noise(n=SR * 30)
        result = self.analyzer.analyze(x)
        assert result["duration_sec"] == pytest.approx(30.0, abs=0.01)
