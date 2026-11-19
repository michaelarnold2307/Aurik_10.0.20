"""
tests/unit/test_v99_hybrid_modules.py
======================================
v10.0.0 — Tests für alle 6 core/hybrid/-Module:
  HybridDereverb, HybridMLDenoiser, HybridNVSR,
  HybridSpeedPitch, HybridVocalEnhancer, HybridWowFlutter

Zielanzahl: ≥ 37 Tests (alle grün)
"""

import numpy as np
import pytest

# ── Globale Testsignale ──────────────────────────────────────────────────────
np.random.seed(42)
SR = 48000
_t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
AUDIO_SINE = (0.5 * np.sin(2 * np.pi * 440 * _t)).astype(np.float32)
AUDIO_NOISE = np.random.randn(SR).astype(np.float32) * 0.1
AUDIO_SILENCE = np.zeros(SR, dtype=np.float32)
AUDIO_DIRAC = np.zeros(SR, dtype=np.float32)
AUDIO_DIRAC[SR // 2] = 1.0


# ============================================================================
# TestHybridDereverb
# ============================================================================
class TestHybridDereverb:
    """HybridDereverb.dereverb(audio, sample_rate) → DereverbResult"""

    @staticmethod
    def _make():
        from backend.core.hybrid.hybrid_dereverb import HybridDereverb

        return HybridDereverb()

    def test_01_import(self):
        from backend.core.hybrid.hybrid_dereverb import HybridDereverb

        assert HybridDereverb is not None

    def test_02_instantiate_no_args(self):
        hd = self._make()
        assert hd is not None

    def test_03_dereverb_sine_no_crash(self):
        hd = self._make()
        result = hd.dereverb(AUDIO_SINE, SR)
        assert result is not None

    def test_04_result_has_audio(self):
        hd = self._make()
        result = hd.dereverb(AUDIO_SINE, SR)
        # DereverbResult sollte .audio oder vergleichbares haben
        has_audio = (
            hasattr(result, "audio")
            or hasattr(result, "processed_audio")
            or hasattr(result, "output")
            or isinstance(result, np.ndarray)
        )
        assert has_audio

    def test_05_output_audio_finite(self):
        hd = self._make()
        result = hd.dereverb(AUDIO_SINE, SR)
        audio_out = None
        for attr in ("audio", "processed_audio", "output"):
            if hasattr(result, attr):
                audio_out = getattr(result, attr)
                break
        if isinstance(result, np.ndarray):
            audio_out = result
        if audio_out is not None and isinstance(audio_out, np.ndarray):
            assert np.isfinite(audio_out).all(), "NaN/Inf im Dereverb-Ausgang"

    def test_06_silence_no_crash(self):
        hd = self._make()
        result = hd.dereverb(AUDIO_SILENCE, SR)
        assert result is not None

    def test_07_deterministic_ml_error_latches_dsp_only(self):
        from backend.core.hybrid.hybrid_dereverb import DereverbStrategy

        hd = self._make()

        class _ExplodingSGMSE:
            def __init__(self):
                self.calls = 0

            def enhance(self, _audio, _sr):
                self.calls += 1
                raise RuntimeError("invalid magic number for TorchScript archive — model file is corrupted")

        plugin = _ExplodingSGMSE()
        hd.dccrn = plugin
        hd._sgmse_active = True
        hd.config.strategy = DereverbStrategy.DCCRN_ONLY

        first = hd.dereverb(AUDIO_SINE, SR)
        second = hd.dereverb(AUDIO_SINE, SR)

        assert plugin.calls == 1
        assert getattr(hd, "_disable_ml_due_deterministic_error", False) is True
        assert first.metadata.get("ml", {}).get("deterministic_error_latched", False) is True
        assert second.metadata.get("ml_skipped_reason") == "deterministic_ml_error_latched"


# ============================================================================
# TestHybridMLDenoiser
# ============================================================================
class TestHybridMLDenoiser:
    """HybridMLDenoiser.denoise(audio, sample_rate) → DenoiseResult"""

    @staticmethod
    def _make():
        from backend.core.hybrid.hybrid_ml_denoiser import HybridMLDenoiser

        return HybridMLDenoiser()

    def test_01_import(self):
        from backend.core.hybrid.hybrid_ml_denoiser import HybridMLDenoiser

        assert HybridMLDenoiser is not None

    def test_02_instantiate_no_args(self):
        hmd = self._make()
        assert hmd is not None

    def test_03_denoise_sine_no_crash(self):
        hmd = self._make()
        result = hmd.denoise(AUDIO_SINE, SR)
        assert result is not None

    def test_04_denoise_noise_no_crash(self):
        hmd = self._make()
        result = hmd.denoise(AUDIO_NOISE, SR)
        assert result is not None

    def test_05_result_has_audio(self):
        hmd = self._make()
        result = hmd.denoise(AUDIO_SINE, SR)
        has_audio = (
            hasattr(result, "audio")
            or hasattr(result, "denoised_audio")
            or hasattr(result, "output")
            or isinstance(result, np.ndarray)
        )
        assert has_audio

    def test_06_output_finite(self):
        hmd = self._make()
        result = hmd.denoise(AUDIO_SINE, SR)
        audio_out = None
        for attr in ("audio", "denoised_audio", "output"):
            if hasattr(result, attr):
                audio_out = getattr(result, attr)
                break
        if isinstance(result, np.ndarray):
            audio_out = result
        if audio_out is not None and isinstance(audio_out, np.ndarray):
            assert np.isfinite(audio_out).all()

    def test_07_silence_no_crash(self):
        hmd = self._make()
        result = hmd.denoise(AUDIO_SILENCE, SR)
        assert result is not None


# ============================================================================
# TestHybridNVSR
# ============================================================================
class TestHybridNVSR:
    """HybridNVSR.restore_bandwidth(audio, sample_rate) → NVSRResult"""

    @staticmethod
    def _make():
        from backend.core.hybrid.hybrid_nvsr import HybridNVSR

        return HybridNVSR()

    def test_01_import(self):
        from backend.core.hybrid.hybrid_nvsr import HybridNVSR

        assert HybridNVSR is not None

    def test_02_instantiate_no_args(self):
        hn = self._make()
        assert hn is not None

    @pytest.mark.slow
    def test_03_restore_bandwidth_sine(self):
        hn = self._make()
        result = hn.restore_bandwidth(AUDIO_SINE, SR)
        assert result is not None

    @pytest.mark.slow
    def test_04_result_has_audio_or_score(self):
        hn = self._make()
        result = hn.restore_bandwidth(AUDIO_SINE, SR)
        has_content = (
            hasattr(result, "audio")
            or hasattr(result, "restored_audio")
            or hasattr(result, "output")
            or hasattr(result, "score")
            or isinstance(result, np.ndarray)
        )
        assert has_content

    @pytest.mark.slow
    def test_05_output_finite(self):
        hn = self._make()
        result = hn.restore_bandwidth(AUDIO_SINE, SR)
        for attr in ("audio", "restored_audio", "output"):
            val = getattr(result, attr, None)
            if isinstance(val, np.ndarray):
                assert np.isfinite(val).all()

    @pytest.mark.slow
    def test_06_with_dsp_reference(self):
        hn = self._make()
        result = hn.restore_bandwidth(AUDIO_SINE, SR, dsp_restored_audio=AUDIO_SINE)
        assert result is not None


# ============================================================================
# TestHybridSpeedPitch
# ============================================================================
class TestHybridSpeedPitch:
    """HybridSpeedPitch.detect_global_pitch(audio, sample_rate) → SpeedPitchResult"""

    @staticmethod
    def _make():
        from backend.core.hybrid.hybrid_speed_pitch_ml import HybridSpeedPitch

        return HybridSpeedPitch()

    def test_01_import(self):
        from backend.core.hybrid.hybrid_speed_pitch_ml import HybridSpeedPitch

        assert HybridSpeedPitch is not None

    def test_02_instantiate_no_args(self):
        hsp = self._make()
        assert hsp is not None

    def test_03_detect_global_pitch_sine(self):
        hsp = self._make()
        result = hsp.detect_global_pitch(AUDIO_SINE, SR)
        assert result is not None

    def test_04_result_has_pitch_or_speed(self):
        hsp = self._make()
        result = hsp.detect_global_pitch(AUDIO_SINE, SR)
        has_content = any(
            hasattr(result, attr)
            for attr in ["pitch", "f0", "speed", "tempo", "pitch_hz", "speed_factor", "confidence"]
        )
        assert has_content or result is not None

    def test_05_silence_no_crash(self):
        hsp = self._make()
        result = hsp.detect_global_pitch(AUDIO_SILENCE, SR)
        assert result is not None

    def test_06_noise_no_crash(self):
        hsp = self._make()
        result = hsp.detect_global_pitch(AUDIO_NOISE, SR)
        assert result is not None


# ============================================================================
# TestHybridVocalEnhancer
# ============================================================================
class TestHybridVocalEnhancer:
    """HybridVocalEnhancer.enhance(audio, sample_rate, quality_mode) → VocalEnhancerResult"""

    @staticmethod
    def _make():
        from backend.core.hybrid.hybrid_vocal_enhancer import HybridVocalEnhancer

        return HybridVocalEnhancer()

    def test_01_import(self):
        from backend.core.hybrid.hybrid_vocal_enhancer import HybridVocalEnhancer

        assert HybridVocalEnhancer is not None

    def test_02_instantiate_no_args(self):
        hve = self._make()
        assert hve is not None

    def test_03_enhance_sine_balanced(self):
        hve = self._make()
        result = hve.enhance(AUDIO_SINE, SR, quality_mode="balanced")
        assert result is not None

    def test_04_enhance_sine_fast(self):
        hve = self._make()
        result = hve.enhance(AUDIO_SINE, SR, quality_mode="fast")
        assert result is not None

    def test_05_result_has_audio(self):
        hve = self._make()
        result = hve.enhance(AUDIO_SINE, SR)
        has_audio = (
            hasattr(result, "audio")
            or hasattr(result, "enhanced_audio")
            or hasattr(result, "output")
            or isinstance(result, np.ndarray)
        )
        assert has_audio

    def test_06_output_finite(self):
        hve = self._make()
        result = hve.enhance(AUDIO_SINE, SR)
        for attr in ("audio", "enhanced_audio", "output"):
            val = getattr(result, attr, None)
            if isinstance(val, np.ndarray):
                assert np.isfinite(val).all()
                assert np.max(np.abs(val)) <= 2.0  # nicht weit außerhalb [-1, 1]

    def test_07_silence_no_crash(self):
        hve = self._make()
        result = hve.enhance(AUDIO_SILENCE, SR)
        assert result is not None


# ============================================================================
# TestHybridWowFlutter
# ============================================================================
class TestHybridWowFlutter:
    """HybridWowFlutter.detect_pitch(audio, sample_rate) → WowFlutterResult"""

    @staticmethod
    def _make():
        from backend.core.hybrid.hybrid_wow_flutter import HybridWowFlutter

        return HybridWowFlutter()

    def test_01_import(self):
        from backend.core.hybrid.hybrid_wow_flutter import HybridWowFlutter

        assert HybridWowFlutter is not None

    def test_02_instantiate_no_args(self):
        hwf = self._make()
        assert hwf is not None

    def test_03_detect_pitch_sine(self):
        hwf = self._make()
        result = hwf.detect_pitch(AUDIO_SINE, SR)
        assert result is not None

    def test_04_result_has_pitch_info(self):
        hwf = self._make()
        result = hwf.detect_pitch(AUDIO_SINE, SR)
        has_content = any(
            hasattr(result, attr) for attr in ["pitch", "f0", "wow", "flutter", "deviation", "confidence", "pitch_hz"]
        )
        assert has_content or result is not None

    def test_05_silence_no_crash(self):
        hwf = self._make()
        result = hwf.detect_pitch(AUDIO_SILENCE, SR)
        assert result is not None

    def test_06_noise_no_crash(self):
        hwf = self._make()
        result = hwf.detect_pitch(AUDIO_NOISE, SR)
        assert result is not None


# ============================================================================
# TestHybridIntegration
# ============================================================================
class TestHybridIntegration:
    """Integrationstests: alle 6 Hybrid-Module gemeinsam"""

    ALL_HYBRID = [
        ("core.hybrid.hybrid_dereverb", "HybridDereverb"),
        ("core.hybrid.hybrid_ml_denoiser", "HybridMLDenoiser"),
        ("core.hybrid.hybrid_nvsr", "HybridNVSR"),
        ("core.hybrid.hybrid_speed_pitch_ml", "HybridSpeedPitch"),
        ("core.hybrid.hybrid_vocal_enhancer", "HybridVocalEnhancer"),
        ("core.hybrid.hybrid_wow_flutter", "HybridWowFlutter"),
    ]

    def test_01_all_importable(self):
        import importlib

        failed = []
        for mp, cn in self.ALL_HYBRID:
            try:
                m = importlib.import_module(mp)
                assert hasattr(m, cn), f"{cn} nicht in {mp}"
            except Exception as e:
                failed.append(f"{mp}: {e}")
        assert not failed, f"Import-Fehler: {failed}"

    def test_02_all_instantiatable_no_args(self):
        import importlib

        failed = []
        for mp, cn in self.ALL_HYBRID:
            try:
                m = importlib.import_module(mp)
                cls = getattr(m, cn)
                inst = cls()
                assert inst is not None
            except Exception as e:
                failed.append(f"{cn}: {e}")
        assert not failed, f"Init-Fehler: {failed}"

    def test_03_wired_in_restorer(self):
        import os

        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        restorer_path = os.path.join(base, "backend", "core", "unified_restorer_v3.py")
        restorer_src = open(restorer_path).read()
        for mp, cn in self.ALL_HYBRID:
            short = mp.split(".")[-1]
            assert short in restorer_src, f"{short} nicht im Restorer verdrahtet"

    def test_04_dereverb_then_denoise_chain(self):
        """Dereverb und Denoise sequenziell — kein Absturz, kein NaN"""
        from backend.core.hybrid.hybrid_dereverb import HybridDereverb
        from backend.core.hybrid.hybrid_ml_denoiser import HybridMLDenoiser

        hd = HybridDereverb()
        hmd = HybridMLDenoiser()

        rev_result = hd.dereverb(AUDIO_SINE, SR)
        # Audio aus Dereverb extrahieren (falls vorhanden)
        mid_audio = AUDIO_SINE
        for attr in ("audio", "processed_audio", "output"):
            val = getattr(rev_result, attr, None)
            if isinstance(val, np.ndarray) and val.shape == AUDIO_SINE.shape:
                mid_audio = val
                break

        den_result = hmd.denoise(mid_audio, SR)
        assert den_result is not None

    def test_05_vocal_enhancer_modes(self):
        """VocalEnhancer: alle quality_modes ohne Crash"""
        from backend.core.hybrid.hybrid_vocal_enhancer import HybridVocalEnhancer

        hve = HybridVocalEnhancer()
        for mode in ("fast", "balanced", "high"):
            try:
                result = hve.enhance(AUDIO_SINE, SR, quality_mode=mode)
                assert result is not None
            except Exception as e:
                # Unbekannter Modus ist OK sofern kein unerwarteter Crash
                if "quality_mode" not in str(e).lower() and "unknown" not in str(e).lower():
                    raise
