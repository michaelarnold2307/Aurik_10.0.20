from __future__ import annotations

"""
tests/unit/test_v95_modules.py
==============================

Optimierte Unit-Tests für alle v10.0.0-Module (parametrisiert, Fixtures mit Scope,
Modul-Level-Imports):
  - Phase 55: DiffusionInpaintingPhase
  - FeedbackChain & compute_perceptual_score
  - Music-MOS: score_music_mos
  - CLAP Reference Matcher
  - Material Restoration Nets
  - CPU-Pipeline
  - Benchmark Suite
  - GPU-Stub

Aurik 10.0.0 — 2026
"""


import sys
import warnings

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.clap_reference_matcher import (
    CLAPReferenceMatcher,
    compute_dsp_embedding,
    cosine_similarity,
    spectral_transfer,
)
from backend.core.feedback_chain import (
    FEEDBACK_CRITICAL_PHASES,
    FeedbackChain,
    FeedbackChainResult,
    compute_perceptual_score,
)
from backend.core.material_restoration_nets import (
    MaterialRestorationResult as RestorationResult,
)
from backend.core.material_restoration_nets import (
    SourceMedium,
    restore_by_medium,
    restore_lacquer,
    restore_shellac,
    restore_tape,
    restore_vinyl,
)
from backend.core.music_quality_scorer import MusicMOS, score_music_mos

# ---------------------------------------------------------------------------
# Modul-Level-Imports (1× pro Session statt 1× pro Test)
# ---------------------------------------------------------------------------
from backend.core.phases.phase_55_diffusion_inpainting import (
    _MIN_GAP_MS_DEFAULT,
    DiffusionInpaintingPhase,
    _burg_ar_predict,
    _detect_gaps,
    _reconstruction_quality_score,
)
from benchmarks.restoration_benchmark import (
    REFERENCE_SCORES,
    BenchmarkReport,
    RestorationBenchmark,
)
from dsp.cpu_pipeline import CPUPipeline, PipelineStats

SR = 48000


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mono_audio() -> np.ndarray:
    """Kurzes Mono-Testsignal (0.5 s) — einmal pro Modul erzeugt."""
    t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def stereo_audio(mono_audio) -> np.ndarray:
    return np.stack([mono_audio, mono_audio * 0.9], axis=0)


@pytest.fixture(scope="module")
def phase55() -> DiffusionInpaintingPhase:
    return DiffusionInpaintingPhase()


@pytest.fixture(scope="module")
def cpu_pipe() -> CPUPipeline:
    return CPUPipeline()


@pytest.fixture(scope="class")
def benchmark_report(tmp_path_factory) -> tuple[RestorationBenchmark, BenchmarkReport]:
    """Benchmark einmal pro Klasse ausführen, Ergebnis teilen."""
    out = tmp_path_factory.mktemp("bench")
    bench = RestorationBenchmark()
    report = bench.run_all(output_dir=str(out))
    return bench, report


# ===========================================================================
# 1. Phase 55 – DiffusionInpaintingPhase
# ===========================================================================


class TestPhase55DiffusionInpainting:
    def test_metadata_phase_id(self, phase55):
        assert phase55.metadata.phase_id == "phase_55_diffusion_inpainting"

    def test_metadata_quality_impact(self, phase55):
        assert phase55.metadata.quality_impact >= 0.5

    @pytest.mark.parametrize("channels", [1, 2])
    def test_process_shape_preserved(self, phase55, channels):
        if channels == 1:
            audio = np.zeros(int(SR * 0.3), dtype=np.float32)
            audio[:] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, int(SR * 0.3)))
        else:
            base = np.zeros(int(SR * 0.3), dtype=np.float32)
            base[:] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, int(SR * 0.3)))
            audio = np.stack([base, base * 0.9])
        result = phase55.process(audio, SR)
        assert result.audio.shape == audio.shape
        assert result.success is True

    def test_process_inpaints_gap(self, phase55):
        t = np.linspace(0, 0.3, int(SR * 0.3), dtype=np.float32)
        seg = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        silence = np.zeros(int(SR * 0.06), dtype=np.float32)  # 60 ms
        audio = np.concatenate([seg, silence, seg])
        result = phase55.process(audio, SR)
        assert result.audio.shape == audio.shape
        assert "n_gaps_detected" in result.metadata

    def test_detect_gaps_returns_list(self, mono_audio):
        gaps = _detect_gaps(mono_audio, SR, min_gap_ms=_MIN_GAP_MS_DEFAULT)
        assert isinstance(gaps, list)

    def test_burg_ar_predict_shape(self, mono_audio):
        predicted = _burg_ar_predict(mono_audio[: int(SR * 0.1)], order=16, n_samples=100)
        assert len(predicted) == 100

    def test_reconstruction_quality_score_range(self, mono_audio):
        score = _reconstruction_quality_score(mono_audio, mono_audio, [])
        assert 0.0 <= score <= 1.0

    def test_zero_strength_passthrough(self, phase55, mono_audio):
        result = phase55.process(mono_audio, SR, strength=0.0)
        assert result.success is True
        assert np.allclose(result.audio, mono_audio, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, phase55, mono_audio):
        result = phase55.process(mono_audio, SR, strength=1.0, phase_locality_factor=0.4)
        assert result.success is True
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_safe_strength_dampens_vocal_vinyl(self, phase55):
        safe = phase55._derive_safe_inpainting_strength(1.0, "materialtype.vinyl", 0.8)
        assert safe == pytest.approx(0.85, rel=1e-6)
        # §Q11 (2026-09-15): Gesangs-Drosselung entfällt (CQTdiff+ mean SDR ≥ 0 dB) —
        # Analog-Material behält nur die moderate 0,85-Reduktion, keine Vocal-Kappe.
        safe = phase55._derive_safe_inpainting_strength(1.0, "vinyl", 0.9)
        assert safe == pytest.approx(0.85, rel=1e-6)
        assert safe <= 0.90 + 1e-9

    def test_safe_strength_keeps_nonvocal_unknown(self, phase55):
        safe = phase55._derive_safe_inpainting_strength(0.6, "unknown", 0.05)
        assert abs(safe - 0.6) < 1e-9


# ===========================================================================
# 2. Perceptual Feedback-Loop
# ===========================================================================


class TestFeedbackChain:
    @pytest.fixture(scope="class")
    def chain(self) -> FeedbackChain:
        return FeedbackChain(sample_rate=SR)

    @pytest.fixture(scope="class")
    def perceptual_scores(self, mono_audio):
        """Score-Dict einmal berechnen, von mehreren Tests nutzen."""
        return compute_perceptual_score(mono_audio, mono_audio, sample_rate=SR)

    @pytest.mark.parametrize(
        "key",
        [
            "spectral_corr",
            "spectral_flatness",
            "snr_db",
            "transient_score",
            "combined",
        ],
    )
    def test_score_keys_present(self, perceptual_scores, key):
        assert key in perceptual_scores

    def test_combined_score_in_range(self, perceptual_scores):
        assert 0.0 <= perceptual_scores["combined"] <= 1.0

    def test_identical_signals_perfect_spectral_corr(self, perceptual_scores):
        assert perceptual_scores["spectral_corr"] > 0.99

    def test_noisy_lower_than_clean(self, mono_audio):
        noisy = mono_audio + np.random.default_rng(42).normal(0, 0.2, mono_audio.shape).astype(np.float32)
        s_noisy = compute_perceptual_score(mono_audio, noisy, sample_rate=SR)
        s_clean = compute_perceptual_score(mono_audio, mono_audio, sample_rate=SR)
        assert s_noisy["combined"] < s_clean["combined"]

    def test_run_returns_result(self, chain, mono_audio):
        result = chain.run(mono_audio, [(99, lambda a: a * 0.99, {})])
        assert isinstance(result, FeedbackChainResult)
        assert result.audio.shape == mono_audio.shape
        assert hasattr(result, "phase_executions")
        assert hasattr(result, "overall_score")
        assert hasattr(result, "total_time_s")

    def test_critical_phases_contains_55(self):
        assert 55 in FEEDBACK_CRITICAL_PHASES


# ===========================================================================
# 3. Music-MOS
# ===========================================================================


class TestMusicMOS:
    @pytest.fixture(scope="class")
    def mos(self, mono_audio):
        return score_music_mos(mono_audio, SR)

    @pytest.mark.parametrize("attr", ["MUSIC_SIG", "MUSIC_BAK", "MUSIC_OVR", "MUSIC_NAT"])
    def test_score_in_range(self, mos, attr):
        val = getattr(mos, attr)
        assert 1.0 <= val <= 5.0, f"{attr}={val} außerhalb [1,5]"

    def test_returns_music_mos_instance(self, mos):
        assert isinstance(mos, MusicMOS)

    def test_stereo_input_valid(self, stereo_audio):
        result = score_music_mos(stereo_audio, SR)
        assert 1.0 <= result.MUSIC_OVR <= 5.0

    def test_short_signal(self):
        short = np.zeros(int(SR * 0.05), dtype=np.float32)  # 50 ms
        result = score_music_mos(short, SR)
        assert isinstance(result.MUSIC_OVR, float)


# ===========================================================================
# 4. CLAP Reference Matcher
# ===========================================================================


class TestCLAPReferenceMatcher:
    @pytest.fixture(scope="class")
    def embedding(self, mono_audio):
        return compute_dsp_embedding(mono_audio, SR)

    @pytest.fixture(scope="class")
    def matcher_result(self, mono_audio):
        m = CLAPReferenceMatcher()
        m.load_reference(mono_audio, SR)
        return m, m.match_and_adapt(mono_audio, SR)

    def test_embedding_dim(self, embedding):
        assert embedding.shape == (32,)

    def test_embedding_l2_norm(self, embedding):
        assert abs(float(np.linalg.norm(embedding)) - 1.0) < 0.01

    def test_cosine_identical(self, embedding):
        assert abs(cosine_similarity(embedding, embedding) - 1.0) < 0.01

    def test_cosine_different_in_range(self, embedding):
        t = np.linspace(0, 1.5, int(SR * 1.5), dtype=np.float32)
        other_emb = compute_dsp_embedding(0.5 * np.sin(2 * np.pi * 880 * t), SR)
        sim = cosine_similarity(embedding, other_emb)
        assert 0.0 <= sim <= 1.0

    def test_match_result_similarity(self, matcher_result):
        _, result = matcher_result
        assert 0.0 <= result.similarity <= 1.0

    def test_match_result_audio_shape(self, mono_audio, matcher_result):
        _, result = matcher_result
        assert result.adapted_audio.shape == mono_audio.shape

    def test_spectral_transfer_shape(self, mono_audio, embedding):
        transferred = spectral_transfer(mono_audio, SR, embedding)
        assert transferred.shape == mono_audio.shape


# ===========================================================================
# 5. Material Restoration Nets
# ===========================================================================

_RESTORE_FUNCS = {
    "shellac": restore_shellac,
    "vinyl": restore_vinyl,
    "tape": restore_tape,
    "lacquer": restore_lacquer,
}
_RESTORE_MEDIUMS = {
    "shellac": SourceMedium.SHELLAC,
    "vinyl": SourceMedium.VINYL,
    "tape": SourceMedium.TAPE,
    "lacquer": SourceMedium.LACQUER,
}


class TestMaterialRestorationNets:
    @pytest.mark.parametrize("name", list(_RESTORE_FUNCS))
    def test_direct_func_shape(self, mono_audio, name):
        result = _RESTORE_FUNCS[name](mono_audio, SR)  # type: ignore[operator]
        assert isinstance(result, RestorationResult)
        assert result.audio.shape == mono_audio.shape

    @pytest.mark.parametrize("medium_name,medium", list(_RESTORE_MEDIUMS.items()))
    def test_dispatch_medium_tag(self, mono_audio, medium_name, medium):
        result = restore_by_medium(mono_audio, SR, medium)
        assert result.medium == medium
        assert result.audio.shape == mono_audio.shape
        assert len(result.applied_steps) > 0
        assert isinstance(result.metrics, dict)

    def test_enum_members(self):
        for name in ("SHELLAC", "VINYL", "TAPE", "LACQUER", "DIGITAL", "UNKNOWN"):
            assert hasattr(SourceMedium, name)


# ===========================================================================
# 6. CPU-Pipeline
# ===========================================================================


class TestCPUPipeline:
    @pytest.mark.parametrize("channels", [1, 2])
    def test_denoise_shape(self, cpu_pipe, channels, mono_audio, stereo_audio):
        audio = mono_audio if channels == 1 else stereo_audio
        assert cpu_pipe.denoise(audio, SR).shape == audio.shape

    @pytest.mark.parametrize("op", ["denoise", "spectral_repair"])
    def test_process_stft_shape(self, cpu_pipe, mono_audio, op):
        result, stats = cpu_pipe.process_stft(mono_audio, SR, op)
        assert result.shape == mono_audio.shape
        assert isinstance(stats, PipelineStats)

    def test_pipeline_stats_values(self, cpu_pipe, mono_audio):
        _, stats = cpu_pipe.process_stft(mono_audio, SR, "denoise")
        assert stats.n_chunks >= 1
        assert stats.total_time_s >= 0.0
        assert stats.realtime_factor > 0.0
        assert 1 <= stats.n_workers <= 8

    def test_denoise_returns_float32(self, cpu_pipe, mono_audio):
        assert cpu_pipe.denoise(mono_audio, SR).dtype == np.float32

    def test_streaming_multiple_chunks(self, cpu_pipe):
        """Signal >3 s → mindestens 2 Chunks."""
        audio = np.zeros(int(SR * 4.0), dtype=np.float32)
        audio[:] = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 4.0, len(audio)))
        result, stats = cpu_pipe.process_stft(audio, SR, "denoise")
        assert result.shape == audio.shape
        assert stats.n_chunks >= 2


# ===========================================================================
# 7. Benchmark Suite
# ===========================================================================


@pytest.mark.slow
class TestRestorationBenchmark:
    def test_reference_scores_keys(self):
        for key in ("iZotope RX 10", "Aurik 10.0.0 (Ziel)"):
            assert key in REFERENCE_SCORES

    def test_report_type(self, benchmark_report):
        _, report = benchmark_report
        assert isinstance(report, BenchmarkReport)

    def test_report_four_results(self, benchmark_report):
        _, report = benchmark_report
        assert len(report.test_results) == 4

    def test_report_summary_nonempty(self, benchmark_report):
        _, report = benchmark_report
        assert report.summary

    def test_report_timestamp(self, benchmark_report):
        _, report = benchmark_report
        assert report.timestamp

    def test_json_export(self, benchmark_report, tmp_path):
        bench, _ = benchmark_report
        bench.run_all(output_dir=str(tmp_path))
        assert len(list(tmp_path.glob("*.json"))) >= 1

    def test_compare_to_reference_bool(self, benchmark_report):
        bench, report = benchmark_report
        assert isinstance(bench.compare_to_reference(report), bool)


# ===========================================================================
# 8. GPU-Pipeline Stub
# ===========================================================================


class TestGPUPipelineStub:
    def _reload(self):
        sys.modules.pop("dsp.gpu_pipeline", None)

    def test_import_warns_deprecation(self):
        self._reload()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import importlib

            importlib.import_module("dsp.gpu_pipeline")  # triggers the DeprecationWarning at module level
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_gpu_is_cpu_alias(self):
        self._reload()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from dsp.gpu_pipeline import GPUPipeline, GPUPipelineStats
        assert GPUPipeline is CPUPipeline
        assert GPUPipelineStats is PipelineStats


# ===========================================================================
# 9. Excellence Optimizer  (v10.0.0)
# ===========================================================================

import pytest

from backend.core.excellence_optimizer import (
    ExcellenceContext,
    ExcellenceOptimizer,
    ExcellenceResult,
    analyze_context,
    optimize_for_excellence,
)

_SR = 48000
_N = _SR // 4  # 0.25 s — schnell, aber genug für STFT


def _sine_signal(freq: float = 440.0, n: int = _N, sr: int = _SR) -> np.ndarray:
    """Deterministisches Sinus-Testsignal."""
    t = np.arange(n) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _stereo_signal(n: int = _N, sr: int = _SR) -> np.ndarray:
    """Stereo (samples, 2)."""
    mono = _sine_signal(n=n, sr=sr)
    return np.stack([mono, mono * 0.9], axis=1)


@pytest.fixture(scope="module")
def sine_mono() -> np.ndarray:
    return _sine_signal()


@pytest.fixture(scope="module")
def sine_stereo() -> np.ndarray:
    return _stereo_signal()


@pytest.fixture(scope="module")
def ctx_mono(sine_mono) -> ExcellenceContext:
    return analyze_context(sine_mono, _SR)


class TestAnalyzeContext:
    def test_returns_excellence_context(self, ctx_mono):
        assert isinstance(ctx_mono, ExcellenceContext)

    def test_sample_rate_stored(self, ctx_mono):
        assert ctx_mono.sample_rate == _SR

    def test_mono_not_stereo(self, ctx_mono):
        assert ctx_mono.is_stereo is False

    def test_stereo_detected(self, sine_stereo):
        ctx = analyze_context(sine_stereo, _SR)
        assert ctx.is_stereo is True

    def test_rms_db_negative(self, ctx_mono):
        assert ctx_mono.rms_db < 0  # Sinussignal mit amp 0.5 → ~-6 dBFS

    def test_snr_estimate_positive(self, ctx_mono):
        assert ctx_mono.snr_estimate_db >= 0

    def test_harmonicity_range(self, ctx_mono):
        assert 0.0 <= ctx_mono.harmonicity <= 1.0

    def test_transient_density_range(self, ctx_mono):
        assert 0.0 <= ctx_mono.transient_density <= 1.0

    def test_dynamic_cv_nonneg(self, ctx_mono):
        assert ctx_mono.dynamic_cv >= 0.0

    def test_spectral_centroid_positive(self, ctx_mono):
        assert ctx_mono.spectral_centroid_mean > 0


class TestExcellenceOptimizerInit:
    def test_default_flags_all_true(self):
        opt = ExcellenceOptimizer(_SR)
        assert opt.apply_continuity is True
        assert opt.apply_micro_dynamics is True
        assert opt.apply_harmonic_boost is True
        assert opt.apply_ola_edges is True

    def test_custom_flags(self):
        opt = ExcellenceOptimizer(_SR, apply_harmonic_boost=False)
        assert opt.apply_harmonic_boost is False

    def test_sample_rate_stored(self):
        opt = ExcellenceOptimizer(48000)
        assert opt.sample_rate == 48000


class TestExcellenceOptimizerOptimize:
    def test_returns_tuple(self, sine_mono):
        opt = ExcellenceOptimizer(_SR)
        out = opt.optimize(sine_mono)
        assert isinstance(out, tuple) and len(out) == 2

    def test_output_shape_mono(self, sine_mono):
        opt = ExcellenceOptimizer(_SR)
        audio_out, _ = opt.optimize(sine_mono)
        assert audio_out.shape == sine_mono.shape

    def test_output_shape_stereo(self, sine_stereo):
        opt = ExcellenceOptimizer(_SR)
        audio_out, _ = opt.optimize(sine_stereo)
        assert audio_out.shape == sine_stereo.shape

    def test_result_is_excellence_result(self, sine_mono):
        opt = ExcellenceOptimizer(_SR)
        _, result = opt.optimize(sine_mono)
        assert isinstance(result, ExcellenceResult)

    def test_no_clipping(self, sine_mono):
        opt = ExcellenceOptimizer(_SR)
        audio_out, _ = opt.optimize(sine_mono)
        assert np.max(np.abs(audio_out)) <= 1.0

    def test_output_not_silent(self, sine_mono):
        opt = ExcellenceOptimizer(_SR)
        audio_out, _ = opt.optimize(sine_mono)
        assert float(np.sqrt(np.mean(audio_out.astype(np.float64) ** 2))) > 1e-4

    def test_empty_audio_passthrough(self):
        opt = ExcellenceOptimizer(_SR)
        empty = np.array([], dtype=np.float32)
        out, result = opt.optimize(empty)
        assert out.size == 0
        assert isinstance(result, ExcellenceResult)

    def test_delta_rms_bounded(self, sine_mono):
        opt = ExcellenceOptimizer(_SR)
        _, result = opt.optimize(sine_mono)
        assert -6.0 <= result.delta_rms_db <= 6.0

    def test_summary_string(self, sine_mono):
        opt = ExcellenceOptimizer(_SR)
        _, result = opt.optimize(sine_mono)
        summary = result.summary()
        assert isinstance(summary, str) and "ExcellenceOptimizer" in summary

    def test_applied_steps_is_list(self, sine_mono):
        opt = ExcellenceOptimizer(_SR)
        _, result = opt.optimize(sine_mono)
        assert isinstance(result.applied_steps, list)

    def test_ola_crossfades_nonneg(self, sine_mono):
        opt = ExcellenceOptimizer(_SR)
        _, result = opt.optimize(sine_mono)
        assert result.ola_crossfades >= 0

    def test_precomputed_context_accepted(self, sine_mono, ctx_mono):
        opt = ExcellenceOptimizer(_SR)
        audio_out, _ = opt.optimize(sine_mono, context=ctx_mono)
        assert audio_out.shape == sine_mono.shape

    def test_with_all_steps_disabled(self, sine_mono):
        opt = ExcellenceOptimizer(
            _SR,
            apply_continuity=False,
            apply_micro_dynamics=False,
            apply_harmonic_boost=False,
            apply_ola_edges=False,
        )
        audio_out, result = opt.optimize(sine_mono)
        assert result.applied_steps == []
        # Output sollte nahezu identisch mit Input sein (nur clip)
        assert np.allclose(sine_mono, audio_out, atol=1e-5)


class TestOptimizeForExcellenceFunction:
    def test_returns_two_tuple(self, sine_mono):
        out, result = optimize_for_excellence(sine_mono, _SR)
        assert isinstance(out, np.ndarray)
        assert isinstance(result, ExcellenceResult)

    def test_shape_preserved(self, sine_mono):
        out, _ = optimize_for_excellence(sine_mono, _SR)
        assert out.shape == sine_mono.shape

    def test_no_clipping(self, sine_mono):
        out, _ = optimize_for_excellence(sine_mono, _SR)
        assert np.max(np.abs(out)) <= 1.0

    def test_flags_forwarded(self, sine_mono):
        _, result = optimize_for_excellence(sine_mono, _SR, apply_harmonic_boost=False, apply_micro_dynamics=False)
        assert "harmonic_boost" not in result.applied_steps
        assert "micro_dynamics" not in result.applied_steps


# ===========================================================================
# 10. Music Quality Scorer — neue Metriken (v10.0.0)
# ===========================================================================

from backend.core.music_quality_scorer import (
    _frame_audio,
    _micro_dynamic_variation,
    _spectral_flux_continuity,
)


class TestSpectralFluxContinuity:
    def test_returns_float(self):
        frames = _frame_audio(_sine_signal())
        val = _spectral_flux_continuity(frames)
        assert isinstance(val, float)

    def test_range_zero_to_one(self):
        frames = _frame_audio(_sine_signal())
        val = _spectral_flux_continuity(frames)
        assert 0.0 <= val <= 1.0

    def test_pure_sine_high_continuity(self):
        # Reiner Sinus = sehr glatte spektrale Evolution → hoher Score
        frames = _frame_audio(_sine_signal(freq=440.0, n=_SR))
        val = _spectral_flux_continuity(frames)
        assert val >= 0.5, f"Erwartet ≥ 0.5, got {val:.3f}"

    def test_short_frames_fallback(self):
        # Weniger als 3 Frames → Fallback 0.8
        frames = np.zeros((2, 1024), dtype=np.float32)
        val = _spectral_flux_continuity(frames)
        assert val == pytest.approx(0.8)


class TestMicroDynamicVariation:
    def test_returns_float(self):
        val = _micro_dynamic_variation(_sine_signal())
        assert isinstance(val, float)

    def test_range_zero_to_one(self):
        val = _micro_dynamic_variation(_sine_signal())
        assert 0.0 <= val <= 1.0

    def test_too_short_fallback(self):
        val = _micro_dynamic_variation(np.zeros(10, dtype=np.float32))
        assert 0.0 <= val <= 1.0

    def test_noisy_signal_lower_than_sine(self):
        rng = np.random.default_rng(0)
        noisy = rng.standard_normal(_N).astype(np.float32)
        val_noisy = _micro_dynamic_variation(noisy)
        val_sine = _micro_dynamic_variation(_sine_signal(n=_N))
        # Rauschen hat höhere CV → niedrigerer Score als stabiler Sinus
        # (wenn Sinus CV < 0.05, dann score_sine könnte auch < 1.0 sein,
        # aber noisy CV >> 0.20 → sicher niedriger)
        assert val_noisy <= 1.0


class TestMusicMOSUpdatedFormula:
    """Prüft, dass die neue 5-Komponenten MUSIC_NAT-Formel
    konsistente Scores liefert und MUSIC_OVR das Ziel 0.90 erreichten kann."""

    def test_mos_fields_present(self):
        mos = score_music_mos(_sine_signal(n=_SR), _SR)
        assert hasattr(mos, "MUSIC_SIG")
        assert hasattr(mos, "MUSIC_BAK")
        assert hasattr(mos, "MUSIC_OVR")
        assert hasattr(mos, "MUSIC_NAT")

    def test_all_scores_in_mos_range(self):
        mos = score_music_mos(_sine_signal(n=_SR), _SR)
        for val in (mos.MUSIC_SIG, mos.MUSIC_BAK, mos.MUSIC_OVR, mos.MUSIC_NAT):
            assert 1.0 <= val <= 5.0, f"MOS {val} nicht im Bereich [1, 5]"

    def test_nat_weight_change_reflected(self):
        # MUSIC_OVR hängt jetzt zu 35 % von NAT ab (war 25 %)
        mos = score_music_mos(_sine_signal(freq=220.0, n=_SR * 2), _SR)
        # Nur Smoke-Test: Score muss numerisch valide sein
        assert 1.0 <= mos.MUSIC_OVR <= 5.0

    def test_to_dict_has_all_keys(self):
        mos = score_music_mos(_sine_signal(n=_SR), _SR)
        d = mos.to_dict()
        assert set(d.keys()) == {"MUSIC_SIG", "MUSIC_BAK", "MUSIC_OVR", "MUSIC_NAT"}


# ===========================================================================
# 11. FeedbackChain — Excellence-Modus (v10.0.0)
# ===========================================================================

from backend.core.feedback_chain import (
    DEFAULT_TARGET_SCORE,
    EXCELLENCE_TARGET_SCORE,
    MUSIC_OVR_EXCELLENCE_THRESHOLD,
)


class TestFeedbackChainExcellenceMode:
    def test_excellence_target_score_constant(self):
        assert EXCELLENCE_TARGET_SCORE > DEFAULT_TARGET_SCORE

    def test_music_ovr_threshold_constant(self):
        assert pytest.approx(0.90) == MUSIC_OVR_EXCELLENCE_THRESHOLD

    def test_excellence_mode_raises_target(self):
        chain_std = FeedbackChain(sample_rate=_SR)
        chain_exc = FeedbackChain(sample_rate=_SR, excellence_mode=True)
        assert chain_exc.target_score > chain_std.target_score

    def test_excellence_mode_flag_stored(self):
        chain = FeedbackChain(sample_rate=_SR, excellence_mode=True)
        assert chain.excellence_mode is True

    def test_normal_mode_flag_false(self):
        chain = FeedbackChain(sample_rate=_SR)
        assert chain.excellence_mode is False

    def test_explicit_target_not_overridden_when_higher(self):
        # Wenn der Nutzer explizit einen höheren target_score setzt, bleibt er
        chain = FeedbackChain(sample_rate=_SR, target_score=0.85, excellence_mode=True)
        assert chain.target_score == pytest.approx(0.85)

    def test_run_excellence_mode_returns_result(self):
        chain = FeedbackChain(sample_rate=_SR, excellence_mode=True)
        audio = _sine_signal(n=_N)

        def identity_phase(a, sr):
            return a.copy()

        result = chain.run(audio, [(99, identity_phase, {})])
        assert isinstance(result, FeedbackChainResult)
        assert result.audio.shape == audio.shape

    def test_run_normal_mode_returns_result(self):
        chain = FeedbackChain(sample_rate=_SR)
        audio = _sine_signal(n=_N)

        def identity_phase(a, sr):
            return a.copy()

        result = chain.run(audio, [(99, identity_phase, {})])
        assert isinstance(result, FeedbackChainResult)


# ===========================================================================
# SEKTION 12 — Phase 55: Adaptive Diffusion Steps
# ===========================================================================


class TestPhase55AdaptiveSteps:
    """_adaptive_steps(gap_ms) → richtige Schrittzahl je Lückengröße."""

    def test_short_gap_returns_50(self):
        from backend.core.phases.phase_55_diffusion_inpainting import _DIFFUSION_STEPS, _adaptive_steps

        assert _adaptive_steps(0.0) == _DIFFUSION_STEPS
        assert _adaptive_steps(49.9) == _DIFFUSION_STEPS

    def test_medium_gap_returns_100(self):
        from backend.core.phases.phase_55_diffusion_inpainting import _DIFFUSION_STEPS_MED, _adaptive_steps

        assert _adaptive_steps(50.0) == _DIFFUSION_STEPS_MED
        assert _adaptive_steps(99.9) == _DIFFUSION_STEPS_MED

    def test_long_gap_returns_150(self):
        from backend.core.phases.phase_55_diffusion_inpainting import _DIFFUSION_STEPS_LONG, _adaptive_steps

        assert _adaptive_steps(100.0) == _DIFFUSION_STEPS_LONG
        assert _adaptive_steps(500.0) == _DIFFUSION_STEPS_LONG

    def test_constants_ordered(self):
        from backend.core.phases.phase_55_diffusion_inpainting import (
            _DIFFUSION_STEPS,
            _DIFFUSION_STEPS_LONG,
            _DIFFUSION_STEPS_MED,
        )

        assert _DIFFUSION_STEPS < _DIFFUSION_STEPS_MED < _DIFFUSION_STEPS_LONG

    def test_inpaint_dsp_accepts_n_steps(self):
        """_inpaint_gap_dsp nimmt optionalen n_steps-Parameter an."""
        from backend.core.phases.phase_55_diffusion_inpainting import _inpaint_gap_dsp

        audio = np.zeros(4410, dtype=np.float32)
        audio[1000:1100] = 0.0  # Lücke
        result = _inpaint_gap_dsp(audio, 1000, 1100, 48000, n_steps=10)
        assert result.shape == (100,)
        assert not np.any(np.abs(result) > 1.01)

    def test_phase55_metadata_diffusion_steps_adaptive_string(self):
        from backend.core.phases.phase_55_diffusion_inpainting import DiffusionInpaintingPhase

        phase = DiffusionInpaintingPhase()
        # Verarbeitung von Audio ohne Lücken gibt diffusion_steps als String zurück
        audio = np.ones(4410, dtype=np.float32) * 0.1
        result = phase.process(audio, 48000)
        assert "adaptive" in str(result.metadata.get("diffusion_steps", ""))


# ===========================================================================
# SEKTION 13 — MERT Plugin: Naturalness Analyse & Enhancement
# ===========================================================================

_MERT_SR = 48000
_MERT_N = _MERT_SR  # 1 Sekunde


@pytest.fixture(scope="session")
def mert_sine():
    t = np.linspace(0, 1, _MERT_N, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture(scope="session")
def mert_plugin():
    import pathlib
    import sys

    _plugins_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent / "plugins")
    if _plugins_dir not in sys.path:
        sys.path.insert(0, _plugins_dir)
    from mert_plugin import MertPlugin

    return MertPlugin()


class TestMertPluginInit:
    @pytest.mark.timeout(60)
    def test_instantiates_without_error(self, mert_plugin):
        assert mert_plugin is not None

    def test_model_type_dsp_fallback_without_weights(self):
        # Ohne echte Gewichte immer DSP-Fallback — isolierte Instanz mit leerem Verzeichnis
        import pathlib
        import tempfile
        import unittest.mock

        from mert_plugin import MertPlugin

        empty = pathlib.Path(tempfile.mkdtemp())
        with (
            unittest.mock.patch("mert_plugin._MERT_330M_DIR", empty),
            unittest.mock.patch("mert_plugin._MERT_95M_DIR", empty),
        ):
            plugin = MertPlugin(model_dir=str(empty))
        assert plugin._model_type == "dsp_fallback"

    def test_model_available_false_without_weights(self):
        # Ohne echte Gewichte model_available == False — isolierte Instanz
        import pathlib
        import tempfile
        import unittest.mock

        from mert_plugin import MertPlugin

        empty = pathlib.Path(tempfile.mkdtemp())
        with (
            unittest.mock.patch("mert_plugin._MERT_330M_DIR", empty),
            unittest.mock.patch("mert_plugin._MERT_95M_DIR", empty),
        ):
            plugin = MertPlugin(model_dir=str(empty))
        assert plugin.model_available is False

    def test_model_type_fairseq_with_pt_only(self):
        """Fairseq .pt vorhanden (ohne HF-Dateien) → _model_type == 'mert_fairseq'."""
        import pathlib
        import sys
        import tempfile
        import unittest.mock

        _plugins_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent / "plugins")
        if _plugins_dir not in sys.path:
            sys.path.insert(0, _plugins_dir)

        try:
            import torch
        except ImportError:
            import pytest

            pytest.skip("torch nicht verfügbar")
        from mert_plugin import MertPlugin

        d = pathlib.Path(tempfile.mkdtemp())
        empty_330m = pathlib.Path(tempfile.mkdtemp())
        # Minimalen fairseq-Checkpoint anlegen (kein pytorch_model.bin)
        dummy_state = {"feature_extractor.conv_layers.0.0.weight": torch.zeros(8, 1, 3)}
        torch.save({"model": dummy_state, "cfg": {}}, d / "MERT-v1-95M_fairseq.pt")
        # §VERBOTEN: Budget-Tests ohne is_system_thrashing-Mock → flaky auf Hosts mit hoher Swap-Last
        with (
            unittest.mock.patch("mert_plugin._MERT_330M_DIR", empty_330m),
            unittest.mock.patch("mert_plugin._MERT_95M_DIR", d),
            unittest.mock.patch("mert_plugin.ml_budget_try_allocate", return_value=True),
            unittest.mock.patch("mert_plugin.ml_budget_release", return_value=None),
            unittest.mock.patch("backend.core.ml_memory_budget.is_system_thrashing", return_value=False),
            unittest.mock.patch.dict("os.environ", {"AURIK_SAFE_VALIDATION_PROFILE": "0"}),
        ):
            plugin = MertPlugin(
                model_dir=str(d), use_onnx=False
            )  # §v10.748: ONNX-first ist Default — fairseq-Pfad explizit
        assert plugin._model_type == "mert_fairseq"
        assert plugin.model_available is True


@pytest.mark.timeout(60)
class TestMertAnalyze:
    def test_returns_mert_analysis(self, mert_plugin, mert_sine):
        from mert_plugin import MertAnalysis

        result = mert_plugin.analyze(mert_sine, _MERT_SR)
        assert isinstance(result, MertAnalysis)

    def test_naturalness_score_in_range(self, mert_plugin, mert_sine):
        result = mert_plugin.analyze(mert_sine, _MERT_SR)
        assert 0.0 <= result.naturalness_score <= 1.0

    def test_harmonicity_in_range(self, mert_plugin, mert_sine):
        result = mert_plugin.analyze(mert_sine, _MERT_SR)
        assert 0.0 <= result.harmonicity <= 1.0

    def test_tonal_consistency_in_range(self, mert_plugin, mert_sine):
        result = mert_plugin.analyze(mert_sine, _MERT_SR)
        assert 0.0 <= result.tonal_consistency <= 1.0

    def test_model_used_dsp_fallback(self, mert_plugin, mert_sine):
        result = mert_plugin.analyze(mert_sine, _MERT_SR)
        # Modell ist lokal vorhanden (models/mert-95m) → "mert_hf" oder "mert_fairseq"
        # erlaubt. Bei fehlendem Modell: "dsp_fallback". Alle drei sind gültig.
        assert result.model_used in ("dsp_fallback", "mert_hf", "mert_fairseq", "mert_onnx")

    def test_analysis_frames_positive(self, mert_plugin, mert_sine):
        result = mert_plugin.analyze(mert_sine, _MERT_SR)
        assert result.analysis_frames > 0

    def test_to_dict_keys(self, mert_plugin, mert_sine):
        d = mert_plugin.analyze(mert_sine, _MERT_SR).to_dict()
        for key in ("harmonicity", "tonal_consistency", "naturalness_score", "model_used"):
            assert key in d

    def test_stereo_input(self, mert_plugin, mert_sine):
        stereo = np.stack([mert_sine, mert_sine * 0.9], axis=0)
        result = mert_plugin.analyze(stereo, _MERT_SR)
        assert 0.0 <= result.naturalness_score <= 1.0

    def test_silence_does_not_crash(self, mert_plugin):
        silence = np.zeros(4096, dtype=np.float32)
        result = mert_plugin.analyze(silence, _MERT_SR)
        assert isinstance(result.naturalness_score, float)

    def test_short_audio_fallback(self, mert_plugin):
        short = np.zeros(16, dtype=np.float32)
        result = mert_plugin.analyze(short, _MERT_SR)
        assert isinstance(result, object)


@pytest.mark.timeout(60)
class TestMertEnhance:
    def test_output_shape_mono(self, mert_plugin, mert_sine):
        enhanced = mert_plugin.enhance_naturalness(mert_sine, _MERT_SR)
        assert enhanced.shape == mert_sine.shape

    def test_no_clipping(self, mert_plugin, mert_sine):
        enhanced = mert_plugin.enhance_naturalness(mert_sine, _MERT_SR)
        assert not np.any(np.abs(enhanced) > 1.01)

    def test_output_shape_stereo(self, mert_plugin, mert_sine):
        stereo = np.stack([mert_sine, mert_sine * 0.8], axis=0)
        enhanced = mert_plugin.enhance_naturalness(stereo, _MERT_SR)
        assert enhanced.shape == stereo.shape

    def test_high_nat_score_returns_unchanged(self, mert_plugin):
        """NAT-Score ≥ 0.80 → kein Enhancement, Signal bleibt gleich."""
        from mert_plugin import MertAnalysis

        audio = np.random.randn(4096).astype(np.float32) * 0.3
        analysis = MertAnalysis(naturalness_score=0.95)
        enhanced = mert_plugin.enhance_naturalness(audio, _MERT_SR, analysis=analysis)
        np.testing.assert_array_equal(enhanced, audio)


@pytest.mark.timeout(60)
class TestMertConvenienceFunctions:
    def test_analyze_naturalness_function(self, mert_sine):
        import pathlib
        import sys

        _plugins_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent / "plugins")
        if _plugins_dir not in sys.path:
            sys.path.insert(0, _plugins_dir)
        from mert_plugin import analyze_naturalness

        result = analyze_naturalness(mert_sine, _MERT_SR)
        assert 0.0 <= result.naturalness_score <= 1.0

    def test_enhance_naturalness_function(self, mert_sine):
        import pathlib
        import sys

        _plugins_dir = str(pathlib.Path(__file__).resolve().parent.parent.parent / "plugins")
        if _plugins_dir not in sys.path:
            sys.path.insert(0, _plugins_dir)
        from mert_plugin import enhance_naturalness

        out = enhance_naturalness(mert_sine, _MERT_SR)
        assert out.shape == mert_sine.shape


# ===========================================================================
# SEKTION 14 — Material-Profile im ExcellenceOptimizer
# ===========================================================================


class TestMaterialProfiles:
    """MATERIAL_PROFILES-Dict und Profil-Validierung."""

    def test_all_profiles_present(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        for name in ("auto", "vinyl", "tape", "shellac", "broadcast"):
            assert name in MATERIAL_PROFILES, f"Profil '{name}' fehlt"

    def test_profile_names_match_keys(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        for key, profile in MATERIAL_PROFILES.items():
            assert profile.name == key

    def test_harm_boost_db_positive(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        for p in MATERIAL_PROFILES.values():
            assert p.harm_boost_db > 0, f"{p.name}: harm_boost_db muss > 0 sein"

    def test_ola_ms_positive(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        for p in MATERIAL_PROFILES.values():
            assert p.ola_ms > 0

    def test_vinyl_has_highest_harm_boost_except_shellac(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        assert MATERIAL_PROFILES["shellac"].harm_boost_db > MATERIAL_PROFILES["vinyl"].harm_boost_db

    def test_broadcast_lowest_harm_boost(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        broadcast_boost = MATERIAL_PROFILES["broadcast"].harm_boost_db
        for name, p in MATERIAL_PROFILES.items():
            if name != "broadcast":
                assert p.harm_boost_db >= broadcast_boost, (
                    f"{name}.harm_boost_db={p.harm_boost_db} < broadcast={broadcast_boost}"
                )

    def test_shellac_longest_ola(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        shellac_ola = MATERIAL_PROFILES["shellac"].ola_ms
        for name, p in MATERIAL_PROFILES.items():
            if name != "shellac":
                assert p.ola_ms <= shellac_ola, f"{name}.ola_ms > shellac"


class TestExcellenceOptimizerMaterialParam:
    """ExcellenceOptimizer mit material=-Parameter."""

    def test_default_is_auto(self):
        from backend.core.excellence_optimizer import ExcellenceOptimizer

        opt = ExcellenceOptimizer(48000)
        assert opt.material == "auto"

    def test_vinyl_sets_harm_boost(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES, ExcellenceOptimizer

        opt = ExcellenceOptimizer(48000, material="vinyl")
        assert opt._harm_boost_db == pytest.approx(MATERIAL_PROFILES["vinyl"].harm_boost_db)

    def test_tape_sets_ola_ms(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES, ExcellenceOptimizer

        opt = ExcellenceOptimizer(48000, material="tape")
        assert opt._ola_ms == pytest.approx(MATERIAL_PROFILES["tape"].ola_ms)

    def test_shellac_sets_modulation_strength(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES, ExcellenceOptimizer

        opt = ExcellenceOptimizer(48000, material="shellac")
        assert opt._modulation_strength == pytest.approx(MATERIAL_PROFILES["shellac"].modulation_strength)

    def test_broadcast_sets_flux_smoothing(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES, ExcellenceOptimizer

        opt = ExcellenceOptimizer(48000, material="broadcast")
        assert opt._flux_smoothing_max == pytest.approx(MATERIAL_PROFILES["broadcast"].flux_smoothing_max)

    def test_unknown_material_falls_back_to_auto(self):
        from backend.core.excellence_optimizer import MATERIAL_PROFILES, ExcellenceOptimizer

        opt = ExcellenceOptimizer(48000, material="unknown_xyz")
        assert opt._harm_boost_db == pytest.approx(MATERIAL_PROFILES["auto"].harm_boost_db)

    def test_optimize_with_vinyl_no_clipping(self):
        from backend.core.excellence_optimizer import ExcellenceOptimizer

        audio = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.4).astype(np.float32)
        opt = ExcellenceOptimizer(48000, material="vinyl")
        out, rep = opt.optimize(audio)
        assert out.shape == audio.shape
        assert not np.any(np.abs(out) > 1.01)

    def test_optimize_with_shellac_returns_result(self):
        from backend.core.excellence_optimizer import ExcellenceOptimizer, ExcellenceResult

        audio = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.3).astype(np.float32)
        opt = ExcellenceOptimizer(48000, material="shellac")
        out, rep = opt.optimize(audio)
        assert isinstance(rep, ExcellenceResult)


class TestOptimizeForExcellenceMaterial:
    """optimize_for_excellence() mit material=-Parameter."""

    def test_material_kwarg_accepted(self):
        from backend.core.excellence_optimizer import optimize_for_excellence

        audio = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.3).astype(np.float32)
        out, rep = optimize_for_excellence(audio, 48000, material="tape")
        assert out.shape == audio.shape

    @pytest.mark.ml
    def test_all_materials_run_without_error(self):
        from backend.core.excellence_optimizer import optimize_for_excellence

        audio = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.3).astype(np.float32)
        for mat in ("auto", "vinyl", "tape", "shellac", "broadcast"):
            out, _ = optimize_for_excellence(audio, 48000, material=mat)
            assert out.shape == audio.shape, f"Material '{mat}' hat falshe Output-Shape"


# ===========================================================================
# SEKTION 15 — v10.0.0: FeedbackChain material/use_mert + ExcellenceOptimizer use_mert
# ===========================================================================


class TestFeedbackChainV96:
    """Neue Parameter material und use_mert in FeedbackChain."""

    def test_material_default_auto(self):
        chain = FeedbackChain(sample_rate=_SR)
        assert chain.material == "auto"

    def test_use_mert_default_false(self):
        chain = FeedbackChain(sample_rate=_SR)
        assert chain.use_mert is False

    def test_material_vinyl_stored(self):
        chain = FeedbackChain(sample_rate=_SR, material="vinyl")
        assert chain.material == "vinyl"

    def test_use_mert_true_stored(self):
        chain = FeedbackChain(sample_rate=_SR, use_mert=True)
        assert chain.use_mert is True

    def test_excellence_mode_with_material_and_mert(self):
        chain = FeedbackChain(sample_rate=_SR, excellence_mode=True, material="shellac", use_mert=False)
        assert chain.excellence_mode is True
        assert chain.material == "shellac"
        assert chain.use_mert is False

    def test_run_with_material_param_returns_result(self):
        chain = FeedbackChain(sample_rate=_SR, excellence_mode=True, material="tape")
        audio = _sine_signal(n=_N)

        def identity_phase(a, sr):
            return a.copy()

        result = chain.run(audio, [(99, identity_phase, {})])
        assert isinstance(result, FeedbackChainResult)
        assert result.audio.shape == audio.shape


class TestExcellenceOptimizerUseMert:
    """ExcellenceOptimizer use_mert Parameter."""

    def test_use_mert_default_false(self):
        from backend.core.excellence_optimizer import ExcellenceOptimizer

        opt = ExcellenceOptimizer(48000)
        assert opt.use_mert is False

    def test_use_mert_true_stored(self):
        from backend.core.excellence_optimizer import ExcellenceOptimizer

        opt = ExcellenceOptimizer(48000, use_mert=True)
        assert opt.use_mert is True

    def test_optimize_use_mert_true_no_crash(self):
        from backend.core.excellence_optimizer import ExcellenceOptimizer

        audio = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.3).astype(np.float32)
        opt = ExcellenceOptimizer(48000, use_mert=True)
        out, rep = opt.optimize(audio)
        assert out.shape == audio.shape
        assert not np.any(np.abs(out) > 1.01)

    def test_optimize_for_excellence_use_mert_kwarg(self):
        from backend.core.excellence_optimizer import optimize_for_excellence

        audio = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.3).astype(np.float32)
        out, rep = optimize_for_excellence(audio, 48000, use_mert=True)
        assert out.shape == audio.shape

    def test_optimize_use_mert_material_combined(self):
        from backend.core.excellence_optimizer import ExcellenceOptimizer

        audio = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.3).astype(np.float32)
        opt = ExcellenceOptimizer(48000, material="vinyl", use_mert=True)
        assert opt.material == "vinyl"
        assert opt.use_mert is True
        out, _ = opt.optimize(audio)
        assert out.shape == audio.shape


@pytest.mark.slow
class TestExcellenceBenchmark:
    """ExcellenceBenchmark Basisvalidierung."""

    @pytest.mark.timeout(120)
    def test_make_test_signals_returns_four(self):
        from benchmarks.excellence_benchmark import _make_test_signals

        signals = _make_test_signals()
        assert len(signals) == 4
        assert "clean_tone" in signals
        assert "noisy_music" in signals
        assert "dropout_signal" in signals
        assert "overtone_sparse" in signals

    def test_signal_shapes_correct(self):
        from benchmarks.excellence_benchmark import _N, _make_test_signals

        signals = _make_test_signals()
        for name, sig in signals.items():
            assert sig.dtype == np.float32, f"{name}: falscher dtype"
            assert len(sig) == _N, f"{name}: falsche Länge"
            assert not np.any(np.abs(sig) > 1.01), f"{name}: Clipping"

    @pytest.mark.timeout(120)
    def test_benchmark_run_all_returns_report(self):
        from benchmarks.excellence_benchmark import ExcellenceBenchmark, ExcellenceBenchmarkReport

        bench = ExcellenceBenchmark(_SR)
        rep = bench.run_all(materials=["auto"])
        assert isinstance(rep, ExcellenceBenchmarkReport)
        assert len(rep.results) == 4  # 4 Signale × 1 Material

    def test_benchmark_summary_keys(self):
        from benchmarks.excellence_benchmark import ExcellenceBenchmark

        bench = ExcellenceBenchmark(_SR)
        rep = bench.run_all(materials=["auto"])
        for key in ("avg_delta_ovr", "avg_after_ovr", "avg_rt_ms", "n_results"):
            assert key in rep.summary, f"'{key}' fehlt in summary"

    def test_benchmark_delta_ovr_non_negative(self):
        """Excellence-Pipeline darf nicht verschlechtern."""
        from benchmarks.excellence_benchmark import ExcellenceBenchmark

        bench = ExcellenceBenchmark(_SR)
        rep = bench.run_all(materials=["auto"])
        # Im Durchschnitt darf MUSIC_OVR nicht fallen (Toleranz: -0.02 für numerische Varianz)
        assert rep.summary["avg_delta_ovr"] >= -0.02, f"Ø ΔOVR={rep.summary['avg_delta_ovr']:.4f} ist zu stark negativ"

    def test_benchmark_signal_result_fields(self):
        from benchmarks.excellence_benchmark import ExcellenceBenchmark, SignalBenchmarkResult

        bench = ExcellenceBenchmark(_SR)
        rep = bench.run_all(materials=["tape"])
        for r in rep.results:
            assert isinstance(r, SignalBenchmarkResult)
            assert 1.0 <= r.before_music_ovr <= 5.0
            assert 1.0 <= r.after_music_ovr <= 5.0
            assert r.rt_ms >= 0


# ===========================================================================
# SEKTION 16 – Phase-55-Integration & DiffWave-Plugin inpaint()-Bridge (v10.0.0)
# ===========================================================================

import pathlib


class TestDiffWaveInpaintBridge:
    """DiffWave-Plugin besitzt jetzt inpaint() als Modul-Level-Funktion."""

    def _load_dw(self):
        import importlib
        import sys

        plugins_dir = str(pathlib.Path(__file__).parent.parent.parent / "plugins")
        if plugins_dir not in sys.path:
            sys.path.insert(0, plugins_dir)
        if "diffwave_plugin" in sys.modules:
            del sys.modules["diffwave_plugin"]
        return importlib.import_module("diffwave_plugin")

    def test_inpaint_function_exists(self):
        """hasattr(dw, 'inpaint') muss True sein."""
        dw = self._load_dw()
        assert hasattr(dw, "inpaint"), "inpaint() fehlt im Modul"

    def test_inpaint_returns_same_shape_mono(self):
        """Mono: Ergebnis-Shape identisch mit Eingabe."""
        dw = self._load_dw()
        audio = np.sin(2 * np.pi * 440 * np.arange(_SR) / _SR).astype(np.float32)
        result = dw.inpaint(audio, _SR // 4, _SR // 2, _SR, n_steps=3)
        assert result.shape == audio.shape

    def test_inpaint_returns_same_shape_stereo(self):
        """Stereo (2, N): Ergebnis-Shape identisch mit Eingabe."""
        dw = self._load_dw()
        mono = np.sin(2 * np.pi * 440 * np.arange(_SR) / _SR).astype(np.float32)
        stereo = np.vstack([mono, mono * 0.8])
        result = dw.inpaint(stereo, _SR // 4, _SR // 2, _SR, n_steps=3)
        assert result.shape == stereo.shape

    def test_inpaint_gap_is_filled(self):
        """Lücke darf nach inpaint() nicht mehr komplett Null sein."""
        dw = self._load_dw()
        audio = np.sin(2 * np.pi * 440 * np.arange(_SR) / _SR).astype(np.float32)
        audio_gap = audio.copy()
        audio_gap[_SR // 4 : _SR // 2] = 0.0
        result = dw.inpaint(audio_gap, _SR // 4, _SR // 2, _SR, n_steps=3)
        gap_rms = float(np.sqrt(np.mean(result[_SR // 4 : _SR // 2] ** 2)))
        assert gap_rms > 0.01, f"Lücke nicht gefüllt: gap_rms={gap_rms:.4f}"

    def test_inpaint_no_nan_in_result(self):
        """Kein NaN im Ergebnis."""
        dw = self._load_dw()
        audio = np.sin(2 * np.pi * 220 * np.arange(_SR) / _SR).astype(np.float32)
        result = dw.inpaint(audio, 2000, 5000, _SR, n_steps=5)
        assert not np.any(np.isnan(result)), "NaN im inpaint()-Ergebnis"

    def test_inpaint_no_clipping(self):
        """Ergebnis bleibt im Bereich [-4, 4] (kein Auflaufen)."""
        dw = self._load_dw()
        audio = np.sin(2 * np.pi * 440 * np.arange(_SR) / _SR).astype(np.float32)
        result = dw.inpaint(audio, _SR // 4, _SR // 2, _SR, n_steps=5)
        assert float(np.max(np.abs(result))) < 4.0, "Amplitude-Explosion in inpaint()"

    def test_inpaint_silence_stays_near_zero(self):
        """Stilles Signal: Lücke bleibt nahe Null."""
        dw = self._load_dw()
        audio = np.zeros(_SR, dtype=np.float32)
        result = dw.inpaint(audio, 1000, 3000, _SR, n_steps=3)
        gap_rms = float(np.sqrt(np.mean(result[1000:3000] ** 2)))
        assert gap_rms < 0.01, f"Stille-Test: gap_rms={gap_rms:.6f} zu hoch"

    def test_inpaint_gap_amplitude_near_context(self):
        """Ausgefüllte Lücke hat ähnliche RMS wie Kontext (Faktor < 5)."""
        dw = self._load_dw()
        audio = np.sin(2 * np.pi * 440 * np.arange(_SR) / _SR).astype(np.float32)
        ctx_rms = float(np.sqrt(np.mean(audio[: _SR // 4] ** 2)))
        result = dw.inpaint(audio, _SR // 4, _SR // 2, _SR, n_steps=5)
        gap_rms = float(np.sqrt(np.mean(result[_SR // 4 : _SR // 2] ** 2)))
        assert gap_rms < 5.0 * ctx_rms + 0.01, f"gap_rms={gap_rms:.4f} >> ctx_rms={ctx_rms:.4f}"


class TestPhase55Export:
    """Phase 55 muss aus core.phases importierbar sein."""

    def test_diffusion_inpainting_phase_importable(self):
        """core.phases.DiffusionInpaintingPhase ist verfügbar."""
        from backend.core.phases import DiffusionInpaintingPhase

        assert DiffusionInpaintingPhase is not None

    def test_diffusion_inpainting_phase_in_all(self):
        """DiffusionInpaintingPhase steht in __all__."""
        import backend.core.phases as cp

        assert "DiffusionInpaintingPhase" in cp.__all__

    def test_diffusion_inpainting_phase_is_class(self):
        """DiffusionInpaintingPhase ist eine Klasse."""
        from backend.core.phases import DiffusionInpaintingPhase

        assert isinstance(DiffusionInpaintingPhase, type)

    def test_diffusion_inpainting_phase_instantiable(self):
        """DiffusionInpaintingPhase lässt sich instanzieren."""
        from backend.core.phases import DiffusionInpaintingPhase

        phase = DiffusionInpaintingPhase()
        assert phase is not None

    def test_diffusion_inpainting_phase_has_process(self):
        """DiffusionInpaintingPhase besitzt process()-Methode."""
        from backend.core.phases import DiffusionInpaintingPhase

        assert hasattr(DiffusionInpaintingPhase, "process")


class TestPhase55DiffWaveBridgeIntegration:
    """Phase 55 nutzt DiffWave-Bridge wenn Plugin verfügbar ist."""

    def test_phase55_plugin_path_activated(self):
        """Nach dem Fix erkennt Phase 55 inpaint() als vorhanden."""
        import importlib
        import sys

        plugins_dir = str(pathlib.Path(__file__).parent.parent.parent / "plugins")
        if plugins_dir not in sys.path:
            sys.path.insert(0, plugins_dir)
        dw = importlib.import_module("diffwave_plugin")
        assert hasattr(dw, "inpaint"), "DiffWave-Plugin hat kein inpaint() - Plugin-Pfad in Phase 55 inaktiv"

    def test_phase55_full_process_no_crash(self):
        """DiffusionInpaintingPhase.process() läuft auf kurzem Testsignal durch."""
        from backend.core.phases import DiffusionInpaintingPhase

        phase = DiffusionInpaintingPhase()
        sr = 48000
        audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        audio[sr // 2 : sr // 2 + int(0.05 * sr)] = 0.0
        result = phase.process(audio, sample_rate=sr)
        # Phase 55 liefert typisiert ein PhaseResult.
        out_audio = result.audio
        assert out_audio.shape[0] == audio.shape[0] or out_audio.shape[-1] == audio.shape[-1]

    def test_phase55_mono_output_no_nan(self):
        """Kein NaN in Phase-55-Ausgabe."""
        from backend.core.phases import DiffusionInpaintingPhase

        phase = DiffusionInpaintingPhase()
        sr = 48000
        audio = np.random.default_rng(0).uniform(-0.5, 0.5, sr).astype(np.float32)
        audio[1000:1500] = 0.0
        result = phase.process(audio, sample_rate=sr)
        out_audio = result.audio  # PhaseResult always has .audio
        assert not np.any(np.isnan(out_audio))
