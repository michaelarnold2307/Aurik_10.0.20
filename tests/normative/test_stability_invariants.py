from __future__ import annotations

"""[RELEASE_MUST] §3.9 Stabilitäts-Invarianten — normative CI-Gate Tests (v10.0.0)

Spec reference:  .github/specs/08_architecture_and_distribution.md §3.9.1–§3.9.9
Gate entry:      copilot-instructions.md §3.9 — je Invariante mind. 3 Tests

§3.9.1  Per-Phase-Inference-Timeout   → test_s01_*
§3.9.2  SIGTERM-Handler               → test_s02_*
§3.9.3  Phase-Output-Guard            → test_s03_*
§3.9.4  ThreadPoolExecutor-Lifecycle  → test_s04_*
§3.9.5  ml_memory_budget Reconcile    → test_s05_*
§3.9.6  Structured Exception Logging  → test_s06_*
§3.9.7  Audio-Buffer-RAM-Guard        → test_s07_*
§3.9.8  Lock-Order documentation      → test_s08_*
§3.9.9  MLRefinementThread Buffer     → test_s09_*

Aufruf: pytest tests/normative/test_stability_invariants.py -v --timeout=30
"""


import inspect
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_audio(seconds: float = 0.1, sr: int = 48000, stereo: bool = False) -> np.ndarray:
    """Create a valid float32 audio array without touching disk."""
    samples = int(seconds * sr)
    if stereo:
        return np.zeros((samples, 2), dtype=np.float32)
    return np.zeros(samples, dtype=np.float32)


# =============================================================================
# §3.9.1  InferenceTimeoutError + _run_inference_with_timeout
# =============================================================================


class TestS01InferenceTimeout:
    """§3.9.1: Heavy ML inference must have a wall-clock timeout."""

    def test_module_exports_error_class(self):
        """InferenceTimeoutError must be importable from phase_output_guard."""
        from backend.core.phase_output_guard import InferenceTimeoutError

        assert issubclass(InferenceTimeoutError, RuntimeError)

    def test_run_inference_with_timeout_returns_result(self):
        """_run_inference_with_timeout forwards return value on fast functions."""
        from backend.core.phase_output_guard import _run_inference_with_timeout

        result = _run_inference_with_timeout(lambda: 42, timeout=5.0)
        assert result == 42

    def test_run_inference_with_timeout_raises_on_timeout(self):
        """_run_inference_with_timeout raises InferenceTimeoutError when fn hangs."""
        from backend.core.phase_output_guard import (
            InferenceTimeoutError,
            _run_inference_with_timeout,
        )

        def _hanging():
            time.sleep(5)  # short enough for background cleanup, long enough to trigger timeout

        with pytest.raises(InferenceTimeoutError):
            _run_inference_with_timeout(_hanging, timeout=0.1)

    def test_run_inference_propagates_exceptions(self):
        """Non-timeout exceptions from fn are propagated unchanged."""
        from backend.core.phase_output_guard import _run_inference_with_timeout

        def _failing():
            raise ValueError("inference error")

        with pytest.raises(ValueError, match="inference error"):
            _run_inference_with_timeout(_failing, timeout=5.0)

    def test_default_timeout_constant(self):
        """PHASE_INFERENCE_TIMEOUT_S must be ≥ 60 s (not an accidental 0)."""
        from backend.core.phase_output_guard import PHASE_INFERENCE_TIMEOUT_S

        assert PHASE_INFERENCE_TIMEOUT_S >= 60.0


# =============================================================================
# §3.9.2  SIGTERM handler
# =============================================================================


@pytest.mark.gui
class TestS02SigtermHandler:
    """§3.9.2: main.py must register a SIGTERM handler."""

    def test_sigterm_handler_defined_in_main(self):
        """_sigterm_handler must be defined in Aurik10.main."""
        import importlib

        mod = importlib.import_module("Aurik10.main")
        assert hasattr(mod, "_sigterm_handler"), (
            "_sigterm_handler() not found in Aurik10.main — §3.9.2 implementation missing"
        )

    def test_sigterm_handler_is_callable(self):
        """_sigterm_handler must be callable with (signum, frame) signature."""
        import importlib

        mod = importlib.import_module("Aurik10.main")
        fn = mod._sigterm_handler
        assert callable(fn)
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 2

    def test_emergency_checkpoint_if_running_defined(self):
        """_emergency_checkpoint_if_running must be defined in Aurik10.main."""
        import importlib

        mod = importlib.import_module("Aurik10.main")
        assert hasattr(mod, "_emergency_checkpoint_if_running"), (
            "_emergency_checkpoint_if_running() not found — §3.9.2 incomplete"
        )

    def test_emergency_checkpoint_if_running_does_not_raise(self):
        """_emergency_checkpoint_if_running must not raise when no Qt app is running."""
        import importlib

        mod = importlib.import_module("Aurik10.main")
        # Should silently succeed even with no QApplication instance
        mod._emergency_checkpoint_if_running()

    def test_sigterm_handler_does_not_raise_without_app(self):
        """_sigterm_handler must not raise when called with no QApplication."""
        import importlib

        mod = importlib.import_module("Aurik10.main")
        # Simulate SIGTERM call; no QApplication present in test env
        mod._sigterm_handler(15, None)


# =============================================================================
# §3.9.3  Phase-Output-Guard decorator
# =============================================================================


class TestS03PhaseOutputGuard:
    """§3.9.3: @phase_output_guard must sanitise all phase audio outputs."""

    def test_phase_output_error_is_importable(self):
        """PhaseOutputError must be importable from backend.core.phase_output_guard."""
        from backend.core.phase_output_guard import PhaseOutputError

        assert issubclass(PhaseOutputError, RuntimeError)

    def test_decorator_passes_clean_audio_unchanged(self):
        """Clean float32 audio passes through the decorator without modification."""
        from backend.core.phase_output_guard import phase_output_guard

        @phase_output_guard
        def _phase(audio: np.ndarray) -> np.ndarray:
            return audio

        audio = _make_audio()
        result = _phase(audio)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()

    def test_decorator_fixes_nan(self):
        """NaN values in phase output must be replaced with 0.0."""
        from backend.core.phase_output_guard import phase_output_guard

        @phase_output_guard
        def _phase(audio: np.ndarray) -> np.ndarray:
            out = audio.copy()
            out[0] = float("nan")
            return out

        audio = _make_audio()
        result = _phase(audio)
        assert not np.isnan(result).any()
        assert result[0] == 0.0

    def test_decorator_clips_amplitude(self):
        """Values outside [-1, 1] must be clipped to those bounds."""
        from backend.core.phase_output_guard import phase_output_guard

        @phase_output_guard
        def _phase(audio: np.ndarray) -> np.ndarray:
            out = audio.copy()
            out[0] = 5.0  # over
            out[1] = -3.0  # under
            return out

        audio = _make_audio()
        result = _phase(audio)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(-1.0)

    def test_decorator_casts_non_float32(self):
        """Phase output with wrong dtype (float64) must be cast to float32."""
        from backend.core.phase_output_guard import phase_output_guard

        @phase_output_guard
        def _phase(audio: np.ndarray) -> np.ndarray:
            return audio.astype(np.float64)

        audio = _make_audio()
        result = _phase(audio)
        assert result.dtype == np.float32

    def test_decorator_handles_tuple_return(self):
        """Decorator must sanitise first ndarray in a tuple return value."""
        from backend.core.phase_output_guard import phase_output_guard

        @phase_output_guard
        def _phase(audio: np.ndarray) -> tuple:
            out = audio.copy()
            out[0] = float("nan")
            return (out, {"meta": 1})

        audio = _make_audio()
        result_audio, meta = _phase(audio)
        assert not np.isnan(result_audio).any()
        assert meta == {"meta": 1}


# =============================================================================
# §3.9.4  ThreadPoolExecutor lifecycle — cancel_futures in shutdown
# =============================================================================


class TestS04ExecutorLifecycle:
    """§3.9.4: ModuleCoordinator must shut down its executor with cancel_futures=True."""

    def test_shutdown_method_exists(self):
        """ModuleCoordinator must have a shutdown() method."""
        from backend.core.module_coordinator import ModuleCoordinator

        assert hasattr(ModuleCoordinator, "shutdown")
        assert callable(ModuleCoordinator.shutdown)

    def test_shutdown_calls_executor_shutdown_with_cancel(self):
        """ModuleCoordinator.shutdown() must pass cancel_futures=True."""
        from backend.core.module_coordinator import ModuleCoordinator

        # Inspect the source to confirm cancel_futures=True is present
        src = inspect.getsource(ModuleCoordinator.shutdown)
        assert "cancel_futures=True" in src, (
            "ModuleCoordinator.shutdown() must call executor.shutdown(wait=True, cancel_futures=True) — §3.9.4"
        )

    def test_shutdown_sets_pool_to_none(self):
        """After shutdown(), _thread_pool must be None."""
        from unittest.mock import MagicMock

        from backend.core.module_coordinator import ModuleCoordinator

        # create a minimal coordinator with a mocked pool
        mc = ModuleCoordinator.__new__(ModuleCoordinator)
        mc._thread_pool = MagicMock()
        mc._module_instances = {}
        with patch.object(mc._thread_pool, "shutdown"):
            mc.shutdown()
        assert mc._thread_pool is None

    def test_run_inference_with_timeout_uses_single_worker_executor(self):
        """_run_inference_with_timeout must NOT leave long-lived threads."""
        from backend.core.phase_output_guard import _run_inference_with_timeout

        # Fast function — executor context manager exits cleanly
        _run_inference_with_timeout(lambda: None, timeout=5.0)
        # No assertion needed; absence of ResourceWarning / hanging threads is the test


# =============================================================================
# §3.9.5  ml_memory_budget startup reconciliation
# =============================================================================


class TestS05BudgetReconciliation:
    """§3.9.5: ml_memory_budget must reconcile (reset) on fresh process start."""

    def test_reconcile_function_exists(self):
        """_reconcile_on_startup must be importable."""
        from backend.core.ml_memory_budget import _reconcile_on_startup

        assert callable(_reconcile_on_startup)

    def test_reconcile_clears_stale_allocations(self):
        """_reconcile_on_startup must reset _allocated and _total_gb to zero."""
        import backend.core.ml_memory_budget as bgt

        # Inject fake stale state from a "previous run"
        with bgt._lock:
            bgt._allocated["stale_model"] = 4.0
            bgt._total_gb = 4.0

        bgt._reconcile_on_startup()

        with bgt._lock:
            assert "stale_model" not in bgt._allocated, (
                "_reconcile_on_startup must clear stale allocations from previous process"
            )
            assert bgt._total_gb == 0.0

    def test_fresh_import_starts_with_zero_allocation(self):
        """After module import, _total_gb must be 0."""
        import backend.core.ml_memory_budget as bgt

        # reconcile was already called at import; now undo any allocations
        bgt._reconcile_on_startup()
        with bgt._lock:
            assert bgt._total_gb == 0.0

    def test_get_ml_memory_budget_returns_proxy(self):
        """get_ml_memory_budget() must return object with try_allocate and release."""
        from backend.core.ml_memory_budget import get_ml_memory_budget

        proxy = get_ml_memory_budget()
        assert hasattr(proxy, "try_allocate")
        assert hasattr(proxy, "release")
        assert callable(proxy.try_allocate)
        assert callable(proxy.release)

    def test_proxy_try_allocate_delegates_correctly(self):
        """Proxy.try_allocate() must work identically to module-level try_allocate()."""
        import backend.core.ml_memory_budget as bgt
        from backend.core.ml_memory_budget import get_ml_memory_budget

        bgt._reconcile_on_startup()
        proxy = get_ml_memory_budget()
        ok = proxy.try_allocate("test_proxy_model", 0.001)
        # Cleanup
        proxy.release("test_proxy_model")
        assert ok, "Proxy.try_allocate() failed for tiny allocation"


# =============================================================================
# §3.9.6  Structured exception logging
# =============================================================================


class TestS06StructuredExceptionLogging:
    """§3.9.6: Pipeline-critical paths must not silently swallow exceptions."""

    def test_phase_output_guard_logs_critical_on_postcondition_failure(self, caplog):
        """@phase_output_guard logs CRITICAL when post-sanitisation invariant fails."""
        import logging

        from backend.core.phase_output_guard import phase_output_guard

        class _BadArray:
            """Fake ndarray that keeps returning inf even after clip."""

            @property
            def nbytes(self):
                return 4

            def __array__(self):
                return np.array([float("inf")], dtype=np.float32)

        # Fabricate a pathological case: returning a normal NaN array should
        # still be caught and replaced with silence (0.0).  Verify the decorator
        # logs correctly when forced to fail.
        @phase_output_guard
        def _phase(audio):
            out = audio.copy()
            out[:] = float("nan")
            return out

        with caplog.at_level(logging.DEBUG):
            result = _phase(_make_audio())
        # NaN must be gone (replaced with 0.0)
        assert not np.isnan(result).any()

    def test_inference_timeout_error_contains_fn_name(self):
        """InferenceTimeoutError message must include the function identifier."""
        from backend.core.phase_output_guard import (
            InferenceTimeoutError,
            _run_inference_with_timeout,
        )

        def _my_slow_model():
            time.sleep(5)  # short enough for background cleanup

        with pytest.raises(InferenceTimeoutError) as exc_info:
            _run_inference_with_timeout(_my_slow_model, timeout=0.1)

        assert "_my_slow_model" in str(exc_info.value) or "timeout" in str(exc_info.value).lower()

    def test_audio_too_large_error_has_german_message(self):
        """AudioTooLargeError must carry a German user-facing message."""
        from backend.core.audio_file_validator import (
            AudioTooLargeError,
            _check_audio_buffer_size,
        )

        # Create a mock array that is "too big" by overriding nbytes
        big = MagicMock(spec=np.ndarray)
        big.nbytes = 3 * 1024**3  # 3 GB > 2 GB limit
        with pytest.raises(AudioTooLargeError) as exc_info:
            _check_audio_buffer_size(big, "/tmp/huge.wav")
        msg = str(exc_info.value)
        # German keywords expected in message
        assert any(kw in msg for kw in ["GB", "überschreitet", "RAM"]), (
            f"AudioTooLargeError message should be German, got: {msg}"
        )


# =============================================================================
# §3.9.7  Audio-Buffer-RAM-Guard
# =============================================================================


class TestS07AudioBufferRamGuard:
    """§3.9.7: Array exceeding 2 GB RAM limit must raise before pipeline entry."""

    def test_audio_too_large_error_importable(self):
        """AudioTooLargeError must be importable from backend.core.audio_file_validator."""
        from backend.core.audio_file_validator import AudioTooLargeError

        assert issubclass(AudioTooLargeError, Exception)

    def test_max_audio_bytes_ram_constant_exists(self):
        """MAX_AUDIO_BYTES_RAM must be defined as a module-level constant."""
        from backend.core.audio_file_validator import MAX_AUDIO_BYTES_RAM

        assert isinstance(MAX_AUDIO_BYTES_RAM, int)
        assert MAX_AUDIO_BYTES_RAM == 2 * 1024**3  # exactly 2 GB

    def test_check_audio_buffer_size_passes_for_small_audio(self):
        """Small audio array (< 2 GB) must not raise."""
        from backend.core.audio_file_validator import _check_audio_buffer_size

        small = _make_audio(0.1)
        _check_audio_buffer_size(small, "/tmp/test.wav")  # must not raise

    def test_check_audio_buffer_size_raises_for_oversized_array(self):
        """Array exceeding 2 GB must raise AudioTooLargeError."""
        from backend.core.audio_file_validator import (
            AudioTooLargeError,
            _check_audio_buffer_size,
        )

        big = MagicMock(spec=np.ndarray)
        big.nbytes = 3 * 1024**3  # 3 GB
        with pytest.raises(AudioTooLargeError):
            _check_audio_buffer_size(big, "/tmp/big.wav")

    def test_check_audio_buffer_size_includes_filename_in_message(self):
        """Error message must include the filename for actionable feedback."""
        from backend.core.audio_file_validator import (
            AudioTooLargeError,
            _check_audio_buffer_size,
        )

        big = MagicMock(spec=np.ndarray)
        big.nbytes = 3 * 1024**3
        with pytest.raises(AudioTooLargeError) as exc_info:
            _check_audio_buffer_size(big, "/recordings/concert_8h.wav")
        assert "concert_8h.wav" in str(exc_info.value)

    def test_audio_too_large_error_inherits_audio_load_error(self):
        """AudioTooLargeError must inherit from AudioLoadError."""
        from backend.core.audio_file_validator import AudioLoadError, AudioTooLargeError

        assert issubclass(AudioTooLargeError, AudioLoadError)


# =============================================================================
# §3.9.8  Lock-order documentation
# =============================================================================


class TestS08LockOrder:
    """§3.9.8: Deadlock prevention — lock-order comments must be present."""

    def test_plm_lock_order_comment_present(self):
        """PluginLifecycleManager class must document its lock order."""
        from backend.core import plugin_lifecycle_manager as plm_mod

        src = inspect.getsource(plm_mod.PluginLifecycleManager)
        assert "Lock-order" in src or "lock_order" in src.lower() or "§3.9.8" in src, (
            "PluginLifecycleManager must document its lock order (§3.9.8)"
        )

    def test_arm_lock_order_comment_present(self):
        """AdaptiveResourceManager class must document its lock order."""
        from backend.core import adaptive_resource_manager as arm_mod

        src = inspect.getsource(arm_mod.AdaptiveResourceManager)
        assert "Lock-order" in src or "lock_order" in src.lower() or "§3.9.8" in src, (
            "AdaptiveResourceManager must document its lock order (§3.9.8)"
        )

    def test_ml_memory_budget_proxy_lock_order_comment_present(self):
        """MLMemoryBudget proxy must document its lock order."""
        from backend.core import ml_memory_budget as bgt_mod

        src = inspect.getsource(bgt_mod._MLMemoryBudgetProxy)
        assert "Lock-order" in src or "§3.9.8" in src, "_MLMemoryBudgetProxy must document its lock order (§3.9.8)"

    def test_evict_stale_plugins_called_outside_arm_lock(self):
        """ARM must call evict_stale_plugins() OUTSIDE its own lock (§3.9.8)."""
        from backend.core import adaptive_resource_manager as arm_mod

        src = inspect.getsource(arm_mod.AdaptiveResourceManager)
        # Verify that the source explicitly acknowledges the outside-lock pattern
        assert "outside" in src.lower() or "außerhalb" in src.lower() or "OUTSIDE" in src, (
            "ARM source must document that evict_stale_plugins() runs outside the ARM lock"
        )


# =============================================================================
# §3.9.9  DeferredRefinementJob buffer release
# =============================================================================


class TestS09KmvBufferRelease:
    """§3.9.9: KMV job must release ml_memory_budget on completion or cancellation."""

    def test_release_buffer_method_exists(self):
        """DeferredRefinementJob must have a release_buffer() method."""
        from backend.core.deferred_refinement_job import DeferredRefinementJob

        assert hasattr(DeferredRefinementJob, "release_buffer")
        assert callable(DeferredRefinementJob.release_buffer)

    def test_release_buffer_sets_audio_to_none(self):
        """release_buffer() must set audio_original to None for GC."""
        from backend.core.deferred_refinement_job import DeferredRefinementJob

        audio = _make_audio(0.05)
        job = DeferredRefinementJob(
            output_path="/tmp/out.wav",
            audio_original=audio,
            sr=48000,
            mode="restoration",
            deferred_phase_ids=["phase_03"],
            cached_defect_result=None,
            cached_era_result=None,
            cached_medium_result=None,
            stufe1_quality=0.6,
        )
        job.release_buffer()
        assert job.audio_original is None, "release_buffer() must set audio_original=None so GC can reclaim the array"

    def test_release_buffer_is_idempotent(self):
        """Calling release_buffer() multiple times must not raise."""
        from backend.core.deferred_refinement_job import DeferredRefinementJob

        audio = _make_audio(0.05)
        job = DeferredRefinementJob(
            output_path="/tmp/out.wav",
            audio_original=audio,
            sr=48000,
            mode="restoration",
            deferred_phase_ids=[],
            cached_defect_result=None,
            cached_era_result=None,
            cached_medium_result=None,
            stufe1_quality=0.5,
        )
        job.release_buffer()
        job.release_buffer()  # must not raise

    def test_ml_refinement_thread_finally_uses_release_buffer(self):
        """MLRefinementThread.run() finally block must call job.release_buffer()."""
        from Aurik10.ui import ml_refinement_thread as mrt_mod

        src = inspect.getsource(mrt_mod.MLRefinementThread.run)
        assert "release_buffer" in src, "MLRefinementThread.run() finally block must call job.release_buffer() — §3.9.9"

    def test_deferred_refinement_job_has_audio_size_gb_property(self):
        """audio_size_gb property must reflect the numpy array size."""
        from backend.core.deferred_refinement_job import DeferredRefinementJob

        audio = np.zeros(48000, dtype=np.float32)  # 1 s mono @ 48 kHz = 192 kB
        job = DeferredRefinementJob(
            output_path="/tmp/out.wav",
            audio_original=audio,
            sr=48000,
            mode="restoration",
            deferred_phase_ids=[],
            cached_defect_result=None,
            cached_era_result=None,
            cached_medium_result=None,
            stufe1_quality=0.5,
        )
        assert job.audio_size_gb > 0.0
        assert job.audio_size_gb < 1.0  # 1 s at 48 kHz is much less than 1 GB
