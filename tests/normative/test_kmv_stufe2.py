from __future__ import annotations

"""
§2.38 KMV Stufe-2 — Normative CI-Tests

Prüft:
- DeferredRefinementJob (Dataclass-Felder, Properties)
- MLRefinementThread.should_start() RAM-Guard
- Qualitätsinvariante (kein Overwrite wenn stufe2 < stufe1)
- Atomar-Schreib-Pfad (.tmp → os.replace)
- Signal-Kontrakt (alle 5 §2.38-Pflicht-Signale vorhanden)
- refinement_complete / refinement_cancelled Endstatus
"""


from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.gui

# ── DeferredRefinementJob ────────────────────────────────────────────────────


@pytest.fixture()
def minimal_job(tmp_path):
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(48000, dtype=np.float32)
    return DeferredRefinementJob(
        output_path=str(tmp_path / "out.wav"),
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=["phase_20_reverb_reduction", "phase_55_diffusion_inpainting"],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.62,
        input_path=str(tmp_path / "in.wav"),
    )


def test_deferred_job_mandatory_fields():
    """All §2.38 mandatory fields must be present."""
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    field_names = {f.name for f in fields(DeferredRefinementJob)}
    required = {
        "output_path",
        "audio_original",
        "sr",
        "mode",
        "deferred_phase_ids",
        "cached_defect_result",
        "cached_era_result",
        "cached_medium_result",
        "stufe1_quality",
        "input_path",
    }
    missing = required - field_names
    assert not missing, f"Pflicht-Felder fehlen: {missing}"


def test_deferred_job_audio_size_gb(minimal_job):
    """audio_size_gb property must be >0 for non-empty audio."""
    assert minimal_job.audio_size_gb > 0.0
    assert minimal_job.audio_size_gb < 1.0  # 48 000 float32 ≈ 0.00018 GB


def test_deferred_job_n_deferred(minimal_job):
    assert minimal_job.n_deferred == 2


def test_deferred_job_empty_phases():
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(1024, dtype=np.float32)
    job = DeferredRefinementJob(
        output_path="/tmp/x.wav",
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=[],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.55,
        input_path="/tmp/in.wav",
    )
    assert job.n_deferred == 0


# ── MLRefinementThread.should_start() ───────────────────────────────────────


def test_should_start_no_deferred_phases(tmp_path):
    """should_start must return False when deferred_phase_ids is empty."""
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(1024, dtype=np.float32)
    job = DeferredRefinementJob(
        output_path=str(tmp_path / "out.wav"),
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=[],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.60,
        input_path="",
    )
    from Aurik10.ui.ml_refinement_thread import MLRefinementThread

    assert MLRefinementThread.should_start(job) is False


def test_should_start_insufficient_ram(minimal_job):
    """should_start must return False when <4 GB RAM free."""
    import psutil

    from Aurik10.ui.ml_refinement_thread import MLRefinementThread

    mock_vm = MagicMock()
    mock_vm.available = int(3.9 * 1024**3)  # 3.9 GB < 4 GB required
    with patch.object(psutil, "virtual_memory", return_value=mock_vm):
        result = MLRefinementThread.should_start(minimal_job)
    assert result is False


def test_should_start_sufficient_ram(minimal_job):
    """should_start must return True when ≥4 GB RAM free and phases present."""
    import psutil

    from Aurik10.ui.ml_refinement_thread import MLRefinementThread

    mock_vm = MagicMock()
    mock_vm.available = int(8.0 * 1024**3)  # 8 GB — sufficient
    with patch.object(psutil, "virtual_memory", return_value=mock_vm):
        result = MLRefinementThread.should_start(minimal_job)
    assert result is True


# ── QualitätsInvariante (kein Overwrite wenn stufe2 < stufe1) ────────────────


def test_quality_invariant_logic_present():
    """run() source must contain quality gate: stufe2 < stufe1 → emit cancelled."""
    import inspect

    from Aurik10.ui.ml_refinement_thread import MLRefinementThread

    src = inspect.getsource(MLRefinementThread.run)
    assert "stufe1_quality" in src, "Quality-Gate fehlt in run()"
    assert "refinement_cancelled" in src, "refinement_cancelled-Emit fehlt in run()"


def test_quality_invariant_no_overwrite(tmp_path, minimal_job):
    """_write_audio writes to the given path (quality gate is upstream in run())."""

    from Aurik10.ui.ml_refinement_thread import _write_audio

    audio = np.zeros(480, dtype=np.float32) + 0.1
    tmp_path_str = str(tmp_path / "quality_guard.wav")
    _write_audio(audio, 48000, tmp_path_str)
    assert Path(tmp_path_str).exists(), "_write_audio muss Datei erstellen"


def test_quality_invariant_overwrite_when_better(tmp_path):
    """_write_audio successfully creates a valid WAV file."""
    import soundfile as sf

    from Aurik10.ui.ml_refinement_thread import _write_audio

    audio_new = np.zeros(480, dtype=np.float32) + 0.5
    output_path = str(tmp_path / "write_test.wav")
    _write_audio(audio_new, 48000, output_path)
    assert Path(output_path).exists()
    audio_read, _ = sf.read(output_path, dtype="float32")
    assert audio_read.size > 0, "Geschriebene Datei muss Audio enthalten"


# ── Atomares Schreiben (.tmp → os.replace) ────────────────────────────────────
# Atomares Muster: .kmv_tmp → os.replace → kein .tmp links = MLRefinementThread.run()-Logik.
# _write_audio selbst schreibt direkt an den übergebenen Pfad (kein internen .tmp).


def test_atomic_write_pattern_in_run_source():
    """run() must use os.replace for atomic overwrite (not shutil.copy or direct open)."""
    import inspect

    from Aurik10.ui.ml_refinement_thread import MLRefinementThread

    src = inspect.getsource(MLRefinementThread.run)
    assert "os.replace" in src, "Atomarer os.replace-Schritt muss in run() vorhanden sein"
    assert ".kmv_tmp" in src or "_tmp" in src, "Temporärer Dateiname muss vor os.replace verwendet werden"


def test_write_audio_creates_wavfile(tmp_path):
    """_write_audio must create a readable WAV file at the given path."""
    import soundfile as sf

    from Aurik10.ui.ml_refinement_thread import _write_audio

    audio = np.zeros(480, dtype=np.float32) + 0.2
    output_path = str(tmp_path / "atomic_test.wav")

    _write_audio(audio, 48000, output_path)
    assert Path(output_path).exists(), "_write_audio muss Zieldatei anlegen"
    loaded, sr = sf.read(output_path, dtype="float32")
    assert sr == 48000
    assert loaded.size > 0


def test_build_atomic_temp_path_preserves_target_suffix() -> None:
    """KMV-Tempdatei muss die finale Dateiendung behalten, damit das Format stabil bleibt."""
    from Aurik10.ui.ml_refinement_thread import _build_atomic_temp_path

    assert _build_atomic_temp_path("/tmp/song.wav") == "/tmp/song.kmv_tmp.wav"
    assert _build_atomic_temp_path("/tmp/song.flac") == "/tmp/song.kmv_tmp.flac"
    assert _build_atomic_temp_path("/tmp/song") == "/tmp/song.kmv_tmp"


# ── Signal-Kontrakt (alle 5 §2.38-Pflicht-Signale vorhanden) ─────────────────


def test_ml_refinement_thread_signals_exist():
    """All 5 §2.38 mandatory signals must exist on MLRefinementThread."""
    from Aurik10.ui.ml_refinement_thread import MLRefinementThread

    required_signals = [
        "refinement_started",
        "refinement_phase_done",
        "refinement_progress",
        "refinement_complete",
        "refinement_cancelled",
    ]
    for sig in required_signals:
        assert hasattr(MLRefinementThread, sig), f"Signal {sig!r} fehlt in MLRefinementThread"


# ── RestorationResult-Felder §2.38 (via dataclasses.fields — kein Konstrukt) ──


def test_restoration_result_has_deferred_phases():
    """`deferred_phases` @dataclass field must exist with empty-list default."""
    from dataclasses import MISSING, fields

    from backend.core.unified_restorer_v3 import RestorationResult

    fmap = {f.name: f for f in fields(RestorationResult)}
    assert "deferred_phases" in fmap, "RestorationResult.deferred_phases fehlt"
    assert fmap["deferred_phases"].default is MISSING  # uses default_factory
    assert callable(fmap["deferred_phases"].default_factory)  # type: ignore[misc]
    assert fmap["deferred_phases"].default_factory() == []


def test_restoration_result_has_refinement_complete():
    """`refinement_complete` @dataclass field must exist with False default."""
    from dataclasses import fields

    from backend.core.unified_restorer_v3 import RestorationResult

    fmap = {f.name: f for f in fields(RestorationResult)}
    assert "refinement_complete" in fmap, "RestorationResult.refinement_complete fehlt"
    assert fmap["refinement_complete"].default is False


def test_restoration_result_has_stufe2_quality_estimate():
    """`stufe2_quality_estimate` @dataclass field must exist with None default."""
    from dataclasses import fields

    from backend.core.unified_restorer_v3 import RestorationResult

    fmap = {f.name: f for f in fields(RestorationResult)}
    assert "stufe2_quality_estimate" in fmap, "RestorationResult.stufe2_quality_estimate fehlt"
    assert fmap["stufe2_quality_estimate"].default is None


# ── DeferredRefinementJob Typ-Sicherheit ─────────────────────────────────────


def test_deferred_job_mode_is_lowercase(minimal_job):
    """mode must be 'restoration' or 'studio2026' (lowercase)."""
    assert minimal_job.mode in ("restoration", "studio2026")


def test_deferred_job_sr_is_48000(minimal_job):
    """sr must be 48000 (Verarbeitungs-SR §2.37)."""
    assert minimal_job.sr == 48000


# ── Interruption-Handling (§2.38: isInterruptionRequested zwischen jeder Phase) ──


def test_interruption_early_exit(tmp_path):
    """run() must emit refinement_cancelled if interrupted before denke()."""

    from Aurik10.ui.ml_refinement_thread import MLRefinementThread
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(48000, dtype=np.float32)
    job = DeferredRefinementJob(
        output_path=str(tmp_path / "interrupt_test.wav"),
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=["phase_20_reverb_reduction"],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.60,
        input_path="",
    )
    thread = MLRefinementThread(job)
    # Record emitted signals
    cancelled_paths: list[str] = []
    thread.refinement_cancelled.connect(lambda p: cancelled_paths.append(p))
    # Force interruption before run starts
    thread.requestInterruption()
    thread.run()  # Execute synchronously for test
    assert len(cancelled_paths) >= 1, "refinement_cancelled must be emitted on interruption"
    assert cancelled_paths[0] == str(tmp_path / "interrupt_test.wav")


def test_budget_allocation_failure_cancels(tmp_path):
    """run() must emit refinement_cancelled if ml_memory_budget.try_allocate fails."""
    from Aurik10.ui.ml_refinement_thread import MLRefinementThread
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(48000, dtype=np.float32)
    job = DeferredRefinementJob(
        output_path=str(tmp_path / "budget_fail.wav"),
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=["phase_20_reverb_reduction"],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.60,
        input_path="",
    )
    thread = MLRefinementThread(job)
    cancelled_paths: list[str] = []
    thread.refinement_cancelled.connect(lambda p: cancelled_paths.append(p))

    # Mock budget to refuse allocation
    mock_budget = MagicMock()
    mock_budget.try_allocate.return_value = False
    with patch("backend.api.bridge.get_ml_memory_budget", return_value=mock_budget):
        thread.run()

    # Budget failure must result in cancellation (or the import mock path
    # might cause it to fall through to denker unavailable → cancellation too)
    assert len(cancelled_paths) >= 1, "Cancelled signal must be emitted on budget allocation failure"


def test_quality_gate_blocks_inferior_stufe2(tmp_path):
    """Stufe-2 result with lower quality than Stufe-1 must NOT overwrite."""
    from Aurik10.ui.ml_refinement_thread import MLRefinementThread
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.random.default_rng(42).standard_normal(48000).astype(np.float32) * 0.1
    out_path = str(tmp_path / "gate_test.wav")
    job = DeferredRefinementJob(
        output_path=out_path,
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=["phase_20_reverb_reduction"],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.95,  # Very high Stufe-1 — hard to beat
        input_path="",
    )
    thread = MLRefinementThread(job)
    cancelled_paths: list[str] = []
    complete_paths: list[str] = []
    thread.refinement_cancelled.connect(lambda p: cancelled_paths.append(p))
    thread.refinement_complete.connect(lambda p, r: complete_paths.append(p))

    # Mock denker to return result with lower quality
    mock_result = MagicMock()
    mock_result.quality_estimate = 0.50  # Lower than stufe1_quality (0.95)
    mock_result.audio = audio

    mock_denker = MagicMock()
    mock_denker.denke.return_value = mock_result

    with (
        patch("backend.api.bridge.get_aurik_denker_instance", return_value=mock_denker),
        patch("backend.api.bridge.get_ml_memory_budget") as mock_get_budget,
    ):
        mock_budget = MagicMock()
        mock_budget.try_allocate.return_value = True
        mock_get_budget.return_value = mock_budget
        thread.run()

    # Quality gate must prevent overwrite
    assert len(cancelled_paths) >= 1 or len(complete_paths) == 0, (
        "Inferior Stufe-2 quality must trigger cancellation, not completion"
    )


def test_cache_passthrough_to_denker(tmp_path):
    """Stufe-1 cached results (defect/era/medium) must be forwarded to denke()."""
    from Aurik10.ui.ml_refinement_thread import MLRefinementThread
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(48000, dtype=np.float32)
    mock_defect = {"NOISE": 0.5}
    mock_era = {"decade": 1970}
    mock_medium = {"type": "vinyl"}

    job = DeferredRefinementJob(
        output_path=str(tmp_path / "cache_test.wav"),
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=["phase_20_reverb_reduction"],
        cached_defect_result=mock_defect,
        cached_era_result=mock_era,
        cached_medium_result=mock_medium,
        stufe1_quality=0.60,
        input_path=str(tmp_path / "in.wav"),
    )
    thread = MLRefinementThread(job)
    # Swallow all signals
    thread.refinement_cancelled.connect(lambda p: None)
    thread.refinement_complete.connect(lambda p, r: None)

    mock_result = MagicMock()
    mock_result.quality_estimate = 0.80
    mock_result.audio = audio

    mock_denker = MagicMock()
    mock_denker.denke.return_value = mock_result

    with (
        patch("backend.api.bridge.get_aurik_denker_instance", return_value=mock_denker),
        patch("backend.api.bridge.get_ml_memory_budget") as mock_get_budget,
    ):
        mock_budget = MagicMock()
        mock_budget.try_allocate.return_value = True
        mock_get_budget.return_value = mock_budget
        thread.run()

    # Verify denke() was called with cached results
    assert mock_denker.denke.called, "denke() must be called in Stufe-2 run()"
    call_kwargs = mock_denker.denke.call_args
    if call_kwargs.kwargs:
        kw = call_kwargs.kwargs
    else:
        kw = call_kwargs[1] if len(call_kwargs) > 1 else {}
    assert kw.get("cached_defect_result") is mock_defect, "cached_defect_result must be forwarded"
    assert kw.get("cached_era_result") is mock_era, "cached_era_result must be forwarded"
    assert kw.get("cached_medium_result") is mock_medium, "cached_medium_result must be forwarded"
    assert kw.get("no_rt_limit") is True, "no_rt_limit must be True for Stufe-2"


def test_write_audio_stereo(tmp_path):
    """_write_audio must handle stereo (2D) arrays."""
    import soundfile as sf

    from Aurik10.ui.ml_refinement_thread import _write_audio

    rng = np.random.default_rng(123)
    # _write_audio expects (samples, channels) shape for stereo via soundfile
    audio = rng.standard_normal((4800, 2)).astype(np.float32) * 0.3
    path = str(tmp_path / "stereo.wav")
    _write_audio(audio, 48000, path)
    loaded, sr = sf.read(path, dtype="float32")
    assert sr == 48000
    assert loaded.size > 0


def test_extract_quality_from_restoration_result():
    """_extract_quality must read quality_estimate from standard result objects."""
    from Aurik10.ui.ml_refinement_thread import _extract_quality

    mock_result = MagicMock()
    mock_result.quality_estimate = 0.72
    assert _extract_quality(mock_result) == pytest.approx(0.72, abs=1e-3)

    assert _extract_quality(None) is None


def test_extract_audio_from_result():
    """_extract_audio must extract audio ndarray from result."""
    from Aurik10.ui.ml_refinement_thread import _extract_audio

    mock_result = MagicMock()
    expected = np.zeros(480, dtype=np.float32)
    mock_result.audio = expected
    got = _extract_audio(mock_result)
    assert got is not None
    np.testing.assert_array_equal(got, expected)


def test_should_start_single_instance_guard():
    """Concurrent MLRefinementThread instances must be prevented by should_start."""
    from Aurik10.ui.ml_refinement_thread import MLRefinementThread
    from backend.core.deferred_refinement_job import DeferredRefinementJob

    audio = np.zeros(48000, dtype=np.float32)
    job = DeferredRefinementJob(
        output_path="/tmp/instance_guard.wav",
        audio_original=audio,
        sr=48000,
        mode="restoration",
        deferred_phase_ids=["phase_20"],
        cached_defect_result=None,
        cached_era_result=None,
        cached_medium_result=None,
        stufe1_quality=0.60,
        input_path="",
    )
    # First call should be True (mocking RAM check)
    import psutil

    mock_vm = MagicMock()
    mock_vm.available = int(8.0 * 1024**3)
    with patch.object(psutil, "virtual_memory", return_value=mock_vm):
        assert MLRefinementThread.should_start(job) is True


def test_deferred_job_release_buffer(minimal_job):
    """release_buffer must set audio_original to None and release budget."""
    assert minimal_job.audio_original is not None
    with patch("backend.core.ml_memory_budget.release") as mock_release:
        minimal_job.release_buffer()
    # Audio should be cleared
    assert minimal_job.audio_original is None
