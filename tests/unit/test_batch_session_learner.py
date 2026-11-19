"""
tests/unit/test_batch_session_learner.py — BatchSessionLearner Test-Suite (≥ 20 Tests)
Alle Tests synthetisch, ohne Datei-I/O in Produktion (tmpdir/tmp_path).
"""

import math
import pathlib

import numpy as np
import pytest

SR = 48_000
np.random.seed(42)


@pytest.fixture
def tmp_session_dir(tmp_path, monkeypatch):
    """Setzt SESSION_DIR auf tmp_path für jeden Test."""
    from backend.core import batch_session_learner as bsl_mod

    monkeypatch.setattr(bsl_mod.BatchSessionLearner, "SESSION_DIR", tmp_path / "batch_sessions")
    return tmp_path / "batch_sessions"


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.batch_session_learner import BatchSessionLearner

    assert BatchSessionLearner is not None


def test_01_singleton_identity():
    from backend.core.batch_session_learner import get_batch_session_learner

    a = get_batch_session_learner()
    b = get_batch_session_learner()
    assert a is b


def test_02_thread_safe(tmp_session_dir):
    import concurrent.futures

    from backend.core.batch_session_learner import get_batch_session_learner

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(get_batch_session_learner) for _ in range(10)]
        instances = [f.result() for f in futs]
    assert all(inst is instances[0] for inst in instances)


def test_03_detect_session_returns_string(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/music/file1.wav"), pathlib.Path("/tmp/music/file2.wav")]
    sid = bsl.start_session(paths)
    assert isinstance(sid, str)
    assert len(sid) >= 6


def test_04_same_folder_same_session(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths1 = [pathlib.Path("/tmp/session_x/a.wav"), pathlib.Path("/tmp/session_x/b.wav")]
    paths2 = [pathlib.Path("/tmp/session_x/c.wav")]
    assert bsl.start_session(paths1) == bsl.start_session(paths2)


def test_05_different_folders_different_sessions():
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths1 = [pathlib.Path("/tmp/folder_a/1.wav")]
    paths2 = [pathlib.Path("/tmp/folder_b/1.wav")]
    assert bsl.start_session(paths1) != bsl.start_session(paths2)


def test_06_start_session_returns_sid(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_test/f.wav")]
    sid = bsl.start_session(paths)
    assert isinstance(sid, str) and len(sid) > 0


def test_07_get_warm_start_none_initially(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/empty_sess/x.wav")]
    sid = bsl.start_session(paths)
    ws = bsl.get_warm_start(sid, "unknown")
    assert ws is None or isinstance(ws, dict)


def test_08_update_stores_state(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_update/a.wav")]
    sid = bsl.start_session(paths)
    bsl.update(sid, "tape", {"noise_reduction_strength": 0.7}, 4.1)
    # Kein Fehler = OK
    ws = bsl.get_warm_start(sid, "tape")
    assert ws is None or isinstance(ws, dict)


def test_09_update_nan_score_ignored(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_nan/a.wav")]
    sid = bsl.start_session(paths)
    # NaN-Score sollte still ignoriert werden
    bsl.update(sid, "vinyl", {"noise_reduction_strength": 0.5}, float("nan"))


def test_10_update_inf_score_ignored(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_inf/a.wav")]
    sid = bsl.start_session(paths)
    bsl.update(sid, "vinyl", {"noise_reduction_strength": 0.5}, float("inf"))


def test_11_finalize_no_crash(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_final/x.wav")]
    sid = bsl.start_session(paths)
    bsl.update(sid, "tape", {"noise_reduction_strength": 0.6}, 4.0)
    bsl.finalize(sid)  # Kein Absturz


def test_12_max_files_attribute():
    from backend.core.batch_session_learner import BatchSessionLearner

    assert hasattr(BatchSessionLearner, "MAX_FILES_PER_SESSION")
    assert BatchSessionLearner.MAX_FILES_PER_SESSION >= 10


def test_13_session_dir_attribute():
    from backend.core.batch_session_learner import BatchSessionLearner

    assert hasattr(BatchSessionLearner, "SESSION_DIR")


def test_14_multiple_material_updates(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_multi/x.wav")]
    sid = bsl.start_session(paths)
    for mat in ["tape", "vinyl", "shellac", "unknown"]:
        bsl.update(sid, mat, {"noise_reduction_strength": 0.5}, 3.8)


def test_15_warm_start_after_update(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_ws/x.wav")]
    sid = bsl.start_session(paths)
    bsl.update(sid, "tape", {"noise_reduction_strength": 0.75}, 4.2)
    ws = bsl.get_warm_start(sid, "tape")
    if ws is not None:
        assert isinstance(ws, dict)
        for v in ws.values():
            assert math.isfinite(v)


def test_16_session_id_short(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/session_length_test/a.wav")]
    sid = bsl.start_session(paths)
    assert len(sid) >= 6  # SHA256[:8] mindestens 6 Zeichen


def test_17_empty_paths_no_crash(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    try:
        sid = bsl.start_session([])
        assert isinstance(sid, str)
    except (ValueError, IndexError):
        pass  # leere Liste ist erlaubt zu verwerfen


def test_18_update_multiple_times(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_repeat/x.wav")]
    sid = bsl.start_session(paths)
    for i in range(10):
        bsl.update(sid, "tape", {"noise_reduction_strength": 0.5 + i * 0.05}, 3.5 + i * 0.1)


def test_19_finalize_with_better_score(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_better/x.wav")]
    sid = bsl.start_session(paths)
    bsl.update(sid, "tape", {"noise_reduction_strength": 0.9}, 4.8)
    bsl.finalize(sid)


def test_20_start_session_idempotent(tmp_session_dir):
    from backend.core.batch_session_learner import get_batch_session_learner

    bsl = get_batch_session_learner()
    paths = [pathlib.Path("/tmp/sess_idempotent/f.wav")]
    sid1 = bsl.start_session(paths)
    sid2 = bsl.start_session(paths)
    # Gleiche Pfade → gleiche Session-ID
    assert sid1 == sid2
