from __future__ import annotations

"""[RELEASE_MUST] Real-Audio Edge/Lag Gate.

Blockiert Releases, wenn die Pipeline auf realem Audio
1) Intro/Outro-Peak-Explosionen erzeugt oder
2) neue L/R-Zeitverschiebung einführt.
"""


import multiprocessing as mp
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest


def _run_real_audio_restore_child(
    audio: np.ndarray,
    sr: int,
    ml_runtime_budget_s: float,
    payload_path: str,
    error_path: str,
) -> None:
    try:
        os.environ.setdefault("AURIK_SAFE_VALIDATION_PROFILE", "1")
        from backend.core.performance_guard import QualityMode
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

        cfg = RestorationConfig(
            mode=QualityMode.FAST,
            enable_performance_guard=True,
            enable_phase_gate=True,
            enable_phase_skipping=True,
        )
        restorer = UnifiedRestorerV3(config=cfg)
        result = restorer.restore(
            audio,
            sample_rate=sr,
            mode="fast",
            ml_runtime_budget_s=ml_runtime_budget_s,
        )
        np.savez(payload_path, audio=np.asarray(result.audio, dtype=np.float32))
    except Exception:
        Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")


def _run_real_audio_restore_with_timeout(
    audio: np.ndarray,
    sr: int,
    ml_runtime_budget_s: float,
    timeout_s: float,
) -> np.ndarray:
    ctx = mp.get_context("spawn")
    tmp_dir = Path(tempfile.mkdtemp(prefix="aurik_real_audio_restore_"))
    payload_path = tmp_dir / "payload.npz"
    error_path = tmp_dir / "error.txt"
    process = ctx.Process(
        target=_run_real_audio_restore_child,
        args=(audio, sr, ml_runtime_budget_s, str(payload_path), str(error_path)),
        daemon=True,
    )
    try:
        process.start()
        process.join(max(0.0, float(timeout_s)))
        if process.is_alive():
            process.terminate()
            process.join(10.0)
            if process.is_alive():
                process.kill()
                process.join(5.0)
            raise RuntimeError(f"real-audio fixture timeout after {float(timeout_s):.1f}s")

        if error_path.exists():
            raise RuntimeError(error_path.read_text(encoding="utf-8"))
        if not payload_path.exists():
            raise RuntimeError(f"real-audio fixture child exited without payload (exitcode={process.exitcode})")

        with np.load(payload_path, allow_pickle=False) as payload_npz:
            return np.asarray(payload_npz["audio"], dtype=np.float32)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _to_samples_first(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0]:
        return arr.T
    return arr


def _safe_db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def _estimate_lr_delay_samples(audio_sf: np.ndarray, max_lag: int = 256) -> int:
    assert audio_sf.ndim == 2 and audio_sf.shape[1] >= 2
    l = audio_sf[:, 0].astype(np.float64)
    r = audio_sf[:, 1].astype(np.float64)
    l = l - np.mean(l)
    r = r - np.mean(r)
    best_lag = 0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            ls = l[-lag:]
            rs = r[: len(r) + lag]
        elif lag > 0:
            ls = l[: len(l) - lag]
            rs = r[lag:]
        else:
            ls = l
            rs = r
        if len(ls) < 128:
            continue
        denom = float(np.linalg.norm(ls) * np.linalg.norm(rs) + 1e-12)
        corr = float(np.dot(ls, rs) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return int(best_lag)


def _edge_peak_excess_db(audio_sf: np.ndarray, sr: int, edge_s: float = 0.5) -> float:
    n_edge = max(1, int(sr * edge_s))
    n = audio_sf.shape[0]
    edge = np.concatenate([audio_sf[:n_edge, :].ravel(), audio_sf[max(0, n - n_edge) :, :].ravel()])
    core_start = min(n_edge, n // 4)
    core_end = max(core_start + 1, n - n_edge)
    core = audio_sf[core_start:core_end, :].ravel()

    edge_peak = float(np.percentile(np.abs(edge), 99.9))
    core_rms = float(np.sqrt(np.mean(core.astype(np.float64) ** 2) + 1e-12))
    return _safe_db(edge_peak) - _safe_db(core_rms)


@pytest.fixture(scope="module")
def real_audio_edge_lag_case(real_audio_gate_case: dict[str, object]) -> dict[str, Any]:
    original = _to_samples_first(np.asarray(real_audio_gate_case["audio"], dtype=np.float32))
    sr = int(cast(int, real_audio_gate_case["sr"]))

    # Runtime-bounded real-audio window for deterministic gating.
    max_seconds = float(os.environ.get("AURIK_REAL_AUDIO_GATE_MAX_SECONDS", "12") or 12.0)
    ml_runtime_budget_s = float(os.environ.get("AURIK_REAL_AUDIO_GATE_ML_RUNTIME_BUDGET_S", "3.0") or 3.0)
    restore_timeout_s = float(os.environ.get("AURIK_REAL_AUDIO_GATE_RESTORE_TIMEOUT_S", "240") or 240.0)
    retry_max_seconds = float(os.environ.get("AURIK_REAL_AUDIO_GATE_RETRY_MAX_SECONDS", "6") or 6.0)
    retry_ml_runtime_budget_s = float(os.environ.get("AURIK_REAL_AUDIO_GATE_RETRY_ML_RUNTIME_BUDGET_S", "2.0") or 2.0)
    retry_timeout_s = float(os.environ.get("AURIK_REAL_AUDIO_GATE_RETRY_TIMEOUT_S", "300") or 300.0)
    max_n = int(sr * max_seconds)
    if original.shape[0] > max_n:
        start = (original.shape[0] - max_n) // 2
        original = original[start : start + max_n]

    try:
        restored_audio = _run_real_audio_restore_with_timeout(
            original.T,
            sr,
            ml_runtime_budget_s,
            restore_timeout_s,
        )
    except RuntimeError as exc:
        # Retry-Pfad gegen Infrastruktur-Latenz (Model-Init/IO-Spitzen) im großen Gate-Lauf.
        # Gate bleibt streng: nur das Laufzeitfenster wird reduziert, Assertions bleiben unverändert.
        if "timeout" not in str(exc).lower():
            pytest.fail(f"Real-audio edge/lag fixture failed: {exc}")

        retry_n = min(original.shape[0], int(sr * retry_max_seconds))
        retry_original = original
        if retry_n > 0 and original.shape[0] > retry_n:
            retry_start = (original.shape[0] - retry_n) // 2
            retry_original = original[retry_start : retry_start + retry_n]

        try:
            restored_audio = _run_real_audio_restore_with_timeout(
                retry_original.T,
                sr,
                retry_ml_runtime_budget_s,
                retry_timeout_s,
            )
            original = retry_original
        except RuntimeError as retry_exc:
            pytest.fail(f"Real-audio edge/lag fixture timed out (first={exc}; retry={retry_exc}).")
    restored = _to_samples_first(restored_audio)

    n = min(original.shape[0], restored.shape[0])
    original = original[:n]
    restored = restored[:n]

    return {
        "path": str(real_audio_gate_case["path"]),
        "sr": sr,
        "original": original,
        "restored": restored,
    }


@pytest.mark.normative
@pytest.mark.ml
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_real_audio_intro_outro_peak_explosion_gate(real_audio_edge_lag_case: dict[str, Any]) -> None:
    orig = np.asarray(real_audio_edge_lag_case["original"], dtype=np.float32)
    rest = np.asarray(real_audio_edge_lag_case["restored"], dtype=np.float32)
    sr = int(real_audio_edge_lag_case["sr"])

    # Gate needs stereo to validate edge behavior in spatially coupled program material.
    assert orig.ndim == 2 and orig.shape[1] == 2, "Real-Audio-Fixture muss stereo sein"
    assert rest.ndim == 2 and rest.shape[1] == 2, "Restauriertes Audio muss stereo bleiben"

    before = _edge_peak_excess_db(orig, sr)
    after = _edge_peak_excess_db(rest, sr)
    delta = float(after - before)

    # Allow a small tolerance for legitimate processing, block clear edge explosions.
    assert delta <= 2.0, (
        f"Edge-Peak-Explosion erkannt: excess before={before:.2f} dB, after={after:.2f} dB, "
        f"delta={delta:.2f} dB on {real_audio_edge_lag_case['path']}"
    )


@pytest.mark.normative
@pytest.mark.ml
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_real_audio_interchannel_delay_no_regress_gate(real_audio_edge_lag_case: dict[str, Any]) -> None:
    orig = np.asarray(real_audio_edge_lag_case["original"], dtype=np.float32)
    rest = np.asarray(real_audio_edge_lag_case["restored"], dtype=np.float32)

    assert orig.ndim == 2 and orig.shape[1] == 2, "Real-Audio-Fixture muss stereo sein"
    assert rest.ndim == 2 and rest.shape[1] == 2, "Restauriertes Audio muss stereo bleiben"

    lag_before = _estimate_lr_delay_samples(orig)
    lag_after = _estimate_lr_delay_samples(rest)
    lag_delta = abs(lag_after - lag_before)

    # Keep strictly below the 1 ms hard-fail envelope; gate at 8 samples (~0.17 ms @ 48 kHz).
    assert lag_delta <= 8, (
        f"Interchannel-Delay-Regression: before={lag_before} samples, after={lag_after} samples, "
        f"delta={lag_delta} on {real_audio_edge_lag_case['path']}"
    )
