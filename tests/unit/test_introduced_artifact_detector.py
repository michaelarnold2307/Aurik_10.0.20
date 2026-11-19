"""
tests/unit/test_introduced_artifact_detector.py — IntroducedArtifactDetector Test-Suite (≥ 25 Tests)
Alle Tests synthetisch, kein ML-Modell-Download erforderlich.
"""

import math

import numpy as np
import pytest

SR = 48_000
np.random.seed(42)


def _audio(dur: float = 3.0, amp: float = 0.3):
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silence(dur: float = 3.0):
    return np.zeros(int(dur * SR), dtype=np.float32)


def _add_click(audio):
    """Fügt synthetischen Click zum Signal hinzu."""
    result = audio.copy()
    pos = len(audio) // 3
    result[pos] += 0.8
    result[pos + 1] += -0.6
    return result


def _add_musical_noise(audio):
    """Fügt Rausch-Burst in Stille-Bereich ein."""
    result = audio.copy()
    start = len(audio) // 2
    result[start : start + 200] += 0.3 * np.random.randn(200).astype(np.float32)
    return np.clip(result, -1.0, 1.0)


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.introduced_artifact_detector import (
        IntroducedArtifactDetector,
    )

    assert IntroducedArtifactDetector is not None


def test_01_detect_returns_result():
    from backend.core.introduced_artifact_detector import IADResult, detect_introduced_artifacts

    orig = _audio(3.0)
    rest = _audio(3.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert isinstance(r, IADResult)


def test_02_identical_signals_no_artifacts():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    audio = _audio(3.0)
    r = detect_introduced_artifacts(audio, audio.copy(), SR)
    assert not r.has_artifacts
    assert r.total_contaminated_fraction <= 0.15


def test_03_no_nan_fraction():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = _add_click(orig)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert math.isfinite(r.total_contaminated_fraction)


def test_04_fraction_bounded():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = _audio(3.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert 0.0 <= r.total_contaminated_fraction <= 1.0


def test_05_has_artifacts_bool():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = orig.copy()
    r = detect_introduced_artifacts(orig, rest, SR)
    assert isinstance(r.has_artifacts, bool)


def test_06_silence_no_crash():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _silence(3.0)
    rest = _silence(3.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert math.isfinite(r.total_contaminated_fraction)


def test_07_singleton_identity():
    from backend.core.introduced_artifact_detector import get_iad

    a = get_iad()
    b = get_iad()
    assert a is b


def test_08_thread_safe():
    import concurrent.futures

    from backend.core.introduced_artifact_detector import get_iad

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(get_iad) for _ in range(10)]
        instances = [f.result() for f in futs]
    assert all(inst is instances[0] for inst in instances)


def test_09_artifact_types_list():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = _audio(3.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert isinstance(r.artifact_types, list)


def test_10_get_artifact_mask_shape():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts, get_iad

    orig = _audio(3.0)
    rest = _audio(3.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    iad = get_iad()
    mask = iad.get_artifact_mask(r, len(orig))
    assert mask.shape == (len(orig),)


def test_11_artifact_mask_bool():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts, get_iad

    orig = _audio(3.0)
    rest = _audio(3.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    mask = get_iad().get_artifact_mask(r, len(orig))
    assert mask.dtype == bool


def test_12_assert_sr():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(2.0)
    rest = _audio(2.0)
    with pytest.raises((AssertionError, ValueError)):
        detect_introduced_artifacts(orig, rest, 44100)


def test_13_different_length_resilience():
    """Gleiche Länge (Pflicht) — Test zur Grenzprüfung."""
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = orig.copy()
    r = detect_introduced_artifacts(orig, rest, SR)
    assert math.isfinite(r.total_contaminated_fraction)


def test_14_high_amplitude_restored():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0, amp=0.3)
    rest = _audio(3.0, amp=0.9)  # Lautstärke stark erhöht
    r = detect_introduced_artifacts(orig, rest, SR)
    assert math.isfinite(r.total_contaminated_fraction)


def test_15_white_noise_restored():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = _audio(3.0) + 0.2 * np.random.randn(len(_audio(3.0))).astype(np.float32)
    rest = np.clip(rest, -1.0, 1.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert math.isfinite(r.total_contaminated_fraction)


def test_16_detect_method_directly():
    from backend.core.introduced_artifact_detector import get_iad

    orig = _audio(3.0)
    rest = _audio(3.0)
    r = get_iad().detect(orig, rest, SR)
    assert math.isfinite(r.total_contaminated_fraction)


def test_17_fraction_close_to_zero_identical():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    audio = _audio(5.0)
    r = detect_introduced_artifacts(audio, audio.copy(), SR)
    assert r.total_contaminated_fraction <= 0.25


def test_18_impulse_signal():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = np.zeros(SR * 3, dtype=np.float32)
    rest = np.zeros(SR * 3, dtype=np.float32)
    rest[SR // 2] = 0.9  # Introduced click
    r = detect_introduced_artifacts(orig, rest, SR)
    assert math.isfinite(r.total_contaminated_fraction)


def test_19_mask_length_correct():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts, get_iad

    orig = _audio(4.0)
    rest = _audio(4.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    for n in [len(orig), len(orig) // 2, SR]:
        mask = get_iad().get_artifact_mask(r, n)
        assert mask.shape == (n,)


def test_20_no_negative_fraction():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = orig.copy() + 0.01 * np.random.randn(len(orig)).astype(np.float32)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert r.total_contaminated_fraction >= 0.0


def test_21_consistent_results():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = _audio(3.0)
    r1 = detect_introduced_artifacts(orig.copy(), rest.copy(), SR)
    r2 = detect_introduced_artifacts(orig.copy(), rest.copy(), SR)
    assert abs(r1.total_contaminated_fraction - r2.total_contaminated_fraction) < 1e-4


def test_22_result_has_key_fields():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = _audio(3.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert hasattr(r, "has_artifacts")
    assert hasattr(r, "total_contaminated_fraction")
    assert hasattr(r, "artifact_types")


def test_23_float64_input():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0).astype(np.float64)
    rest = _audio(3.0).astype(np.float64)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert math.isfinite(r.total_contaminated_fraction)


def test_24_confidence_field():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = _audio(3.0)
    rest = _audio(3.0)
    r = detect_introduced_artifacts(orig, rest, SR)
    if hasattr(r, "confidence"):
        assert math.isfinite(r.confidence)
        assert 0.0 <= r.confidence <= 1.0


def test_25_very_long_audio():
    from backend.core.introduced_artifact_detector import detect_introduced_artifacts

    orig = (np.random.randn(SR * 30) * 0.2).astype(np.float32)
    rest = orig.copy() + 0.01 * np.random.randn(len(orig)).astype(np.float32)
    r = detect_introduced_artifacts(orig, rest, SR)
    assert math.isfinite(r.total_contaminated_fraction)
    assert 0.0 <= r.total_contaminated_fraction <= 1.0


def test_26_ml_hallucination_window_cap_limits_runtime_cost(monkeypatch):
    from backend.core.introduced_artifact_detector import IntroducedArtifactDetector

    iad = IntroducedArtifactDetector()
    # Force every frame through harmonicity path.
    iad.HALLUCINATION_MIN_RMS = 0.0
    iad.MAX_HALLUCINATION_WINDOWS = 5

    calls = {"n": 0}

    def _fake_harmonicity(frame, sr):
        calls["n"] += 1
        return 0.0

    monkeypatch.setattr(iad, "_harmonicity", _fake_harmonicity)
    residuum = (np.random.randn(SR * 120) * 0.1).astype(np.float32)  # 2 min

    _ = iad._detect_ml_hallucinations(residuum, residuum, SR)
    assert calls["n"] <= iad.MAX_HALLUCINATION_WINDOWS


def test_27_pvoc_smearing_threshold_tightens_for_transient_dense_material():
    from backend.core.introduced_artifact_detector import IntroducedArtifactDetector

    iad = IntroducedArtifactDetector()

    duration_s = 3.0
    n = int(duration_s * SR)
    shift = 1024  # ~21.3 ms @ 48 kHz; should exceed 15 ms but not 50 ms

    # High transient density: pulses every 0.2 s (~5 events/s)
    orig_dense = np.zeros(n, dtype=np.float32)
    for pos in range(SR // 10, n - 64, int(0.2 * SR)):
        orig_dense[pos : pos + 64] = 0.9
    rest_dense = np.zeros_like(orig_dense)
    for pos in range(SR // 10 + shift, n - 64, int(0.2 * SR)):
        rest_dense[pos : pos + 64] = 0.9

    # Low transient density: one isolated event (~0.33 events/s)
    orig_sparse = np.zeros(n, dtype=np.float32)
    pos_sparse = n // 2
    orig_sparse[pos_sparse : pos_sparse + 64] = 0.9
    rest_sparse = np.zeros_like(orig_sparse)
    rest_sparse[pos_sparse + shift : pos_sparse + shift + 64] = 0.9

    dense_artifacts = iad._detect_pvoc_smearing(orig_dense, rest_dense, SR)
    sparse_artifacts = iad._detect_pvoc_smearing(orig_sparse, rest_sparse, SR)

    assert len(dense_artifacts) >= 1
    assert len(sparse_artifacts) == 0
