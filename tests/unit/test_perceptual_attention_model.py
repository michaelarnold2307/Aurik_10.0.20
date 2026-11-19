"""
tests/unit/test_perceptual_attention_model.py — PerceptualAttentionModel Test-Suite (≥ 20 Tests)
Alle Tests synthetisch, kein ML-Modell-Download erforderlich.
"""

import numpy as np
import pytest

SR = 48_000
np.random.seed(42)


def _audio(dur: float = 3.0, amp: float = 0.3):
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silence(dur: float = 3.0):
    return np.zeros(int(dur * SR), dtype=np.float32)


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.perceptual_attention_model import PerceptualAttentionModel

    assert PerceptualAttentionModel is not None


def test_01_compute_saliency_returns_ndarray():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    pam = get_perceptual_attention_model()
    smap = pam.compute_saliency_map(audio, SR)
    assert isinstance(smap, np.ndarray)


def test_02_saliency_shape():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    pam = get_perceptual_attention_model()
    smap = pam.compute_saliency_map(audio, SR)
    assert smap.ndim == 2
    n_frames, n_bands = smap.shape
    assert n_bands == 24  # 24 Bark-Bänder


def test_03_saliency_bounds():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    smap = get_perceptual_attention_model().compute_saliency_map(audio, SR)
    assert np.all(smap >= 0.3 - 1e-6)
    assert np.all(smap <= 2.0 + 1e-6)


def test_04_no_nan_in_saliency():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    smap = get_perceptual_attention_model().compute_saliency_map(audio, SR)
    assert np.isfinite(smap).all()


def test_05_silence_input():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _silence(3.0)
    smap = get_perceptual_attention_model().compute_saliency_map(audio, SR)
    assert np.isfinite(smap).all()
    assert smap.ndim == 2


def test_06_singleton_identity():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    a = get_perceptual_attention_model()
    b = get_perceptual_attention_model()
    assert a is b


def test_07_thread_safe():
    import concurrent.futures

    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(get_perceptual_attention_model) for _ in range(10)]
        instances = [f.result() for f in futs]
    assert all(inst is instances[0] for inst in instances)


def test_08_apply_to_gain_shape():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    pam = get_perceptual_attention_model()
    smap = pam.compute_saliency_map(audio, SR)
    n_frames = smap.shape[0]
    base_gain = np.ones(n_frames, dtype=np.float32)
    result = pam.apply_to_gain(base_gain, smap)
    assert result.shape == base_gain.shape


def test_09_apply_to_gain_no_nan():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    pam = get_perceptual_attention_model()
    smap = pam.compute_saliency_map(audio, SR)
    n_frames = smap.shape[0]
    base_gain = np.ones(n_frames, dtype=np.float32) * 0.7
    result = pam.apply_to_gain(base_gain, smap)
    assert np.isfinite(result).all()


def test_10_apply_to_gain_positive():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    pam = get_perceptual_attention_model()
    smap = pam.compute_saliency_map(audio, SR)
    n_frames = smap.shape[0]
    base_gain = np.ones(n_frames, dtype=np.float32)
    result = pam.apply_to_gain(base_gain, smap)
    assert np.all(result >= 0.0)


def test_11_dtype_float32():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    smap = get_perceptual_attention_model().compute_saliency_map(audio, SR)
    assert smap.dtype == np.float32


def test_12_very_short_audio():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(0.3)
    smap = get_perceptual_attention_model().compute_saliency_map(audio, SR)
    assert np.isfinite(smap).all()


def test_13_white_noise_input():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = (np.random.randn(SR * 3) * 0.1).astype(np.float32)
    smap = get_perceptual_attention_model().compute_saliency_map(audio, SR)
    assert np.isfinite(smap).all()
    assert np.all(smap >= 0.29)


def test_14_assert_sr():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(2.0)
    with pytest.raises((AssertionError, ValueError)):
        get_perceptual_attention_model().compute_saliency_map(audio, 44100)


def test_15_saliency_columns_24():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(5.0)
    smap = get_perceptual_attention_model().compute_saliency_map(audio, SR)
    assert smap.shape[1] == 24


def test_16_n_frames_positive():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    smap = get_perceptual_attention_model().compute_saliency_map(audio, SR)
    assert smap.shape[0] > 0


def test_17_apply_to_gain_zero_base():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    pam = get_perceptual_attention_model()
    smap = pam.compute_saliency_map(audio, SR)
    n_frames = smap.shape[0]
    base_gain = np.zeros(n_frames, dtype=np.float32)
    result = pam.apply_to_gain(base_gain, smap)
    assert np.isfinite(result).all()


def test_18_impulse_signal():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = np.zeros(SR * 3, dtype=np.float32)
    audio[SR] = 0.9
    smap = get_perceptual_attention_model().compute_saliency_map(audio, SR)
    assert np.isfinite(smap).all()


def test_19_bands_24_for_varying_durations():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    pam = get_perceptual_attention_model()
    for dur in [1.0, 3.0, 10.0]:
        audio = _audio(dur)
        smap = pam.compute_saliency_map(audio, SR)
        assert smap.shape[1] == 24, f"Bands ≠ 24 für dur={dur}"
        assert np.isfinite(smap).all()


def test_20_base_gain_float64():
    from backend.core.perceptual_attention_model import get_perceptual_attention_model

    audio = _audio(3.0)
    pam = get_perceptual_attention_model()
    smap = pam.compute_saliency_map(audio, SR)
    n_frames = smap.shape[0]
    base_gain = np.ones(n_frames, dtype=np.float64)
    result = pam.apply_to_gain(base_gain, smap)
    assert np.isfinite(result).all()
