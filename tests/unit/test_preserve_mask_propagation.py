from __future__ import annotations

"""
tests/unit/test_preserve_mask_propagation.py
Aurik 10.0.0 — §2.44 §4.8a-ii: preserve_mask Erzeugung und Propagation.

Testet:
- get_preserve_mask() gibt ndarray zurück (non-None, nie Fehler)
- PRESERVE-Bins haben erwartete Werte > 0
- Zero-Maske für leere preserve_features
- shape = 1025 (n_fft=2048 standard)
- Werte im gültigen Bereich [0, 1]
"""


import numpy as np
import pytest


@pytest.fixture
def iac():
    from backend.core.intentional_artifact_classifier import get_intentional_artifact_classifier

    return get_intentional_artifact_classifier()


@pytest.fixture
def dummy_audio():
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, 48000).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Rückgabe-Typ + Shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_preserve_mask_returns_ndarray_for_shellac(iac, dummy_audio):
    result = iac.get_preserve_mask(dummy_audio, 48000, material_type="shellac")
    assert isinstance(result, np.ndarray), "Rückgabe muss np.ndarray sein"


def test_get_preserve_mask_shape_is_1025(iac, dummy_audio):
    for mat in ("shellac", "vinyl", "tape", "cd_digital", "mp3_low"):
        mask = iac.get_preserve_mask(dummy_audio, 48000, material_type=mat)
        assert mask.shape == (1025,), f"Material {mat}: shape={mask.shape} erwartet (1025,)"


def test_get_preserve_mask_dtype_is_float32(iac, dummy_audio):
    mask = iac.get_preserve_mask(dummy_audio, 48000, material_type="vinyl")
    assert mask.dtype == np.float32


def test_get_preserve_mask_values_in_range_0_1(iac, dummy_audio):
    for mat in ("shellac", "vinyl", "tape", "reel_tape", "wax_cylinder", "cd_digital", "mp3_low", "lacquer_disc"):
        mask = iac.get_preserve_mask(dummy_audio, 48000, material_type=mat)
        assert float(mask.min()) >= 0.0, f"Material {mat}: mask < 0 gefunden"
        assert float(mask.max()) <= 1.0 + 1e-6, f"Material {mat}: mask > 1 gefunden"


# ---------------------------------------------------------------------------
# 2. PRESERVE-Feature-Abdeckung
# ---------------------------------------------------------------------------


def test_shellac_h2_h4_bins_are_protected(iac, dummy_audio):
    """shellac h2_h4_harmonic_saturation → Bins 2–8 kHz müssen mask > 0 haben."""
    mask = iac.get_preserve_mask(dummy_audio, 48000, material_type="shellac")
    # 2 kHz ≈ bin 85, 8 kHz ≈ bin 341 @ 48kHz n_fft=2048
    freqs = np.linspace(0, 24000, 1025)
    lo = int(np.searchsorted(freqs, 2000))
    hi = int(np.searchsorted(freqs, 8000))
    assert mask[lo:hi].max() >= 0.79, "H2/H4-Bereich muss Maske ≥ 0.79 haben"


def test_shellac_above_8khz_is_fully_preserved(iac, dummy_audio):
    """bandwidth_ceiling_8khz → alle Bins > 8 kHz: mask = 1.0."""
    mask = iac.get_preserve_mask(dummy_audio, 48000, material_type="shellac")
    freqs = np.linspace(0, 24000, 1025)
    lo = int(np.searchsorted(freqs, 8000))
    assert float(mask[lo:].min()) >= 0.99, "Bins > 8kHz (BW-Ceiling) müssen mask=1.0 haben"


def test_wax_cylinder_above_3khz_is_fully_preserved(iac, dummy_audio):
    """trichter_bandlimit_3khz → Bins > 3 kHz: mask = 1.0."""
    mask = iac.get_preserve_mask(dummy_audio, 48000, material_type="wax_cylinder")
    freqs = np.linspace(0, 24000, 1025)
    lo = int(np.searchsorted(freqs, 3000))
    assert float(mask[lo:].min()) >= 0.99, "Bins > 3kHz (Trichter-Ceiling) müssen mask=1.0 haben"


def test_cd_digital_mask_very_low(iac, dummy_audio):
    """CD hat kaum PRESERVE-Merkmale → mask.max() ≤ 0.35 (nur Dithering/Linear-Phase)."""
    mask = iac.get_preserve_mask(dummy_audio, 48000, material_type="cd_digital")
    assert float(mask.max()) <= 0.35, f"CD mask.max()={mask.max():.3f} > 0.35 (zu aggressiv)"


def test_tape_hf_bins_protected(iac, dummy_audio):
    """tape: bias_noise_texture → Bins > 15 kHz: mask ≥ 0.79."""
    mask = iac.get_preserve_mask(dummy_audio, 48000, material_type="tape")
    freqs = np.linspace(0, 24000, 1025)
    lo = int(np.searchsorted(freqs, 15000))
    assert float(mask[lo:].max()) >= 0.79, "Tape HF-Bereich (Bias-Noise) muss mask ≥ 0.79 haben"


# ---------------------------------------------------------------------------
# 3. iac_result Parameter (§2.44 API)
# ---------------------------------------------------------------------------


def test_get_preserve_mask_accepts_iac_result(iac, dummy_audio):
    """get_preserve_mask() akzeptiert IntentionalArtifactResult aus classify()."""
    iac_result = iac.classify("shellac")
    mask = iac.get_preserve_mask(dummy_audio, 48000, iac_result=iac_result)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (1025,)


def test_iac_result_and_material_type_give_same_mask(iac, dummy_audio):
    """iac_result und material_type-Pfad müssen identisches Ergebnis liefern."""
    iac_result = iac.classify("vinyl")
    mask_from_result = iac.get_preserve_mask(dummy_audio, 48000, iac_result=iac_result)
    mask_from_type = iac.get_preserve_mask(dummy_audio, 48000, material_type="vinyl")
    np.testing.assert_array_almost_equal(mask_from_result, mask_from_type, decimal=5)


# ---------------------------------------------------------------------------
# 4. Robustheit / non-blocking
# ---------------------------------------------------------------------------


def test_get_preserve_mask_never_returns_none(iac, dummy_audio):
    """Auch bei ungültigem Material darf keine Exception geworfen werden."""
    mask = iac.get_preserve_mask(dummy_audio, 48000, material_type="unknown_xyz")
    assert mask is not None
    assert isinstance(mask, np.ndarray)


def test_get_preserve_mask_no_args_returns_zero_mask(iac, dummy_audio):
    """Ohne iac_result und material_type → Zero-Maske, kein Crash."""
    mask = iac.get_preserve_mask(dummy_audio, 48000)
    assert isinstance(mask, np.ndarray)
    assert np.all(mask == 0.0)


def test_get_preserve_mask_empty_preserve_features_gives_zero_mask(iac, dummy_audio):
    """material_type=cd_digital: kaum PRESERVE-Features → max ≤ 0.35."""
    # Direkt mit leerem IntentionalArtifactResult testen
    from backend.core.intentional_artifact_classifier import IntentionalArtifactResult

    empty_result = IntentionalArtifactResult(material_type="cd_digital")  # preserve_features=[]
    mask = iac.get_preserve_mask(dummy_audio, 48000, iac_result=empty_result)
    assert np.all(mask == 0.0), "Leere preserve_features → Zero-Maske erwartet"


def test_get_preserve_mask_no_nan_or_inf(iac, dummy_audio):
    """Keine NaN/Inf in der Maske."""
    for mat in ("shellac", "vinyl", "tape", "reel_tape", "wax_cylinder", "cd_digital"):
        mask = iac.get_preserve_mask(dummy_audio, 48000, material_type=mat)
        assert np.all(np.isfinite(mask)), f"Material {mat}: NaN/Inf in Maske gefunden"
