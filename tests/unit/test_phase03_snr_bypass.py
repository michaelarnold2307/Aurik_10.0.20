from __future__ import annotations

import pytest

"""Unit tests — §2.47 Phase_03 SNR > 35 dB Dry-Signal Bypass."""


import numpy as np

from backend.core.phases.phase_03_denoise import DenoisePhase

SR = 48_000


@pytest.mark.unit
def test_phase03_snr_bypass_for_clean_signal() -> None:
    """Clean signal must bypass denoise to avoid unnecessary artefacts."""
    t = np.arange(SR, dtype=np.float32) / SR
    tone = 0.35 * np.sin(2.0 * np.pi * 440.0 * t)
    noise = 0.00005 * np.random.default_rng(7).standard_normal(SR).astype(np.float32)
    # Match the phase_03 frame-percentile SNR estimator with sparse active frames:
    # 10% active tone, 90% near-noise floor.
    audio = noise.copy()
    active_n = SR // 10
    active_start = (SR - active_n) // 2
    audio[active_start : active_start + active_n] += tone[active_start : active_start + active_n]
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    phase = DenoisePhase(sample_rate=SR)
    result = phase.process(audio, material_type="tape", sample_rate=SR, quality_mode="quality")

    assert result.success is True
    assert result.metadata.get("snr_bypass") is True
    assert result.metadata.get("algorithm") == "snr_bypass"
    assert any("SNR > 35 dB" in w for w in result.warnings)
