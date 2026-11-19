"""Tests für H1+H2: Masking-Fusion + Musical-Noise-Gate (§Witness-SOTA).

Invarianten: Never-worsen (nie Energie hinzufügen), deterministisch,
Layout-agnostisch, Passthrough bei identischem Kandidat.
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.hybrid_denoise_fusion import masked_denoise_fusion, musical_noise_gate

SR = 22050


def _tone_hiss(
    sr: int = SR, dur: float = 2.0, noise_gain: float = 0.05, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    clean = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    hiss = (rng.standard_normal(len(t)) * noise_gain).astype(np.float32)
    return clean, clean + hiss


def test_passthrough_identical_candidate() -> None:
    clean, _ = _tone_hiss()
    fused, report = masked_denoise_fusion(clean.copy(), clean, SR)
    assert fused.shape == clean.shape
    assert np.allclose(fused, clean, atol=1e-4)
    assert report["never_worsen"] is True


def test_never_worsen_louder_candidate_keeps_baseline() -> None:
    clean, _ = _tone_hiss()
    baseline = clean
    candidate = clean + 0.2 * np.sin(2 * np.pi * 3000.0 * np.arange(len(clean)) / SR).astype(np.float32)
    fused, report = masked_denoise_fusion(candidate, baseline, SR)
    # Kandidat ist lauter → darf sich nirgends durchsetzen.
    assert report["mean_blend"] < 0.05
    assert np.allclose(fused, baseline, atol=1e-4)


def test_fusion_takes_audibly_quieter_candidate() -> None:
    clean, noisy = _tone_hiss(noise_gain=0.2, seed=11)
    # Kandidat = stark entrauscht (deutlich leiser im Rauschband) —
    # die Fusion darf dessen leisere Bänder übernehmen.
    fused, report = masked_denoise_fusion(clean, noisy, SR)
    assert report["mean_blend"] > 0.3, f"Erwartete deutliche Übernahme, mean_blend={report['mean_blend']}"
    # Ergebnis darf nirgends lauter als das Baseline-Noisy sein (Never-worsen).
    assert np.mean(np.abs(fused)) <= np.mean(np.abs(noisy)) + 1e-6


def test_stereo_layout_preserved_both_orientations() -> None:
    clean, noisy = _tone_hiss(noise_gain=0.15, seed=23)
    stereo_cn = np.stack([noisy, noisy * 0.9]).astype(np.float32)  # (C, N)
    clean_cn = np.stack([clean, clean * 0.9]).astype(np.float32)
    fused, _ = masked_denoise_fusion(clean_cn, stereo_cn, SR)
    assert fused.ndim == 2 and fused.shape[0] == 2
    # (N, C)-Layout wird normalisiert und zurücktransponiert.
    fused_t, _ = masked_denoise_fusion(clean_cn.T, stereo_cn.T, SR)
    assert fused_t.shape == stereo_cn.T.shape
    assert np.allclose(np.abs(fused), np.abs(fused_t).T, atol=1e-3)


def test_determinism_fusion_and_gate() -> None:
    clean, noisy = _tone_hiss(noise_gain=0.1, seed=31)
    f1, r1 = masked_denoise_fusion(clean, noisy, SR)
    f2, r2 = masked_denoise_fusion(clean, noisy, SR)
    assert np.array_equal(f1, f2) and r1 == r2
    g1, _ = musical_noise_gate(f1, SR)
    g2, _ = musical_noise_gate(f1, SR)
    assert np.array_equal(g1, g2)


def test_musical_noise_gate_never_increases_energy() -> None:
    clean, noisy = _tone_hiss(noise_gain=0.15, seed=41)
    gated, report = musical_noise_gate(noisy, SR)
    assert report["never_worsen"] is True
    assert np.mean(np.abs(gated)) <= np.mean(np.abs(noisy)) + 1e-6
    # Rauschen wird gedämpft, Ton bleibt.
    assert np.mean(np.abs(gated)) < np.mean(np.abs(noisy))
    assert np.allclose(np.max(np.abs(gated)), np.max(np.abs(noisy)), rtol=0.15)


def test_musical_noise_gate_mono_and_stereo() -> None:
    _, noisy = _tone_hiss(noise_gain=0.12, seed=53)
    g_mono, _ = musical_noise_gate(noisy, SR)
    assert g_mono.shape == noisy.shape
    stereo = np.stack([noisy, noisy]).astype(np.float32)
    g_stereo, _ = musical_noise_gate(stereo, SR)
    assert g_stereo.shape == stereo.shape
