"""Tests für das Additive-Synthesis-Gate (FlashSR/BigVGAN-Absicherung, §B4/B5).

Invarianten: Never-worsen (nie Energie über max(Baseline, Schwelle + 1 dB)),
hörbarer Original-Inhalt bleibt unangetastet, Onset-Frames geschützt,
deterministisch, layout-agnostisch.
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.additive_synthesis_gate import additive_synthesis_gate

SR = 22050


def _lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, cutoff / (sr / 2.0), btype="low", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def _band_rms(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [lo / (sr / 2.0), min(hi / (sr / 2.0), 0.99)], btype="band", output="sos")
    y = sosfiltfilt(sos, x)
    return float(np.sqrt(np.mean(y**2)))


def test_passthrough_identical_candidate() -> None:
    rng = np.random.default_rng(3)
    x = (0.4 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR) + 0.05 * rng.standard_normal(SR)).astype(np.float32)
    gated, report = additive_synthesis_gate(x.copy(), x, SR)
    assert np.allclose(gated, x, atol=1e-4)
    assert report["never_worsen"] is True


def test_audible_content_untouched() -> None:
    """Breitbandiger Baseline: Synthese-Energie darf nie hörbar über dem Baseline liegen."""
    rng = np.random.default_rng(5)
    base = (rng.standard_normal(SR) * 0.1).astype(np.float32)  # Breitband → überall hörbar
    cand = base + 0.1 * np.sin(2 * np.pi * 8000 * np.arange(SR) / SR).astype(np.float32)
    gated, report = additive_synthesis_gate(cand, base, SR)
    # Never-worsen gegenüber dem KANDIDATEN (Deckelung) und gegenüber dem
    # Baseline + Schwelle: der 8-kHz-Ton darf nicht hörbar über dem
    # Baseline-Rauschpegel landen.
    e_g = float(np.mean(gated**2))
    e_b = float(np.mean(base**2))
    e_c = float(np.mean(cand**2))
    assert e_g <= e_c + 1e-6
    assert e_g <= e_b * 2.0 + 1e-3, f"Synthese über Schwelle: {e_g:.6f} vs {e_b:.6f}"
    assert report["never_worsen"] is True


def test_hf_synthesis_released_only_in_inaudible_bands() -> None:
    """Tiefpass-Baseline: HF-Synthese wird freigegeben, aber auf die Schwelle gedeckelt."""
    rng = np.random.default_rng(7)
    t = np.arange(SR * 2) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    base = _lowpass(tone + 0.02 * rng.standard_normal(len(t)).astype(np.float32), SR, 4000.0)
    # Kandidat: Baseline + LAUTES synthetisches HF-Rauschen (10 kHz).
    hf = (rng.standard_normal(len(t)) * 0.3).astype(np.float32)
    cand = base + hf
    gated, report = additive_synthesis_gate(cand, base, SR)
    # HF-Band des Gates darf nicht die Kandidat-Lautstärke übernehmen.
    gated_hf = _band_rms(gated, SR, 9000.0, 12000.0)
    cand_hf = _band_rms(cand, SR, 9000.0, 12000.0)
    assert gated_hf < cand_hf, f"HF muss gedeckelt sein: gated={gated_hf:.4f} cand={cand_hf:.4f}"
    # Und darf die Kandidat-Lautstärke nur anteilig übernehmen (Deckel auf die
    # Maskierungsschwelle — HF-Synthese bleibt weit unter dem Kandidaten).
    assert gated_hf < cand_hf * 0.5, f"HF nicht ausreichend gedeckelt: {gated_hf:.6f} vs {cand_hf:.6f}"
    # Tiefer Bandbereich (hörbarer Inhalt) bleibt unangetastet.
    assert np.allclose(_band_rms(gated, SR, 200.0, 2000.0), _band_rms(base, SR, 200.0, 2000.0), rtol=0.05)
    assert report["bands_released"] >= 1


def test_onset_frames_protected() -> None:
    """Starke Onsets dürfen keine Synthese-Energie bekommen."""
    rng = np.random.default_rng(11)
    t = np.arange(SR * 2) / SR
    base = np.zeros(SR * 2, dtype=np.float32)
    # Pulsfolge: Onsets alle 0.25 s.
    for k in range(8):
        s0 = int(k * 0.25 * SR)
        base[s0 : s0 + 64] = 0.8 * np.hanning(64).astype(np.float32)
    cand = base + (rng.standard_normal(SR * 2) * 0.05).astype(np.float32)
    gated, report = additive_synthesis_gate(cand, base, SR)
    assert report["onset_frames_protected"] > 0
    # In der Onset-Region bleibt die Energie ≈ Baseline (keine Synthese hinzugefügt).
    onset_region = slice(int(0.25 * SR), int(0.25 * SR) + 64)
    e_g = float(np.mean(gated[onset_region] ** 2))
    e_b = float(np.mean(base[onset_region] ** 2))
    assert e_g <= e_b + 1e-3, f"Onset-Region darf nicht lauter werden: {e_g:.6f} vs {e_b:.6f}"


def test_stereo_layout_and_determinism() -> None:
    rng = np.random.default_rng(13)
    base = (rng.standard_normal(SR) * 0.05).astype(np.float32)
    stereo = np.stack([base, base * 0.9]).astype(np.float32)
    cand = np.stack([base + 0.05 * rng.standard_normal(SR).astype(np.float32)] * 2)
    g1, r1 = additive_synthesis_gate(cand, stereo, SR)
    g2, r2 = additive_synthesis_gate(cand, stereo, SR)
    assert np.array_equal(g1, g2) and r1 == r2
    assert g1.shape == stereo.shape
    # (N, C)-Layout wird bedient.
    g_t, _ = additive_synthesis_gate(cand.T, stereo.T, SR)
    assert g_t.shape == stereo.T.shape
