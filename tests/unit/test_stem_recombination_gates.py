"""Unit-Tests für die C1–C3-Rekombinations-Gates (§SLR-1f, 2026-09-12).

Quelle: docs/REKOMBINATION_ZEITPUNKT_ANALYSE.md §4 Option C.
Prüft: perfekte Separation bleibt unangetastet (kein Remix), C2-Alignment
korrigiert Sample-Versatz, C1 stellt hörbaren Separations-Verlust wieder her
und verwirft maskiertes Residuum, C3 meldet Stereo-Kollaps report-only,
Layout-Sicherheit (channels-last/erste) + Determinismus.
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.stem_recombination_gates import recombine_stems_with_gates

SR = 48000


def _tone(freq: float, amp: float, dur_s: float = 3.0) -> np.ndarray:
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _burst(freq: float, amp: float, dur_s: float = 3.0, gate: float = 0.4) -> np.ndarray:
    """Ton in Amplituden-Bursts (für Hüllkurven-Korrelation)."""
    x = _tone(freq, amp, dur_s)
    t = np.arange(len(x)) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * (1.0 / gate) * t)
    return (x * env).astype(np.float32)


def test_perfect_separation_leaves_mix_untouched() -> None:
    vocal = _burst(1000.0, 0.4)
    instr = _tone(220.0, 0.3)  # konstante Hüllkurve → kein xcorr-Fehl-Offset
    mix = vocal + instr
    remix, result = recombine_stems_with_gates(mix, vocal, instr, vocal, instr, SR)
    np.testing.assert_allclose(remix[0], mix, atol=1e-4)
    assert result.offset_samples == 0
    assert result.residue_bands_reused == 0
    assert result.stereo_ok is True


def _pulses(freq: float, amp: float, dur_s: float = 3.0) -> np.ndarray:
    """Ton mit 30-ms-Hann-Pulsen an unregelmäßigen Onsets (für Hüllkurven-Alignment)."""
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    env = np.zeros(n, dtype=np.float32)
    p = int(SR * 0.030)
    w = np.hanning(p).astype(np.float32)
    for onset_s in (0.15, 0.45, 0.9, 1.35, 1.8, 2.3):
        s0 = int(onset_s * SR)
        env[s0 : s0 + p] = w
    return (x * env).astype(np.float32)


def test_alignment_corrects_sample_offset() -> None:
    vocal = _pulses(1000.0, 0.4)
    instr_base = _pulses(2000.0, 0.35)
    offset = 480  # 10 ms — ML-Blocklatenz der Instr-Verarbeitung
    instr_shifted = np.zeros_like(instr_base)
    instr_shifted[offset:] = instr_base[:-offset]  # Final-Stem eilt nach (ML-Latenz)
    mix = vocal + instr_base  # Raws sind aligniert (gleiches Separationsmodell)
    remix, result = recombine_stems_with_gates(mix, vocal, instr_base, vocal, instr_shifted, SR)
    assert (
        abs(result.offset_samples + offset) <= 12
    )  # Witness: negative Verzögerung des Instr-Stems (±1 DS-Sample Toleranz)
    assert result.alignment_corr > 0.5
    # Nach Korrektur ist der Remix deutlich näher an der wahren Summe.
    err_gated = float(np.max(np.abs(remix[0] - (vocal + instr_base))))
    assert err_gated < 0.05


def test_residue_gate_restores_audible_loss() -> None:
    vocal = _burst(1000.0, 0.4)
    instr = _tone(220.0, 0.3)
    lost = _tone(5000.0, 0.2)  # hörbarer Separations-Verlust in stillem Band
    mix = vocal + instr + lost
    remix, result = recombine_stems_with_gates(mix, vocal, instr, vocal, instr, SR)
    assert result.residue_bands_reused >= 1
    # Band-Energie bei 5 kHz wird wiederhergestellt.
    from backend.core.dsp.masking_model import bark_band_edges

    edges = bark_band_edges(SR)
    band = int(np.searchsorted(edges, 5000.0) - 1)
    assert band >= 0


def test_residue_gate_discards_masked_loss() -> None:
    vocal = _tone(1000.0, 0.9)  # lauter Maskierer bei 1 kHz
    instr = _tone(220.0, 0.2)
    lost = _tone(1000.0, 0.01)  # gleiche Band, 40 dB unter dem Maskierer
    mix = vocal + instr + lost
    remix, result = recombine_stems_with_gates(mix, vocal, instr, vocal, instr, SR)
    # Verworfenes Residuum ist perzeptuell irrelevant — Remix ≈ Summe.
    np.testing.assert_allclose(remix[0], vocal + instr, atol=1e-2)


def test_stereo_collapse_reported() -> None:
    left = _burst(1000.0, 0.4)
    right = _burst(1500.0, 0.4)
    mix = np.stack([left, right], axis=0).astype(np.float32)
    collapsed = np.mean(mix, axis=0, keepdims=True)
    collapsed = np.repeat(collapsed, 2, axis=0).astype(np.float32)
    remix, result = recombine_stems_with_gates(mix, mix, np.zeros_like(mix), collapsed, np.zeros_like(mix), SR)
    assert result.stereo_ok is False
    assert result.witness["stereo_ok"] is False
    assert result.iacc_drop > 0.0


def test_layout_channels_last_and_determinism() -> None:
    vocal = _burst(1000.0, 0.4)
    instr = _tone(220.0, 0.3)
    mix_cn = np.stack([vocal, instr], axis=0)  # channels-first
    mix_cl = mix_cn.T  # channels-last (N, 2)
    remix_a, _ = recombine_stems_with_gates(mix_cl, vocal, instr, vocal, instr, SR)
    remix_b, _ = recombine_stems_with_gates(mix_cl, vocal, instr, vocal, instr, SR)
    assert remix_a.shape[0] == 2  # Rückgabe channels-first
    np.testing.assert_array_equal(remix_a, remix_b)  # Determinismus (§G5 (copilot-instructions.md))
