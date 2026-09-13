"""§Witness-SOTA IN-V1+V2: Tests für das Inpainting-Naht-Gate (phase_55).

Roadmap docs/TODOS_SOTA_ROADMAP.md SOTA-IN-V1+V2: additive_synthesis_gate-
Prinzip (Band-weise Additive-Kappung) + C2-artiges Hüllkurven-Alignment um
die ML-/DSP-Inpainting-Kandidaten in phase_55.
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.inpainting_seam_gate import inpainting_seam_gate

SR = 44100


def _tone_gap_context(amp: float = 0.3, freq: float = 440.0, gap_s: float = 0.1) -> tuple[np.ndarray, int, int]:
    """2 s Sinus-Kontext mit einer stillen Lücke in der Mitte."""
    t = np.arange(SR * 2) / SR
    ch = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    start, end = SR // 2, SR // 2 + int(gap_s * SR)
    ch[start:end] = 0.0
    return ch, start, end


def test_seam_gate_aligns_candidate_to_context_envelope() -> None:
    """IN-V2: 3× lauterer Kandidat wird an die Kontext-Hüllkurve angeglichen —
    an beiden Nähten keine Pegelstufe mehr."""
    ch, start, end = _tone_gap_context(amp=0.3)
    cand = (0.9 * np.sin(2 * np.pi * 440 * np.arange(end - start) / SR)).astype(np.float32)
    gated, report = inpainting_seam_gate(ch, cand, start, end, SR)

    assert report["applied"] is True
    assert gated.shape == (end - start,)
    assert np.all(np.isfinite(gated))
    # Kandidat war 3× lauter als der Kontext → Peak muss auf Kontext-Niveau fallen.
    assert float(np.max(np.abs(gated))) < 0.45
    # Naht-Kontinuität: RMS des gated Fill-Rands ≈ Kontext-RMS an der Naht.
    frame = SR // 100
    ctx_rms = float(np.sqrt(np.mean(ch[start - frame : start] ** 2)))
    gated_edge_rms = float(np.sqrt(np.mean(gated[:frame] ** 2)))
    assert abs(gated_edge_rms - ctx_rms) < 0.5 * ctx_rms


def test_seam_gate_caps_fill_energy_at_context_plus_margin() -> None:
    """IN-V1: 10× lauterer Kandidat wird band-weise auf Kontext + 6 dB gedeckelt."""
    ch, start, end = _tone_gap_context(amp=0.3)
    cand = (3.0 * np.sin(2 * np.pi * 440 * np.arange(end - start) / SR)).astype(np.float32)
    gated, _ = inpainting_seam_gate(ch, cand, start, end, SR)
    # Kontext-Peak 0.3 → Kappe bei +6 dB ≈ 0.6 (Alignment-Floor wirkt zusätzlich).
    assert float(np.max(np.abs(gated))) <= 0.65


def test_seam_gate_silent_context_damps_fill() -> None:
    """Stiller Kontext: Fill wird deutlich gedämpft (nie lauter als Kontext + Margin)."""
    ch = np.zeros(SR * 2, dtype=np.float32)
    start, end = SR // 2, SR // 2 + SR // 10
    cand = (0.9 * np.sin(2 * np.pi * 440 * np.arange(end - start) / SR)).astype(np.float32)
    gated, _ = inpainting_seam_gate(ch, cand, start, end, SR)
    assert float(np.max(np.abs(gated))) <= 0.9 * 0.25  # Alignment-Floor 0.2 + Marge


def test_seam_gate_passthrough_for_aligned_candidate() -> None:
    """Bereits passender Kandidat: nur minimale Änderung (Gain ≈ 1, Kappe ≈ 1)."""
    ch, start, end = _tone_gap_context(amp=0.3)
    cand = (0.3 * np.sin(2 * np.pi * 440 * np.arange(end - start) / SR)).astype(np.float32)
    gated, report = inpainting_seam_gate(ch, cand, start, end, SR)
    # Alignment-Gain ≈ 1 → Peak bleibt ≈ Kontext-Niveau (Kappe +6 dB greift nicht).
    assert float(np.max(np.abs(gated))) < 0.45
    assert abs(float(report.get("align_gain_left", 1.0)) - 1.0) < 0.25


def test_seam_gate_deterministic() -> None:
    """Determinismus §G5 (copilot-instructions.md)."""
    ch, start, end = _tone_gap_context()
    rng = np.random.default_rng(7)
    cand = rng.standard_normal(end - start).astype(np.float32) * 0.5
    a, _ = inpainting_seam_gate(ch, cand, start, end, SR)
    b, _ = inpainting_seam_gate(ch, cand, start, end, SR)
    assert np.array_equal(a, b)


def test_seam_gate_short_gap_returns_unchanged() -> None:
    """Gaps < 8 Samples: Gate nicht anwendbar — Kandidat unverändert."""
    ch = np.zeros(SR, dtype=np.float32)
    cand = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    gated, report = inpainting_seam_gate(ch, cand, 100, 103, SR)
    assert report["applied"] is False
    assert np.array_equal(gated, cand)


def test_seam_gate_nan_safe() -> None:
    """NaN/Inf im Kandidaten werden zu 0 — kein NaN-Output (§0a)."""
    ch, start, end = _tone_gap_context()
    cand = np.full(end - start, np.nan, dtype=np.float32)
    gated, _ = inpainting_seam_gate(ch, cand, start, end, SR)
    assert np.all(np.isfinite(gated))
