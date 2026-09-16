"""tests/unit/test_phase_55_diffusion_inpainting.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_55_diffusion_inpainting import DiffusionInpaintingPhase


@pytest.fixture
def phase():
    return DiffusionInpaintingPhase()


@pytest.fixture
def audio():
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5 + rng.randn(48000) * 0.01).astype(np.float32)


def test_returns_ndarray(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert isinstance(result.audio, np.ndarray)


def test_no_nan_inf(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert np.isfinite(result.audio).all()


def test_not_silent(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert float(np.sqrt(np.mean(result.audio**2))) > 1e-10


def test_length_preserved(phase, audio):
    result = phase.process(audio, sample_rate=48000, material_type="vinyl")
    assert len(result.audio) == len(audio)


# ─── §SOTA-VOCAL-INPAINT-S3: DiffWave-Vokal-Verdrahtung ────────────────────


def test_try_diffwave_vocal_not_ready_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ohne F1-Checkpoint bleibt die bisherige Drosselung der Status quo."""
    from backend.core.phases import phase_55_diffusion_inpainting as p55

    monkeypatch.setattr("backend.core.dsp.diffwave_torch_inpaint.diffwave_vocal_ready", lambda: False)
    x = np.zeros(48000, dtype=np.float32)
    assert p55._try_diffwave_vocal(x, 1000, 4000, 48000) is None


def test_try_diffwave_vocal_uses_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mit Checkpoint liefert der Pfad das deterministische Fill-Segment."""
    from backend.core.phases import phase_55_diffusion_inpainting as p55

    fake_fill = np.linspace(0.01, 0.05, 3000, dtype=np.float32)
    monkeypatch.setattr("backend.core.dsp.diffwave_torch_inpaint.diffwave_vocal_ready", lambda: True)
    monkeypatch.setattr("backend.core.dsp.diffwave_torch_inpaint.diffwave_inpaint_gap", lambda ch, s, e, sr: fake_fill)
    x = np.zeros(48000, dtype=np.float32)
    out = p55._try_diffwave_vocal(x, 1000, 4000, 48000)
    assert out is not None
    assert out.shape == (3000,)
    assert np.array_equal(out, fake_fill)


def test_try_diffwave_vocal_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fehler im Modul ⇒ None (§V6 (copilot-instructions.md)-Fallback)."""
    from backend.core.phases import phase_55_diffusion_inpainting as p55

    monkeypatch.setattr("backend.core.dsp.diffwave_torch_inpaint.diffwave_vocal_ready", lambda: True)
    monkeypatch.setattr(
        "backend.core.dsp.diffwave_torch_inpaint.diffwave_inpaint_gap",
        lambda ch, s, e, sr: (_ for _ in ()).throw(RuntimeError("kaputt")),
    )
    x = np.zeros(48000, dtype=np.float32)
    assert p55._try_diffwave_vocal(x, 1000, 4000, 48000) is None


def test_derive_safe_strength_unthrottled_when_ready(phase) -> None:
    """§Q11 (2026-09-15): Kaskade ≥ 0 dB auf Vokal-Lücken → Gesangs-Drosselung entfällt.

    Nur Analog-Faktor (0.85) bleibt als moderate Reduktion für analoges Material.
    Die IN-V1/V2-Naht-Gates + Damage-Guard schützen vor Overfill-Artefakten.
    """
    # Analog + Gesang: nur Analog-Faktor (0.85), keine Gesangs-Drosselung mehr
    analog_vocal = phase._derive_safe_inpainting_strength(1.0, "vinyl", 0.5, vocal_fill_ready=True)
    assert analog_vocal == pytest.approx(0.85)
    # Analog + Gesang mit explizitem False (Default ist True): ebenfalls keine Drosselung
    analog_vocal_legacy = phase._derive_safe_inpainting_strength(1.0, "vinyl", 0.5, vocal_fill_ready=False)
    assert analog_vocal_legacy == pytest.approx(0.85)
    # Ohne Gesang: Analog-Faktor unverändert
    no_vocals = phase._derive_safe_inpainting_strength(1.0, "vinyl", 0.1, vocal_fill_ready=True)
    assert no_vocals == pytest.approx(0.85)
    # Nicht-Analog + Gesang: keine Reduktion
    digital_vocal = phase._derive_safe_inpainting_strength(1.0, "cd_digital", 0.9, vocal_fill_ready=True)
    assert digital_vocal == pytest.approx(1.0)


def test_vocal_gap_cqtdiff_first_s4(monkeypatch: pytest.MonkeyPatch) -> None:
    """§SOTA-VOCAL-INPAINT S4 (2026-09-16): Gesangslücken (≥ 50 ms) versuchen
    CQTdiff+ VOR FlowMatching (Evidenz: S4 mean −2,76 dB vs. Q11 0,0 dB)."""
    from backend.core.phases import phase_55_diffusion_inpainting as p55

    calls: list[str] = []
    fake = np.full(14400, 0.1, dtype=np.float32)

    def _fake_cqtdiff(channel, start, end, sr):
        calls.append("cqtdiff")
        return fake

    def _fake_flow(channel, start, end, sr, goal_weights=None, restorability_score=65.0):
        calls.append("flow")
        return fake

    monkeypatch.setattr(p55, "_try_cqtdiff_plus_plugin", _fake_cqtdiff)
    monkeypatch.setattr(p55, "_try_flow_matching_plugin", _fake_flow)
    monkeypatch.setattr(
        "backend.core.dsp.audibility_gate.defect_audibility",
        lambda *a, **k: {"skippable": False},
    )

    sr = 48000
    gap = (sr // 2, sr // 2 + 14400)
    t = np.linspace(0, 2, sr * 2, endpoint=False, dtype=np.float32)
    channel = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    channel[gap[0] : gap[1]] = 0.0  # echte Lücke (Stille-Region)
    _, stats = p55._process_channel(
        channel,
        sr,
        20.0,
        precomputed_gaps=[gap],
        wall_budget_s=60.0,
        vocals_confidence=0.60,
    )
    assert calls and calls[0] == "cqtdiff"
    assert "flow" not in calls
    assert stats.get("s4_vocal_cqtdiff_first", 0) == 1


def test_non_vocal_gap_keeps_flow_matching_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Status quo ohne Vokal-Konfidenz: FlowMatching bleibt TIER-0 erster Versuch."""
    from backend.core.phases import phase_55_diffusion_inpainting as p55

    calls: list[str] = []
    fake = np.full(14400, 0.1, dtype=np.float32)

    def _fake_cqtdiff(channel, start, end, sr):
        calls.append("cqtdiff")
        return fake

    def _fake_flow(channel, start, end, sr, goal_weights=None, restorability_score=65.0):
        calls.append("flow")
        return fake

    monkeypatch.setattr(p55, "_try_cqtdiff_plus_plugin", _fake_cqtdiff)
    monkeypatch.setattr(p55, "_try_flow_matching_plugin", _fake_flow)
    monkeypatch.setattr(
        "backend.core.dsp.audibility_gate.defect_audibility",
        lambda *a, **k: {"skippable": False},
    )

    sr = 48000
    gap = (sr // 2, sr // 2 + 14400)
    t = np.linspace(0, 2, sr * 2, endpoint=False, dtype=np.float32)
    channel = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    channel[gap[0] : gap[1]] = 0.0  # echte Lücke (Stille-Region)
    _, stats = p55._process_channel(
        channel,
        sr,
        20.0,
        precomputed_gaps=[gap],
        wall_budget_s=60.0,
        vocals_confidence=0.0,
    )
    assert calls and calls[0] == "flow"
    assert "cqtdiff" not in calls
    assert stats.get("flow_matching_tier0_used", 0) == 1
