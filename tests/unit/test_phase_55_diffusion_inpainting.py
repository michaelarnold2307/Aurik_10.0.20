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
    """S3: Bei aktivem Finetune-Pfad entfällt die Gesangs-Drosselung (0.78/Cap 0.58)."""
    # Analog + Gesang, ohne Fill: 1.0 × 0.78 × 0.85 = 0.663 → Cap 0.58
    throttled = phase._derive_safe_inpainting_strength(1.0, "vinyl", 0.5, vocal_fill_ready=False)
    assert throttled == pytest.approx(0.58)
    # Mit Fill: nur Analog-Faktor bleibt (0.85), keine Gesangs-Drosselung
    unthrottled = phase._derive_safe_inpainting_strength(1.0, "vinyl", 0.5, vocal_fill_ready=True)
    assert unthrottled == pytest.approx(0.85)
    # Ohne Gesang bleibt alles wie bisher
    assert phase._derive_safe_inpainting_strength(1.0, "vinyl", 0.1, vocal_fill_ready=True) == pytest.approx(0.85)
