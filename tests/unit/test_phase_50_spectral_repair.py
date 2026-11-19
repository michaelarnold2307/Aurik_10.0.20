"""tests/unit/test_phase_50_spectral_repair.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_50_spectral_repair import SpectralRepairPhase


@pytest.fixture
def phase():
    return SpectralRepairPhase()


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


# ---------------------------------------------------------------------------
# §SOTA-PSY-A1: subaudible Defekte werden übersprungen (Builder-Maske leer)
# ---------------------------------------------------------------------------


class TestPsyA1SubaudibleGate:
    def _masked_defect_audio(self, amp: float = 0.001) -> np.ndarray:
        sr = 48000
        rng = np.random.RandomState(1)
        x = (rng.randn(sr) * 0.05).astype(np.float32)  # Masker
        c = sr // 2
        x[c : c + 64] += amp  # winziger (subaudibler) Defekt
        return x

    def test_subaudible_defect_skippable_mask_empty(self, phase):
        x = self._masked_defect_audio(amp=0.001)
        sr = 48000
        mask, coverage = phase._build_locality_profile(
            len(x), sr, {"aliasing": [(0.49, 0.51)]}, event_metadata=None, audio=x
        )
        assert np.all(mask == 1.0), "Subaudibler Defekt muss zu leerer (repariere-überall) Maske führen"

    def test_audible_defect_not_skipped_mask_localized(self, phase):
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        x = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        c = sr // 2
        x[c : c + 64] += 1.0
        mask, coverage = phase._build_locality_profile(
            len(x), sr, {"aliasing": [(0.49, 0.51)]}, event_metadata=None, audio=x
        )
        assert float(np.mean(mask)) < 1.0, "Hörbarer Defekt muss eine lokale Maske (Mittelwert < 1) erzeugen"

    def test_fail_open_on_error(self, phase, monkeypatch):
        import backend.core.dsp.masking_model as mm

        monkeypatch.setattr(
            mm, "compute_masking_threshold_db", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaputt"))
        )
        sr = 48000
        x = (np.random.RandomState(2).randn(sr) * 0.05).astype(np.float32)
        mask, coverage = phase._build_locality_profile(
            len(x), sr, {"aliasing": [(0.49, 0.51)]}, event_metadata=None, audio=x
        )
        assert coverage >= 0.0
