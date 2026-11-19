"""tests/unit/test_phase_23_spectral_repair.py — §v10.700 I1."""

import numpy as np
import pytest

from backend.core.phases.phase_23_spectral_repair import SpectralRepair


@pytest.fixture
def phase():
    return SpectralRepair()


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
        """Kräftiger Rausch-Masker + winziger Defekt-Burst in der Mitte."""
        sr = 48000
        rng = np.random.RandomState(1)
        x = (rng.randn(sr) * 0.05).astype(np.float32)  # Masker
        c = sr // 2
        x[c : c + 64] += amp  # winziger (subaudibler) Defekt
        return x

    def test_subaudible_defect_skippable_mask_empty(self, phase):
        """Subaudibler Defekt ⇒ Builder liefert leere Maske (Dry-Passthrough)."""
        x = self._masked_defect_audio(amp=0.001)
        sr = 48000
        mask, coverage = phase._build_defect_locality_profile(
            len(x),
            sr,
            {"aliasing": [(0.49, 0.51)]},
            defect_event_metadata=None,
            audio=x,
        )
        # subaudibel ⇒ Region wird übersprungen ⇒ Maske bleibt leer ⇒ Fallback "repariere überall"
        assert np.all(mask == 1.0), "Subaudibler Defekt muss zu leerer (repariere-überall) Maske führen"

    def test_audible_defect_not_skipped_mask_localized(self, phase):
        """Hörbarer Defekt ⇒ Builder erzeugt lokale Maske (nicht überall 1)."""
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        x = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        c = sr // 2
        x[c : c + 64] += 1.0  # lauter Klick
        mask, coverage = phase._build_defect_locality_profile(
            len(x),
            sr,
            {"aliasing": [(0.49, 0.51)]},
            defect_event_metadata=None,
            audio=x,
        )
        # hörbar ⇒ Region wird NICHT übersprungen ⇒ Maske lokal (Mittelwert < 1)
        assert float(np.mean(mask)) < 1.0, "Hörbarer Defekt muss eine lokale Maske (Mittelwert < 1) erzeugen"

    def test_fail_open_on_error(self, phase, monkeypatch):
        """§V6 (copilot-instructions.md): Gate-Fehler darf nicht blockieren (fail-open)."""
        import backend.core.dsp.masking_model as mm

        monkeypatch.setattr(
            mm, "compute_masking_threshold_db", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaputt"))
        )
        sr = 48000
        x = (np.random.RandomState(2).randn(sr) * 0.05).astype(np.float32)
        mask, coverage = phase._build_defect_locality_profile(
            len(x), sr, {"aliasing": [(0.49, 0.51)]}, defect_event_metadata=None, audio=x
        )
        # fail-open: Reparatur freigegeben ⇒ Maske lokal (nicht blockiert)
        assert coverage >= 0.0


class TestGateSpikeTemporalCompactness:
    """§SR-TG: z-score-Spikes mit zu langem zeitlichem Run sind Musik, kein Crackle.

    Produktionsbefund vinyl/1970: §v10.709 Quality-Degradation #1 nach
    phase_23 (timbre_authentizitaet) — anhaltende Obertöne/Becken-Wash wurden
    als Spikes detektiert und durch Interpolation ersetzt.
    """

    def test_sustained_run_removed_entirely(self, phase):
        mask = np.zeros((8, 60), dtype=bool)
        mask[3, 20:45] = True  # 25-Frame-Run (anhaltender Inhalt)
        out = phase._gate_spike_temporal_compactness(mask, max_run_frames=24)
        assert not out[3, 20:45].any()  # kompletter Run entfernt (auch Onset)

    def test_dense_crackle_train_survives(self, phase):
        # Nutzerbefund: dichtes Knistern (Trains) muss repariert werden —
        # 12-Frame-Run bleibt bei max_run_frames=24 erhalten.
        mask = np.zeros((8, 60), dtype=bool)
        mask[6, 10:22] = True  # 12-Frame-Knistern-Train
        out = phase._gate_spike_temporal_compactness(mask, max_run_frames=24)
        assert out[6, 10:22].all()

    def test_compact_spike_survives(self, phase):
        mask = np.zeros((8, 60), dtype=bool)
        mask[2, 10:12] = True  # 2-Frame-Impuls
        mask[5, 30:34] = True  # 4-Frame-Impuls (Grenze)
        out = phase._gate_spike_temporal_compactness(mask, max_run_frames=4)
        assert out[2, 10:12].all()
        assert out[5, 30:34].all()

    def test_gap_splits_runs(self, phase):
        mask = np.zeros((8, 60), dtype=bool)
        mask[4, 5:10] = True  # 5 Frames — Run zu lang
        mask[4, 12:15] = True  # 3 Frames — ok
        out = phase._gate_spike_temporal_compactness(mask, max_run_frames=4)
        assert not out[4, 5:10].any()
        assert out[4, 12:15].all()

    def test_deterministic_and_ndim_guard(self, phase):
        rng = np.random.RandomState(7)
        mask = rng.rand(6, 50) > 0.7
        out1 = phase._gate_spike_temporal_compactness(mask)
        out2 = phase._gate_spike_temporal_compactness(mask)
        assert np.array_equal(out1, out2)
        one_d = np.ones(20, dtype=bool)
        assert np.array_equal(phase._gate_spike_temporal_compactness(one_d), one_d)
