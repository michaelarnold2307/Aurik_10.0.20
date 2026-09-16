"""§SOTA-HR-V1 (Q6/F3, 2026-09-15) — BigVGAN-Aktivierungsvertrag: Tests.

Deckt die CPU-Vorbereitung des GPU-Buildouts:
- ``bigvgan_v2_ready()`` ist fail-closed (False ohne F3-Validierungs-Flag)
- ``hr_v1_activation_status()`` meldet Grund + Checkpoint-Präsenz (ZEUGE)
- Flag ON + Checkpoint ⇒ ready() True (Schaltstelle funktioniert)
- phase_07 meldet ``hr_v1``-Metadatum mit attempted=False (Status quo unverändert)

Autor: Aurik Testing Team
"""

import numpy as np

import plugins.bigvgan_v2_plugin as bvg


class TestActivationContract:
    def test_ready_false_by_default(self):
        assert bvg.bigvgan_v2_ready() is False

    def test_status_reports_f3_pending(self):
        status = bvg.hr_v1_activation_status()
        assert status["activated"] is False
        assert status["reason"] == "f3_validation_pending"
        # bigvgan_v2.pth ist lokal vorhanden — der Checkpoint allein reicht NICHT.
        assert status["checkpoint_present"] is True

    def test_flag_is_single_switch(self):
        old = bvg.BIGVGAN_V2_HR_ACTIVATED
        try:
            bvg.BIGVGAN_V2_HR_ACTIVATED = True
            assert bvg.bigvgan_v2_ready() is True
            assert bvg.hr_v1_activation_status()["reason"] == "activated"
        finally:
            bvg.BIGVGAN_V2_HR_ACTIVATED = old


class TestPhase07HrV1Witness:
    def test_phase07_reports_hr_v1_not_attempted(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

        t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
        x = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        res = HarmonicRestorationPhase().process(x, sample_rate=48000, material_type=MaterialType.VINYL)
        hr = res.metadata.get("hr_v1")
        assert hr is not None, "phase_07 meldet kein hr_v1-Metadatum"
        assert hr.get("attempted") is False  # Aktivierungsvertrag: Status quo
        assert np.isfinite(np.asarray(res.audio)).all()


class TestPhase07HrV1ActivatedFallback:
    """Aktivierter Zweig (Flag ON, Checkpoint vorhanden) — ohne GPU:
    Synthese-Fehler und Modell-Leerlauf müssen fail-closed auf DSP fallen."""

    @staticmethod
    def _run_phase07() -> "object":
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

        t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
        x = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        return HarmonicRestorationPhase().process(x, sample_rate=48000, material_type=MaterialType.VINYL)

    def test_synthesis_failure_falls_back_dsp(self, monkeypatch):
        monkeypatch.setattr(bvg, "BIGVGAN_V2_HR_ACTIVATED", True)

        def _raise(*_a, **_k):
            raise RuntimeError("simulierter Synthese-Fehler")

        monkeypatch.setattr(bvg, "synthesize_audio", _raise)
        res = self._run_phase07()
        hr = res.metadata.get("hr_v1")
        assert hr.get("attempted") is True
        assert hr.get("applied") is False  # §V6-fail-closed: DSP-Status quo
        assert np.isfinite(np.asarray(res.audio)).all()

    def test_model_used_none_keeps_dsp(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(bvg, "BIGVGAN_V2_HR_ACTIVATED", True)

        def _none(*_a, **_k):
            return SimpleNamespace(audio=np.zeros(48000, dtype=np.float32), model_used="none", pqs_mos=0.0)

        monkeypatch.setattr(bvg, "synthesize_audio", _none)
        res = self._run_phase07()
        hr = res.metadata.get("hr_v1")
        assert hr.get("attempted") is True
        assert hr.get("applied") is False  # kein Modell-Ergebnis ⇒ kein Eingriff
        assert np.isfinite(np.asarray(res.audio)).all()


class TestHrV1WitnessSpectralRepair:
    """Q6/F3 (2026-09-16): 23/50/03 exportieren den hr_v1-Witness
    (fail-closed — Flag aus ⇒ attempted=False, Status quo unverändert)."""

    @staticmethod
    def _mono() -> np.ndarray:
        t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
        return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_phase_50_reports_hr_v1_not_attempted(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_50_spectral_repair import SpectralRepairPhase

        x = self._mono()
        res = SpectralRepairPhase().process(x, sample_rate=48000, material_type=MaterialType.VINYL)
        hr = res.metadata.get("hr_v1")
        assert hr is not None, "phase_50 meldet kein hr_v1-Metadatum"
        assert hr.get("attempted") is False
        assert np.isfinite(np.asarray(res.audio)).all()

    def test_phase_23_reports_hr_v1_not_attempted(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_23_spectral_repair import SpectralRepair

        x = self._mono()
        res = SpectralRepair().process(x, sample_rate=48000, material_type=MaterialType.VINYL)
        hr = res.metadata.get("hr_v1")
        assert hr is not None, "phase_23 meldet kein hr_v1-Metadatum"
        assert hr.get("attempted") is False
        assert np.isfinite(np.asarray(res.audio)).all()

    def test_phase_03_reports_hr_v1_not_attempted(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_03_denoise import DenoisePhase

        x = self._mono()
        res = DenoisePhase().process(x, sample_rate=48000, material_type=MaterialType.VINYL)
        hr = res.metadata.get("hr_v1")
        assert hr is not None, "phase_03 meldet kein hr_v1-Metadatum"
        assert hr.get("attempted") is False
        assert np.isfinite(np.asarray(res.audio)).all()
